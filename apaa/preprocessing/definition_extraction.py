import os

import tqdm

from apaa.data.structures.agda_tree import (
    AgdaDefinition,
    AgdaDefinitionForest,
    AgdaForest,
    AgdaNode,
    AgdaTree,
)
from apaa.other.helpers import Locations, NodeType, Other


LOGGER = Other.create_logger(__name__)


def dump_definition_to_text_file(
    out_dir_text: str, definition_forest: AgdaDefinitionForest
):
    os.makedirs(out_dir_text, exist_ok=True)
    for def_tree in tqdm.tqdm(definition_forest):
        out_file = os.path.basename(def_tree.info)
        assert "." in out_file, out_file
        out_file: str = os.path.join(
            out_dir_text, out_file[: out_file.rfind(".")].replace(".", "_") + ".txt"
        )
        def_name = def_tree.root.children[0].node_description.replace(" ", ".")
        with open(out_file, "a", encoding="utf-8") as f:
            print(def_name, file=f)
            print(def_tree, file=f)
            print("", file=f)


def check_for_nested_definitions(definition_trees: AgdaDefinitionForest):
    """
    We check whether there is an s-expression, such that '(... (:entry x)' where x contains
    another (:entry ...).

    """
    for i, def_tree in tqdm.tqdm(enumerate(definition_trees)):
        n_defs = 0
        current_root = def_tree.root
        while current_root is not None:
            if current_root.node_type == NodeType.ENTRY:
                n_defs += 1
            current_root = current_root.parent
        if n_defs > 1:
            LOGGER.info(
                f"Tree {i} ({def_tree.info}) contains {n_defs - 1} "
                f"sub-definitions and you need better preprocessing"
            )


def extract_definitions(
    forest: AgdaForest, out_file_pickle: str, out_dir_text: str
) -> AgdaForest:
    if os.path.exists(out_file_pickle):
        LOGGER.info(f"Loading definitions from '{out_file_pickle}'")
        definition_forest = AgdaForest.load(out_file_pickle)
    else:
        LOGGER.info("Extracting definitions from forest.")
        definition_trees: list[AgdaDefinition] = []
        tree: AgdaTree
        node: AgdaNode
        for i, tree in tqdm.tqdm(enumerate(forest)):
            is_internal = forest.is_tree_internal(i)
            for node in tree.nodes:
                if node.node_type == NodeType.ENTRY:
                    definition_trees.append(
                        AgdaDefinition(tree.info, node, is_internal)
                    )
        LOGGER.info(f"Found {len(definition_trees)} definitions.")
        definition_forest = AgdaDefinitionForest(definition_trees)
        definition_forest.dump(out_file_pickle)
    dump_definition_to_text_file(out_dir_text, definition_forest)
    LOGGER.info(
        f"Definitions are now present in '{out_dir_text}' and '{out_file_pickle}'"
    )
    return definition_forest


def create_library_definitions(library: str) -> None:
    library_file = Locations.forest_dump(library)
    definitions_pickle_file = Locations.definitions_pickled(library)
    definitions_text_dir = Locations.definitions_text_dir(library)
    LOGGER.info(f"Creating library definitions from '{library_file}'")
    if os.path.exists(definitions_pickle_file) and os.path.exists(definitions_text_dir):
        LOGGER.info(
            f"The definitions were already extracted from '{definitions_pickle_file}'"
        )
        return
    elif not os.path.exists(library_file):
        raise ValueError(f"Create the library file '{library_file}' first.")
    extract_definitions(
        AgdaForest.load(library_file), definitions_pickle_file, definitions_text_dir
    )
