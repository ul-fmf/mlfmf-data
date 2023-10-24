from apaa.data.manipulation import (
    prepare_dataset,
    prepare_internal_cv_dataset,
    get_theorems_and_other,
)
import pickle
from apaa.data.structures import AgdaForest, KnowledgeGraph, AgdaDefinition
from apaa.other.helpers import Locations, NodeType
from apaa.preprocessing import create_library_definitions
from main_learner import LOGGER


import os


def load_and_dump_trees(library: str, library_dir: str):
    tree_dump_file = Locations.forest_dump(library)
    if not os.path.exists(tree_dump_file):
        LOGGER.info(f"Constructing forest from files in '{library_dir}'")
        forest = AgdaForest.create_from_files(library_dir)
        LOGGER.info(f"Dumping to '{tree_dump_file}'")
        forest.dump(tree_dump_file)
    else:
        LOGGER.info(f"Loading forest from '{tree_dump_file}'")
        forest = AgdaForest.load(tree_dump_file)
    LOGGER.info("Forest available.")
    return forest


def load_and_dump_tree_unimath():
    return load_and_dump_trees(Locations.NAME_UNIMATH, Locations.SEXP_DIR_UNIMATH)


def load_and_dump_tree_agda_test():
    return load_and_dump_trees(Locations.NAME_AGDA_TEST, Locations.SEXP_DIR_AGDA_TEST)


def load_and_dump_tree_stdlib():
    return load_and_dump_trees(Locations.NAME_STDLIB, Locations.SEXP_DIR_STDLIB)


def do_all_for_library(library: str, library_dir: str, stage: int):
    LOGGER.info(f"Starting {library} from stage {stage}")
    if stage <= 0:
        load_and_dump_trees(library, library_dir)
    if stage <= 1:
        create_library_definitions(library)
    if stage <= 2:
        kg_loc = Locations.knowledge_graph(library)
        if not os.path.exists(kg_loc):
            kg = KnowledgeGraph.create_from_definitions_file(
                library, Locations.definitions_pickled(library)
            )
            kg.dump(kg_loc)
        g_loc = Locations.knowledge_graph_pure(library)
        if not os.path.exists(g_loc):
            kg = KnowledgeGraph.load(kg_loc)
            kg.dump_pure(g_loc)


def do_all_agda_test(stage: int = 0):
    do_all_for_library(Locations.NAME_AGDA_TEST, Locations.SEXP_DIR_AGDA_TEST, stage)


def do_all_lean_test(stage: int = 0):
    do_all_for_library(Locations.NAME_LEAN_TEST, Locations.SEXP_DIR_LEAN_TEST, stage)


def do_all_stdlib(stage: int = 0):
    do_all_for_library(Locations.NAME_STDLIB, Locations.SEXP_DIR_STDLIB, stage)


def do_all_unimath(stage: int = 0):
    do_all_for_library(Locations.NAME_UNIMATH, Locations.SEXP_DIR_UNIMATH, stage)


def create_datasets(libraries: list[str], p_test: float, ps_body_to_keep: list[float]):
    for library in libraries:
        for p_body_to_keep in ps_body_to_keep:
            dataset_file = Locations.dataset(library, p_test, p_body_to_keep)
            if os.path.exists(dataset_file):
                LOGGER.info(
                    f"Dataset for {library} (p_test = {p_test}, "
                    f"p_body_to_keep = {p_body_to_keep}) already exists."
                )
                continue
            LOGGER.info(f"Preparing dataset {dataset_file}")
            # this must be loaded every single time
            kg = KnowledgeGraph.load(Locations.knowledge_graph(library))
            dataset = prepare_dataset(
                kg.graph,
                kg.id_to_definition,
                p_test=p_test,
                p_def_to_keep=p_body_to_keep,
            )
            with open(dataset_file, "wb") as f:
                pickle.dump(dataset, f)


def check_internal_datasets(internal_files: list[str], n_defs: int):
    test_defs_set = set()
    n_warnings = 0
    for internal_file in internal_files:
        with open(internal_file, "rb") as f:
            _, (_, test_defs), _ = pickle.load(f)
        for def_id in test_defs:
            if def_id in test_defs_set:
                n_warnings += 1
                if n_warnings < 10:
                    LOGGER.warning(f"Definition {def_id} is in multiple folds.")
            test_defs_set.add(def_id)
    if len(test_defs_set) != n_defs:
        LOGGER.warning(f"Missing {n_defs - len(test_defs_set)} test defs")
    LOGGER.info(f"Tests passed with {n_warnings} warnings.")


def create_internal_cv_dataset(
    libraries: list[str], n_folds: int, ps_body_to_keep: list[float]
):
    import itertools

    internal_files = []
    n_defs = -1
    for lib, p_body, fold in itertools.product(
        libraries, ps_body_to_keep, range(n_folds)
    ):
        lib_internal = Locations.library_name_to_internal(lib, n_folds, fold)
        dataset_file = Locations.dataset(lib, 0.2, p_body)
        internal_dataset_file = Locations.dataset(lib_internal, 0.2, p_body)
        internal_pure_kg = Locations.knowledge_graph_pure(lib_internal)
        if os.path.exists(internal_dataset_file) and os.path.exists(internal_pure_kg):
            LOGGER.info(f"Dataset '{internal_dataset_file}' already exists.")
            continue
        elif os.path.exists(internal_dataset_file):
            LOGGER.info("Dataset exists, but not pure KG. Creating it.")
            with open(dataset_file, "rb") as f:
                graph, _, _ = pickle.load(f)
            with open(internal_pure_kg, "wb") as f:
                pickle.dump(graph, f)
            continue
        LOGGER.info(f"Preparing dataset {internal_dataset_file}")
        # this must be loaded every single time
        dataset_file = Locations.dataset(lib, 0.2, p_body)
        if not os.path.exists(dataset_file):
            raise ValueError(f"Dataset {dataset_file} does not exist.")
        with open(dataset_file, "rb") as f:
            train_graph, (train_defs, external_test_defs), _ = pickle.load(f)

        theorem_like_tag = NodeType.get_theorem_like_tag(train_graph)
        ids_in_order = sorted(train_defs)
        definitions_ids, _ = get_theorems_and_other(
            ids_in_order, train_defs, theorem_like_tag
        )
        if n_defs == -1:
            n_defs = len(definitions_ids)
        else:
            assert n_defs == len(definitions_ids)
        internal_dataset = prepare_internal_cv_dataset(
            train_graph, definitions_ids, train_defs, fold, n_folds, p_body, 12345
        )
        internal_dataset[1][0].update(external_test_defs)  # add them to train ...
        # dump the dataset
        with open(internal_dataset_file, "wb") as f:
            pickle.dump(internal_dataset, f)
        internal_files.append(internal_dataset_file)
        # dump the graph: this is the same graph for every fold,
        # but makes life much easier later
        with open(dataset_file, "rb") as f:
            graph, _, _ = pickle.load(f)
        with open(internal_pure_kg, "wb") as f:
            pickle.dump(graph, f)
    check_internal_datasets(internal_files, n_defs)


def do_all_and_datasets():
    do_all_agda_test(0)
    do_all_unimath(0)
    do_all_stdlib(0)
    create_datasets(
        [
            Locations.NAME_AGDA_TEST,
            Locations.NAME_LEAN_TEST,
            Locations.NAME_STDLIB,
            Locations.NAME_UNIMATH,
        ][:1],
        0.2,
        [0.0, 0.1, 0.25, 0.5, 0.75, 0.9],
    )


def do_efficient_for_lean_test():
    AgdaDefinition.create_from_files(
        Locations.SEXP_DIR_LEAN_TEST, Locations.dag_dir(Locations.NAME_LEAN_TEST)
    )


def do_efficient_all(libs: list[str]):
    pairs = [
        (Locations.SEXP_DIR_AGDA_TEST, Locations.NAME_AGDA_TEST),
        (Locations.SEXP_DIR_LEAN_TEST, Locations.NAME_LEAN_TEST),
        (Locations.SEXP_DIR_STDLIB, Locations.NAME_STDLIB),
        (Locations.SEXP_DIR_UNIMATH, Locations.NAME_UNIMATH),
        (Locations.SEXP_DIR_TYPE_TOPOLOGY, Locations.NAME_TYPE_TOPOLOGY),
        (Locations.SEXP_DIR_MATHLIB, Locations.NAME_MATHLIB),
    ]
    for sexp_dir, lib in pairs:
        if lib not in libs:
            continue
        LOGGER.info(f"############################################### Starting {lib}")
        AgdaDefinition.create_from_files(sexp_dir, Locations.dag_dir(lib))


if __name__ == "__main__":
    do_all_agda_test()
