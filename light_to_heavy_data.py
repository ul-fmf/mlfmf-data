from light_weight_data_loader import load_library, EntryNode, Entry
from apaa.other.helpers import NodeType
from apaa.data.structures import AgdaNode, AgdaDefinition, AgdaDefinitionForest, KnowledgeGraph
from apaa.data.manipulation import prepare_dataset

import os
import pickle

import tqdm
import networkx as nx


def create_heavy_entry(entry_node: EntryNode) -> AgdaNode:
    node_type = NodeType(entry_node.type)
    description = entry_node.description
    return AgdaNode(node_type, description, None, [])


def convert_entry(entry: Entry, module_name: str, is_internal: bool) -> AgdaDefinition:
    stack = [entry.root]
    original = {}
    converted = {}
    # convert node by node
    while stack:
        current = stack.pop()
        if current.id in converted:
            continue
        original[current.id] = current
        converted[current.id] = create_heavy_entry(current)
        for child in current.children:
            stack.append(child)
    # connect parents and children
    for node_id, node in original.items():
        converted_node = converted[node_id]
        converted_children = [converted[child.id] for child in node.children]
        AgdaNode.connect_parent_to_children(converted_node, converted_children)
    # create tree that looks like
    #        module node
    #   module name     entry tree
    module_node = AgdaNode(NodeType.MODULE, "", None, [])
    entry_root = converted[entry.root.id]
    module_name_node = AgdaNode(NodeType.MODULE_NAME, module_name, None, [])
    AgdaNode.connect_parent_to_children(module_node, [module_name_node, entry_root])
    tree = AgdaDefinition(entry.name, entry_root, is_internal)
    return tree


def convert_entries(
    entries: list[Entry],
    entry_to_module: dict[str, str],
    entry_to_is_internal: dict[str, bool],
) -> AgdaDefinitionForest:
    trees = []
    for entry in tqdm.tqdm(entries):
        module_name = entry_to_module[entry.name]
        trees.append(convert_entry(entry, module_name, entry_to_is_internal[entry.name]))
    forest = AgdaDefinitionForest(trees)
    return forest


def compute_external_modules(graph: nx.MultiDiGraph) -> set[str]:
    external_modules = set()
    e_source = "external source"
    if e_source not in graph.nodes:
        return external_modules
    stack = [e_source]
    while stack:
        current = stack.pop()
        external_modules.add(current)
        for u, v, key in graph.edges(nbunch=current, keys=True):
            if key == "CONTAINS":
                stack.append(v)
    return {f'"{e}"' for e in external_modules}


def convert_to_heavy_data(library_path: str, optimized: bool):
    lib_name = os.path.basename(library_path)
    entries, graph = load_library(library_path, optimized=optimized)
    # get meta data for every entry
    entry_to_module: dict[str, str] = {}
    for u, v, key in graph.edges(keys=True):
        if key == "DEFINES":
            entry_to_module[f'"{v}"'] = f'"{u}"'
    entry_to_is_internal: dict[str, bool] = {}
    external_modules = compute_external_modules(graph)
    for entry, module in entry_to_module.items():
        entry_to_is_internal[entry] = module not in external_modules
    for entry in entries:
        assert entry.name[0] == entry.name[-1] == '"', entry.name
        assert entry.name in entry_to_module, entry.name
    # do the conversion
    heavy_entries = convert_entries(entries, entry_to_module, entry_to_is_internal)
    # use the standard procedure to obtain everything else:
    # 1. create knowledge graph from the definitions
    # 2. create dataset
    # 3. create train/test set
    ent_loc = os.path.join(library_path, "entries.pkl")
    kg_loc = os.path.join(library_path, "kg.pkl")
    g_loc = os.path.join(library_path, "graph.pkl")
    dataset_loc = os.path.join(library_path, "dataset.pkl")
    heavy_entries.dump(ent_loc)
    kg = KnowledgeGraph.create_from_definitions_file(lib_name, ent_loc)
    kg.dump(kg_loc)
    kg.dump_pure(g_loc)
    dataset = prepare_dataset(
        kg.graph,
        kg.id_to_definition,
        p_test=0.2,
        p_def_to_keep=0.1,
    )
    with open(dataset_loc, "wb") as f:
        pickle.dump(dataset, f)



if __name__ == "__main__":
    lib_loc = r"D:\sexp_dumps\submission\test_agda"  # change this
    convert_to_heavy_data(lib_loc, optimized=True)
