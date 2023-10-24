import networkx as nx
import tqdm
import os
import zipfile
import pickle


class EntryNode:
    def __init__(self, node_id: int, node_type: str, node_description: str):
        self.id = node_id
        self.type = node_type
        self.description = node_description
        self.parents: list["EntryNode"] = []
        self.children: list["EntryNode"] = []

    def __repr__(self) -> str:
        return f"EntryNode({self.id}, {self.type}, {self.description})"

    def add_children(self, children: list["EntryNode"], force: bool = False):
        for child in children:
            if force or child not in self.children:
                self.children.append(child)
            if force or self not in child.parents:
                child.parents.append(self)


class Entry:
    def __init__(self, name: str, root: EntryNode):
        self.name = name
        self.root = root
        self.is_tree = self.compute_is_tree()

    def __repr__(self) -> str:
        return f"Entry({self.name}, {self.root})"

    def compute_is_tree(self):
        stack = [self.root]
        processed_nodes = {self.root.id}
        while stack:
            current = stack.pop()
            if len(current.parents) > 1:
                return False
            for child in current.children:
                c_id = child.id
                if c_id in processed_nodes:
                    continue
                stack.append(child)
                processed_nodes.add(c_id)
        return True
    
    def check(self):
        stack = [self.root]
        n = 0
        while stack:
            current = stack.pop()
            n += 1
            # check whether parents and children are consistent
            for parent in current.parents:
                assert current in parent.children, (current, parent.children)
            for child in current.children:
                assert current in child.parents, (current, child.parents)
            stack.extend(current.children)
        return n


def load_entry(entry_file: str, optimized: bool):
    if optimized:
        return load_entry_optimized(entry_file)
    nodes = {}
    root: EntryNode | None = None
    with open(entry_file, "r", encoding="utf-8") as f:
        f.readline()  # id, type, description, children ids
        for line in f:
            parts = line.split("\t")
            node_id = int(parts[0])
            node_type = parts[1]
            node_description = parts[2]
            node_children = eval(parts[3])
            node = EntryNode(node_id, node_type, node_description)
            nodes[node_id] = (node, node_children)
            if root is None:
                root = node
    assert root is not None
    for node, children_ids in nodes.values():
        node.add_children([nodes[c][0] for c in children_ids])
    name = root.children[0].description
    entry = Entry(name, root)
    return entry


def load_entry_optimized(entry_file: str):
    nodes = {}
    root: EntryNode | None = None
    with open(entry_file, "r", encoding="utf-8") as f:
        f.readline()  # id, type, description, children ids
        for line in f:
            parts = line.split("\t")
            node_id = int(parts[0])
            node_type = parts[1]
            node_description = parts[2]
            node_children = eval(parts[3])
            node = EntryNode(node_id, node_type, node_description)
            nodes[node_id] = (node, node_children)
            if root is None:
                root = node
    assert root is not None
    # for all the nodes, mark which of the children of the root
    # they belong to: REF_TYPE/REF BODY
    node_to_parent = {c: i for i, c in enumerate(nodes[root.id][1])}
    node_to_parent[root.id] = -1
    stack = nodes[root.id][1][:]
    while stack:
        current_id = stack.pop()
        node, children_ids = nodes[current_id] 
        for child_id in children_ids:
            if child_id in node_to_parent:
                continue
            node_to_parent[child_id] = node_to_parent[current_id]
            stack.append(child_id)
    # add children to root
    root.add_children([nodes[c][0] for c in nodes[root.id][1]])
    # add other name nodes to their parents
    for node, _ in nodes.values():
        if node.type == ":name" and node not in root.children:
            parent_index = node_to_parent[node.id]
            root.children[parent_index].add_children([node], True)
    name = root.children[0].description
    entry = Entry(name, root)
    # entry.check()
    return entry


def postprocess_entry(entry: Entry):
    """
    Too slow.
    """
    stack = [entry.root]
    processed = set()
    while stack:
        current = stack.pop()
        if current.id in processed:
            continue
        processed.add(current.id)
        stack.extend(current.children)
        if not (current in entry.root.children or current.type in [":name", ":type", ":entry"]):
            for parent in current.parents:
                parent.children.remove(current)
            for child in current.children:
                child.parents.remove(current)
                for parent in current.parents:
                    if parent not in child.parents:
                        child.parents.append(parent)
                        parent.children.append(child)


def load_entries(entry_dir: str, optimized: bool) -> list[Entry]:
    entries = []
    print("Loading entries ...")
    for file in tqdm.tqdm(os.listdir(entry_dir)):
        if not file.endswith(".dag") or "scripts.runLinter" in file:
            continue
        entry_file = os.path.join(entry_dir, file)
        entries.append(load_entry(entry_file, optimized))
    print(f"Loaded {len(entries)} entries.")
    return entries


def load_graph(graph_file: str) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    print("Loading network ...")
    with open(graph_file, encoding="utf-8") as f:
        for line in tqdm.tqdm(f):
            parts = line.split("\t")[1:]
            parts = [d.strip() for d in parts]  # check this
            if line.startswith("node"):
                node, properties = parts
                properties = eval(properties)
                graph.add_node(node, **properties)
            else:
                source, sink, edge_type, properties = parts
                properties = eval(properties)
                graph.add_edge(source, sink, edge_type, **properties)
    print(f"Loaded G(V, E) where (|V|, |E|) = ({len(graph.nodes)}, {len(graph.edges)})")
    return graph


def try_unzip(zip_file, entry_dir):
    with zipfile.ZipFile(zip_file, 'r') as z:
        for file_info in tqdm.tqdm(z.infolist()):
            if file_info.filename.endswith('.dag'):
                file_info.filename = os.path.basename(file_info.filename)
                z.extract(file_info, entry_dir)


def load_library(library_name: str, optimized: bool = False) -> tuple[list[Entry], nx.MultiDiGraph]:
    entry_dir = f"{library_name}/entries"
    zip_file = f"{library_name}/entries.zip"
    network_file = f"{library_name}/network.csv"
    if not os.path.exists(entry_dir):
        if os.path.exists(zip_file):
            print(f"Did not found {entry_dir}, but {zip_file} exists. Unzipping ...")
            try_unzip(zip_file, entry_dir)
        else:
            print(f"Did not found {entry_dir} nor {zip_file}.")
    bad = False	
    if not os.path.exists(entry_dir):
        print(f"Did not found {entry_dir}.")
        bad = True
    if not os.path.exists(network_file):
        print(f"Did not found {network_file}.")
        bad = True
    if bad:
        return [], nx.MultiDiGraph()
    return load_entries(entry_dir, optimized), load_graph(network_file)
