from typing import Any, List, Optional, Dict, Set, Tuple, Iterator, Union
import tqdm
import pickle
import re
import os
import networkx as nx


from apaa.other.helpers import Locations, Other, MyTypes, TextManipulation
from apaa.other.helpers import NodeType


LOGGER = Other.create_logger(__file__)


class AgdaNode:
    _next_id = 0
    _max_depth_ever = 0
    _max_nodes_ever = 0

    def __init__(
        self,
        node_type: NodeType,
        node_description: str,
        parent: Optional["AgdaNode"],
        children: List["AgdaNode"],
    ):
        self._id = AgdaNode._generate_next_id()
        self._node_type: NodeType = node_type
        self._node_description: str = node_description
        self._parent: Optional[AgdaNode] = parent
        self._children: List[AgdaNode] = children

    def __str__(self) -> str:
        return self._str_helper(0)

    def __repr__(self):
        return f"Node({self.node_type}, {self.node_description}, id={self._id})"

    def __lt__(self, other: Any):
        if not isinstance(other, AgdaNode):
            raise ValueError(f"AgdaNode < {other.__class__} !?")
        return self.node_description < other.node_description

    def _str_helper(self, depth: int) -> str:
        if depth > 50:
            children = "<there is more after depth 50 ...>"
        elif self.children:
            children = "\n" + "\n".join(
                child._str_helper(depth + 1) for child in self.children
            )
        else:
            children = ""
        return f"{'  ' * depth}({self.node_type.value} {self._node_description} id={self._id}{children})"

    @property
    def full_text(self) -> str:
        full_text_children = " ".join(map(lambda t: t.full_text, self.children))
        separator = " " if full_text_children else ""
        return (
            f"({self.node_type} {self.node_description}{separator}{full_text_children})"
        )

    @property
    def node_type(self) -> NodeType:
        return self._node_type

    @property
    def node_description(self):
        return self._node_description

    @node_description.setter
    def node_description(self, description: str):
        self._node_description = description

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, parent: Optional["AgdaNode"]):
        if self._parent is not None and parent is not None:
            LOGGER.warning(
                f"{self.name} has two parents: {self._parent.name} and {parent.name}"
            )
        self._parent = parent

    @property
    def children(self):
        return self._children

    @children.setter
    def children(self, children: List["AgdaNode"]):
        self._children = children

    @property
    def n_nodes(self) -> int:
        n = 0
        stack = [self]
        while stack:
            current = stack.pop()
            n += 1
            stack.extend(current.children)
        if n > AgdaNode._max_nodes_ever:
            LOGGER.info(f"New max number of nodes: {n}")
            AgdaNode._max_nodes_ever = n
        return n

    @property
    def nodes(self) -> Iterator["AgdaNode"]:
        stack = [self]
        while stack:
            current = stack.pop()
            yield current
            stack.extend(current.children)

    @property
    def depth(self) -> int:
        stack: list[tuple[int, "AgdaNode"]] = [(1, self)]
        max_seen = 0
        while stack:
            current_depth, current_node = stack.pop()
            max_seen = max(max_seen, current_depth)
            for child in current_node.children:
                stack.append((current_depth + 1, child))
        if max_seen > AgdaNode._max_depth_ever:
            LOGGER.info(f"New max depth: {max_seen}")
            AgdaNode._max_depth_ever = max_seen
        return max_seen

    @property
    def name(self) -> MyTypes.NODE:
        """Does some checks regarding the format of the node description and returns it."""
        if not NodeType.is_name(self.node_type):
            raise ValueError(
                f"Cannot get the name of node of type {self.node_type} (not a name!)"
            )
        if not (self.node_description[0] == self.node_description[-1] == '"'):
            raise ValueError(f"Expected quotation marks in {self.node_description}")
        if (
            self.node_type == NodeType.MODULE_NAME
            and self.node_description.count(" ") != 0
        ):
            raise ValueError(
                f"Did not expect any spaces in :module-name {self.node_description}"
            )
        return self.node_description[1:-1]

    @staticmethod
    def name_to_description(name: str) -> str:
        """Should be the right inverse of description_to_name. Will remove it later ... (id = id)"""
        return f'"{name}"'

    @staticmethod
    def connect_parent_to_children(parent: "AgdaNode", children: List["AgdaNode"]):
        for child in children:
            child.parent = parent
        parent.children = children

    @staticmethod
    def _generate_next_id():
        AgdaNode._next_id += 1
        return AgdaNode._next_id

    def compute_topological_sort(self) -> list["AgdaNode"] | None:
        def walker(node: AgdaNode):
            if node._id in permanent:
                return
            elif node._id in temporary:
                has_cycle[0] = True
                return
            temporary.add(node._id)
            for child in node.children:
                walker(child)
            temporary.remove(node._id)
            permanent.add(node._id)
            topological_order.append(node)

        permanent = set()
        temporary = set()
        has_cycle = [False]
        topological_order = []  # decreasing
        for node in self.nodes:
            if node in permanent:
                continue
            walker(node)
        if has_cycle[0]:
            return None
        return topological_order


class AgdaTree:
    def __init__(self, info: str, root: AgdaNode, show_stats: bool = False, post_process: bool = True):
        self._info = info
        self._root = root
        self._depth = -1
        if post_process:
            self._postprocess(show_stats)

    def _postprocess(self, show_stats: bool):
        if show_stats:
            _ = self.depth
            _ = self.n_nodes
        self._postprocess_nested_module_names()
        self._tree_to_dag()
        # self.compute_topological_sort()
        

    def _tree_to_dag(self):
        """
        Takes care of (:node) and (:ref) nodes.

        """
        entry_nodes = [n for n in self.root.nodes if n.node_type == NodeType.ENTRY]
        for entry_root in entry_nodes:            
            type_root = entry_root.children[1]
            if type_root.children:
                self._tree_to_dag_helper(type_root)
            body_root = entry_root.children[2]
            if body_root.children:
                self._tree_to_dag_helper(body_root.children[0])

    def _tree_to_dag_helper(self, root_node: AgdaNode):
        node_map: dict[str, AgdaNode] = {}
        current_node = root_node
        while current_node.node_type == NodeType.NODE:
            node_map[current_node.node_description] = current_node.children[0]
            current_node = current_node.children[1]
        ref_nodes = [n for n in root_node.nodes if n.node_type == NodeType.REF]
        for ref_node in ref_nodes:
            which_ref = node_map[ref_node.node_description]
            parent_node = ref_node.parent
            assert parent_node is not None
            parent_node.children = [c if c is not ref_node else which_ref for c in parent_node.children]
            ref_node.parent = None
        # remove (:node) nodes: connect current to the parent of root_node (which is the first (:node) node)
        if not node_map:
            return
        parent = root_node.parent
        assert parent is not None
        parent.children = [c if c is not root_node else current_node for c in parent.children]
        for node in node_map.values():
            node.parent = None

    def _postprocess_nested_module_names(self):
        """
        Sometimes, a module is stored as (:module-name (:name "foo.bar.Baz")) instead of
        (:module-name "foo.bar.Baz"). This function fixes this.
        """
        for node in self.nodes:
            if node.node_type == NodeType.MODULE_NAME and not node.node_description:
                assert (
                    len(node.children) == 1
                    and node.children[0].node_type == NodeType.NAME
                    and node.children[0].node_description
                )
                node.node_description = node.children[0].node_description
                node.children[0].parent = None
                node.children = []

    def compute_topological_sort(self) -> list[AgdaNode]:
        """Returns True if the tree contains a cycle."""
        top_order = self.root.compute_topological_sort()
        if top_order is None:
            raise ValueError("The tree contains a cycle.")
        return top_order

    @property
    def info(self):
        return self._info

    @property
    def root(self):
        return self._root

    @property
    def depth(self):
        if self._depth < 0:
            self._depth = self.root.depth
        return self._depth

    @property
    def n_nodes(self):
        return self.root.n_nodes

    @property
    def nodes(self) -> Iterator[AgdaNode]:
        yield from self.root.nodes

    def __str__(self):
        return str(self.root)

    def create_subtree(self, node: AgdaNode):
        return AgdaTree(self.info, node)
    
    @staticmethod
    def is_this_a_quote(raw: str, i: int):
        n_backslash = 0
        while raw[i - 1 - n_backslash] == "\\":
            n_backslash += 1
        return n_backslash % 2 == 0

    @staticmethod
    def create_from_file(file: str) -> "AgdaTree":
        """
        Converts an s-expression (sexp) in the file to a tree.

        We read sexp from left to right. Every time we face a ``(``, a new node starts.
        Every time we face a ``)``, the node ends and we can parse it.
        Note that the children nodes (if any) were necessarily already processed.

        We assume that every node looks like

        ``(tag x1 x2 ... xN child1 child2 ... childM)``

        where N, M >= 0, ``tag`` is a "node type", xi give some
        additional information that may be necessary
        (e.g., ``(:module-name Fully Qualified Path)``),
        and children (if any) are also nodes.

        :param file:
        :param immediate_dump:
        :return: parsed AgdaTree
        """
        with open(file, encoding="utf-8") as f:
            raw = "".join([line.strip() for line in f])
        return AgdaTree.parse_raw_sexp(file, raw)

    @staticmethod
    def parse_raw_sexp(file: str, raw: str):
        stack: List[int] = [-1]
        index_to_end: Dict[int, int] = {}  # i_start: i_end
        index_to_parent: Dict[int, int] = {}  # i_start: i_parent
        index_to_children: Dict[int, List[int]] = {}  # i_start: [i_child1, ...]
        index_to_node: Dict[int, AgdaNode] = {}
        in_string = False
        for i, c in enumerate(raw):
            if c == '"' and AgdaTree.is_this_a_quote(raw, i):
                in_string = not in_string
            if c == "(" and not in_string:
                index_to_parent[i] = stack[-1]
                if stack[-1] not in index_to_children:
                    index_to_children[stack[-1]] = []
                index_to_children[stack[-1]].append(i)
                stack.append(i)
            elif c == ")" and not in_string:
                i_start = stack.pop()
                i_children_start = index_to_children.get(i_start, [])
                i_children = [(i0, index_to_end[i0]) for i0 in i_children_start]
                try:
                    node = AgdaTree.parse_node(raw, i_start, i_children, i)
                except ValueError:
                    LOGGER.error(f"Error in file {file} at position {i}:")
                    raise
                children = [index_to_node[i_child] for i_child in i_children_start]
                AgdaNode.connect_parent_to_children(node, children)
                index_to_node[i_start] = node
                index_to_end[i_start] = i
        assert stack == [-1] and not in_string, (file, stack)
        return AgdaTree(file, index_to_node[0], True)

    @staticmethod
    def _find_entry_intervals(
        raw: str,
    ) -> tuple[tuple[int, int], list[tuple[int, int]]]:
        """
        Finds the starting part (module name) and the sub-s-expressions that belong to entries.
        """
        module_part = None
        entries = []
        # find module
        allowed = [
            tag.value for tag in [NodeType.MODULE, NodeType.MODULE_NAME, NodeType.NAME]
        ]
        if NodeType.ENTRY.value not in raw:
            return (0, len(raw) - 1), []
        for i, c in enumerate(raw):
            if c == "(" and not any(
                raw[i + 1 : i + 1 + len(value)] == value for value in allowed
            ):
                module_part = (0, i - 1)
                break
        if module_part is None:
            raise ValueError(f"No module part found in raw = '{raw[:100]}...'.")
        # find entries
        i_start = -1
        count = -1
        entry = NodeType.ENTRY.value
        in_string = False
        global_count = 0
        for i, c in enumerate(raw):
            if c == '"' and AgdaTree.is_this_a_quote(raw, i):
                in_string = not in_string
            if in_string:
                continue

            if c == "(":
                global_count += 1
            elif c == ")":
                global_count -= 1

            if c == "(" and raw[i + 1 : i + 1 + len(entry)] == entry:
                i_start = i
                count = 1
            elif c == "(" and i_start >= 0:
                count += 1
            elif c == ")" and i_start >= 0:
                count -= 1
                if count == 0:
                    entries.append((i_start, i + 1))
                    i_start = -1
        if global_count != 0:
            raise ValueError(f"global_count = {global_count}, raw = '{raw[:100]}...'")
        return module_part, entries

    @staticmethod
    def parse_node(
        raw: str, i_start: int, i_children: List[Tuple[int, int]], i_end: int
    ):
        if i_children:
            i_end_first_part = i_children[0][0]
            i_end_children = i_children[-1][1] + 1
            if i_end_children < i_end - 1:
                # process what is after the children (anything?)
                last = raw[i_end_children:i_end]
                if last:
                    if "(" in last or ")" in last:
                        raise ValueError(f"Something follows the last child: {last}")
                    else:
                        LOGGER.error(f"Something follows the last child: {last}")
        else:
            i_end_first_part = i_end
        to_process = raw[i_start + 1 : i_end_first_part].strip()
        if not to_process:
            raise ValueError(
                f"Substring raw[{i_start}, {i_end}) of raw =\n"
                f"{raw}\n should be processed, but nothing left."
            )
        node_types, rest = AgdaTree.extract_tags(to_process)
        if len(node_types) != 1:
            LOGGER.warning(
                f"Expected only 1 node type, got {node_types}. Will ignore the rest"
            )
        node_type = node_types[0]
        return AgdaNode(node_type, rest, None, [])

    @staticmethod
    def extract_tags(node_body: str) -> Tuple[List[NodeType], str]:
        node_tags: List[NodeType] = []
        found_any = True
        while node_body and found_any:
            first_part = node_body
            if " " in first_part:
                first_part = first_part[: first_part.find(" ")]
            found_any = False
            for tag in NodeType:
                if first_part == tag.value:
                    node_tags.append(tag)
                    node_body = node_body[len(tag.value) :].strip()
                    found_any = True
                    break
        if not node_tags:
            raise ValueError(f"No existing node types found in '{node_body}'")
        return node_tags, node_body

    @staticmethod
    def pretty_print(file: str, file_out: str = "temp.txt"):
        """
        Naive version of __str__ for this class. Useful for debugging etc.

        :param file:
        :param file_out:
        :return:
        """
        with open(file, encoding="utf-8") as f:
            raw = "".join([line.strip() for line in f])
        indent = -1
        characters: List[str] = []
        for c in raw:
            if c == "(":
                indent += 1
                characters.append(f"\n{indent * '  '}")
            elif c == ")":
                indent -= 1
            characters.append(c)
        with open(file_out, "w", encoding="utf-8") as f:
            print("".join(characters).strip(), file=f)


class AgdaForest:
    def __init__(self, trees: List[AgdaTree], is_internal: List[bool]):
        if len(trees) != len(is_internal):
            raise ValueError(
                "Lists in AgdaForest(trees, is_external) should have the same length!"
            )

        self._trees = trees
        self._is_internal = is_internal
        self._iter_index = 0

    @property
    def trees(self):
        return self._trees

    def is_tree_internal(self, i_tree: int):
        return self._is_internal[i_tree]

    def __len__(self):
        return len(self.trees)

    def __iter__(self):
        self._iter_index = 0
        return self

    def __next__(self) -> AgdaTree:
        if self._iter_index < len(self.trees):
            self._iter_index += 1
            return self.trees[self._iter_index - 1]
        else:
            raise StopIteration

    def dump(self, file: str):
        with open(file, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file: str) -> "AgdaForest":
        with open(file, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def create_from_files(
        sexp_dir: str, source_root_dir: Optional[str] = None
    ) -> "AgdaForest":
        """
        Creates a tree for every ``sexp`` file in the root directory (or any of its sub(sub...)directories).

        :param sexp_dir: path to the directory that contains sexp files
        :param source_root_dir: path to the root directory of (library) sources. If omitted, it defaults to
               ``sexp_dir/..``

        :return: a forest
        """
        if source_root_dir is None:
            source_root_dir = os.path.join(sexp_dir, "..")

        trees: List[AgdaTree] = []
        is_internal: List[bool] = []
        sexp_files = Other.get_all_files(sexp_dir, file_extension="sexp")
        for path in tqdm.tqdm(sexp_files):
            print(path)
            trees.append(AgdaTree.create_from_file(path))
            is_internal.append(AgdaForest._is_sexp_internal(source_root_dir, path))
        return AgdaForest(trees, is_internal)

    @staticmethod
    def _is_sexp_internal(library_root_dir: str, sexp_path: str):
        sexp_file = os.path.basename(sexp_path)
        assert sexp_file.endswith("sexp")
        file_extension = sexp_file[: sexp_file.rfind(".")]
        parts = file_extension.split(".")
        module_name = parts[-1] + "."
        module_dir = os.path.join(library_root_dir, *parts[:-1])
        if not os.path.exists(module_dir):
            return False
        file: str
        for file in os.listdir(module_dir):
            if file.startswith(module_name):
                return True
        LOGGER.warning(f"{sexp_file} appears to be external?")
        return False


class AgdaDefinition(AgdaTree):
    def __init__(self, info: str, root: AgdaNode, is_internal: bool):
        super().__init__(info, root, post_process=False)
        self._module_name = self._find_module_name()
        self._is_internal = is_internal

    @staticmethod
    def create_from_files(
        sexp_dir: str,
        out_dir: str,
        source_root_dir: Optional[str] = None,
    ):
        """
        Creates a tree for every ``sexp`` file in the root directory (or any of its sub(sub...)directories).

        :param sexp_dir: path to the directory that contains sexp files
        :param source_root_dir: path to the root directory of (library) sources. If omitted, it defaults to
               ``sexp_dir/..``

        :return: a forest
        """
        if source_root_dir is None:
            source_root_dir = os.path.join(sexp_dir, "..")

        sexp_files = Other.get_all_files(sexp_dir, file_extension="sexp")
        os.makedirs(out_dir, exist_ok=True)
        for path in tqdm.tqdm(sexp_files):
            out_file = os.path.basename(path)
            out_file = out_file[: out_file.rfind(".")]
            finished = os.path.join(out_dir, out_file + ".done")
            if os.path.exists(finished):
                continue
            LOGGER.info(f"Processing {path}")
            is_internal = AgdaForest._is_sexp_internal(source_root_dir, path)
            for i, raw_sexp in tqdm.tqdm(enumerate(AgdaDefinition.create_from_file(path))):
                
                out_path = os.path.join(
                    out_dir,
                    f"{out_file}_{i:04}.dag",
                )
                if os.path.exists(out_path):
                    LOGGER.info(f"Skipping {out_path} as it already exists.")
                    continue
                tree = AgdaTree.parse_raw_sexp(path, raw_sexp)
                # find a single entry
                the_entry_node = tree.root.children[1]
                assert the_entry_node.node_type == NodeType.ENTRY
                definition = AgdaDefinition(tree.info, the_entry_node, is_internal)
                AgdaDefinition.dump_to_dag_file(definition, out_path)
            with open(finished, "w") as _:
                pass

    @staticmethod
    def create_from_file(file: str) -> Iterator[str]:
        """
        Converts an s-expression (sexp) in the file to a tree.

        We read sexp from left to right. Every time we face a ``(``, a new node starts.
        Every time we face a ``)``, the node ends and we can parse it.
        Note that the children nodes (if any) were necessarily already processed.

        We assume that every node looks like

        ``(tag x1 x2 ... xN child1 child2 ... childM)``

        where N, M >= 0, ``tag`` is a "node type", xi give some
        additional information that may be necessary
        (e.g., ``(:module-name Fully Qualified Path)``),
        and children (if any) are also nodes.

        :param file:
        :param immediate_dump:
        :return: parsed AgdaTree
        """
        with open(file, encoding="utf-8") as f:
            raw = "".join([line.strip() for line in f])

        module_part, entries = AgdaTree._find_entry_intervals(raw)
        module_part_str = raw[module_part[0] : module_part[1]]
        for i_start, i_end in entries:
            raw_sexp = f"{module_part_str} {raw[i_start:i_end]})"
            yield raw_sexp

    @staticmethod
    def dump_to_dag_file(definition: "AgdaDefinition", out_file: str):
        """
        Dumps the definition to a file in the DAG format.
        """
        stack = [definition.root]
        processed_nodes: set[int] = {definition.root._id}
        with open(out_file, "w", encoding="utf-8") as f:
            print(f"NODE ID\tNODE TYPE\tNODE DESCRIPTION\tCHILDREN IDS", file=f)
            while stack:
                current = stack.pop()
                # process
                chidren_ids = [child._id for child in current.children]
                print(f"{current._id}\t{current.node_type.value}\t{current.node_description}\t{chidren_ids}", file=f)
                # stack children
                for child in current.children:
                    if child._id not in processed_nodes:
                        stack.append(child)
                        processed_nodes.add(child._id)

    @property
    def name(self):
        return self.root.children[0].name

    @property
    def type(self):
        return self.root.children[1]

    @property
    def body(self):
        return self.root.children[2]

    @property
    def type_nodes(self) -> Iterator[AgdaNode]:
        yield from self.type.nodes

    @property
    def body_nodes(self) -> Iterator[AgdaNode]:
        yield from self.body.nodes

    @property
    def definition_type(self):
        return self.root.children[2].node_type

    @property
    def module_name(self) -> str:
        return self._module_name

    @property
    def is_internal(self):
        return self._is_internal

    def _find_module_name(self):
        node = self.root
        while node is not None and node.node_type != NodeType.MODULE:
            node = node.parent
        if node is None:
            raise ValueError(f"Could not found module for {self.name}")
        return node.children[0].name

    def to_words(
        self,
        graph: nx.MultiDiGraph,
        id_to_definition: Dict[MyTypes.NODE, "AgdaDefinition"],
    ) -> List[str]:
        relevant_nodes: List[MyTypes.NODE] = []
        if not AgdaDefinition.is_with_definition(self.name):
            relevant_nodes.append(self.name)
            relevant_nodes.extend(self._get_helper_nodes(graph))
        parts: List[str] = []
        for node in relevant_nodes:
            related_definition = id_to_definition[node]
            parts.append(related_definition.type.full_text)
            parts.append(related_definition.body.full_text)
        text = TextManipulation.normalize_tree_text(" ".join(parts))
        return TextManipulation.extract_words(text)

    def _get_helper_nodes(self, graph: nx.MultiDiGraph) -> List[MyTypes.NODE]:
        helper_nodes: List[MyTypes.NODE] = []
        to_do = [self.name]
        processed: Set[MyTypes.NODE] = {self.name}
        while to_do:
            node = to_do.pop()
            for sink in graph[node]:
                if sink in processed:
                    continue
                elif AgdaDefinition.is_with_definition(
                    sink
                ) or AgdaDefinition.is_rewrite_definition(sink):
                    helper_nodes.append(sink)
                    to_do.append(sink)
                    processed.add(sink)
        return helper_nodes

    @staticmethod
    def named_nodes(node_iterator: Iterator[AgdaNode]) -> Iterator[AgdaNode]:
        for node in node_iterator:
            if not node.node_type.is_name():
                continue
            yield node

    @staticmethod
    def is_with_definition(qualified_name: MyTypes.NODE):
        """
        Computes whether the qualified name refers to a with definition.

        :param qualified_name: A string of form 'module.submodule. ... .sub...submodule.unqualified'
        :return: the boolean value of 'unqualified matches regular expression with-some_number'. For example,
                 'with-123' matches it, whereas 'with-' and 'myth-123' do not.
        """
        unqualified = qualified_name.split(".")[-1]
        return re.match("with-\\d+ \\d+$", unqualified) is not None

    @staticmethod
    def is_rewrite_definition(qualified_name: Union[str, MyTypes.NODE]):
        """
        Computes whether the qualified name refers to a rewrite definition.

        :param qualified_name: A string of form 'module.submodule. ... .sub...submodule.unqualified'
        :return: the boolean value of 'unqualified matches regular expression -rewriteN'. For example,
                 '-rewrite123' matches it, whereas '-rewrite' and 'write-123' do not.
        """
        unqualified = qualified_name.split(".")[-1]
        return re.match("-rewrite\\d+ \\d+$", unqualified) is not None

    @staticmethod
    def is_normal_definition(qualified_name: str | MyTypes.NODE):
        return not AgdaDefinition.is_with_definition(
            qualified_name
        ) and not AgdaDefinition.is_rewrite_definition(qualified_name)

    @staticmethod
    def dummy_definition(identifier: MyTypes.NODE) -> "AgdaDefinition":
        dummy_module = AgdaNode(NodeType.MODULE, "", None, [])
        module_name = AgdaNode(NodeType.MODULE_NAME, '"a.dummy.module"', None, [])
        root_node = AgdaNode(NodeType.EXTERNAL, "", None, [])
        name_node = AgdaNode(
            NodeType.NAME, AgdaNode.name_to_description(identifier), None, []
        )
        type_node = AgdaNode(NodeType.EXTERNAL, "", None, [])
        body_node = AgdaNode(NodeType.EXTERNAL, "", None, [])
        AgdaNode.connect_parent_to_children(dummy_module, [module_name, root_node])
        AgdaNode.connect_parent_to_children(
            root_node, [name_node, type_node, body_node]
        )
        return AgdaDefinition(Locations.NON_FILE, root_node, False)


class AgdaDefinitionForest(AgdaForest):
    def __init__(self, definitions: List[AgdaDefinition]):
        is_internal = [definition.is_internal for definition in definitions]
        super().__init__(definitions, is_internal)

    @staticmethod
    def load(file: str) -> "AgdaDefinitionForest":
        with open(file, "rb") as f:
            return pickle.load(f)


def test_agda_tree():
    files = ["test_data/small.agda-sexp", "test_data/Agda.Primitive.agda-sexp"]
    for file in files:
        tree = AgdaTree.create_from_file(file)
        LOGGER.info(f"{tree.depth}, {tree.n_nodes}, {len(list(tree.nodes))}")


if __name__ == "__main__":
    print(
        NodeType.NAME.is_name(),
        NodeType.MODULE_NAME.is_name(),
        NodeType.MODULE.is_name(),
    )
