from typing import Dict, Tuple, Set, Iterator, List, Any
import networkx as nx
import tqdm
import pickle
import re
import os

from apaa.other.helpers import NodeType, Other, Locations, MyTypes, EdgeType
from .agda_tree import AgdaNode, AgdaDefinition, AgdaDefinitionForest
from .graph_properties import GraphProperties
from .database import DatabaseManipulation


Node = MyTypes.NODE

LOGGER = Other.create_logger(__file__)


class KnowledgeGraph:
    def __init__(
        self, library: str, definitions: AgdaDefinitionForest, debug: bool = True
    ):
        self._library = library
        self.id_to_definition: Dict[Node, AgdaDefinition] = {}
        self._graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self.definition_types: Set[NodeType] = set()
        self._init(definitions)
        self._graph_properties: GraphProperties = self._init_graph_properties(debug)
        self._init_meta()
        self._check()

    def _init(self, definitions: AgdaDefinitionForest):
        self._init_graph_nodes(definitions)
        self._init_rest_of_graph(definitions)
        LOGGER.info(
            f"Created a graph with (counts below exclude module nodes/edges)\n"
            f"    - {len(self.graph.nodes)} node(s)\n"
            f"    - {len(self.graph.edges)} edge(s)\n"
            f"    - {len(self.definition_types)} node type(s): {self.definition_types}."
        )

    def _check(self):
        """
        The following checks are made:
        - no dummy nodes
        - all nodes have label
        - all reference edges have weight

        :return:
        """
        self._check_all_labelled()
        self._check_weighted_references()
        LOGGER.info("Checks of the graph passed.")

    def _check_all_labelled(self):
        node: Node
        for node in self.graph:
            if "label" not in self.graph.nodes[node]:
                raise ValueError(
                    f"{node} has no property label: {self.graph.nodes[node]}"
                )

    def _check_weighted_references(self):
        for source, sink, edge_type, params in self.graph.edges(data=True, keys=True):
            assert isinstance(edge_type, EdgeType)
            if edge_type.is_reference() and "w" not in params:
                raise ValueError(
                    f"Edge {source}--{edge_type}-->{sink} has no 'w': {params}"
                )

    def add_definition(self, definition: AgdaDefinition):
        name = definition.name
        d_type = definition.body.node_type
        if self.should_add_a_new_node(name, d_type, definition):
            self.id_to_definition[name] = definition
            self.add_node_to_graph(name, d_type)
    
    def should_add_a_new_node(self, name: Node, d_type: NodeType, definition: AgdaDefinition) -> bool:
        if name not in self.id_to_definition:
            return True
        LOGGER.warning(f"Definition name {definition.name} repeats!")
        # hopefully, this is some locally-instantied entry that was imported from somewhere:
        # let's test its qualified name
        if definition.name.startswith(definition.module_name):
            raise ValueError(
                f"Definition {definition.name} is already in the graph and "
                f"it starts with its module name {definition.module_name}!"
            )
        # Let's also test whether this is a constructor
        other_d_type = self.id_to_definition[name].body.node_type
        if not (d_type == other_d_type == NodeType.CONSTRUCTOR):
            other_module_name = self.id_to_definition[name].module_name
            raise ValueError(
                f"The new and the present definitions for {definition.module_name}/{other_module_name}: {name} are not both ':constructors': "
                f"{d_type}, {other_d_type}!"
            )
        return False

    def add_node_to_graph(self, name: Node, node_type: NodeType) -> bool:
        if name in self.graph.nodes:
            return False
        self.graph.add_node(name, label=node_type)
        self.definition_types.add(node_type)
        return True

    def add_edge_to_graph(
        self,
        source: Node,
        sink: Node,
        edge_type: EdgeType,
        **edge_data: Any,
    ):
        if self.graph.get_edge_data(source, sink, edge_type, default=None) is None:
            self.graph.add_edge(source, sink, edge_type, **edge_data)
        elif (
            "w" in self.graph[source][sink][edge_type]
            and "w" in edge_data
            and len(edge_data) == 1
        ):
            self.graph[source][sink][edge_type]["w"] += edge_data["w"]
        else:
            raise ValueError(
                f"Edge ({source}, {sink}, {edge_type}) already present, "
                "but cannot merge. "
                f"Edge data: {edge_data}"
            )

    def _init_graph_nodes(self, definitions: AgdaDefinitionForest):
        LOGGER.info(f"Processing {len(definitions)} definition(s)")
        for definition in definitions:
            self.add_definition(definition)
        self._update_with_missing_nodes(definitions)

    def _update_with_missing_nodes(self, definitions: AgdaDefinitionForest):
        missing: Set[Node] = set()
        for definition in tqdm.tqdm(definitions):
            for node in AgdaDefinition.named_nodes(definition.nodes):
                name = node.name
                if name not in self.id_to_definition:
                    missing.add(name)
                    new_definition = AgdaDefinition.dummy_definition(name)
                    self.add_definition(new_definition)
        if missing:
            LOGGER.warning(
                f"The following {len(missing)} definitions "
                f"were missing: {sorted(missing)}"
            )
        else:
            LOGGER.info("No missing definitions found.")

    def _init_rest_of_graph(self, definitions: AgdaDefinitionForest):
        with_to_main: Dict[Node, Node] = {}
        rewrite_to_main: Dict[Node, Node] = {}
        to_from_with_edges: Dict[Node, List[Tuple[Node, EdgeType]]] = {}
        to_from_rewrite_edges: Dict[Node, List[Tuple[Node, EdgeType]]] = {}
        for definition in tqdm.tqdm(definitions):
            assert isinstance(definition, AgdaDefinition)
            if AgdaDefinition.is_with_definition(definition.name):
                KnowledgeGraph._update_main_definition_of_with(
                    definition.name, to_from_with_edges, with_to_main
                )
            elif AgdaDefinition.is_rewrite_definition(definition.name):
                KnowledgeGraph._update_main_definition_of_rewrite(
                    definition.name, to_from_rewrite_edges, rewrite_to_main
                )
            self._init_edges_declaration(
                definition, to_from_with_edges, to_from_rewrite_edges
            )
            self._init_edges_body(definition, to_from_with_edges, to_from_rewrite_edges)
        self._postprocess_with_nodes(with_to_main, to_from_with_edges)
        self._postprocess_rewrite_nodes(rewrite_to_main)

    def _init_edges_of_type(
        self,
        definition: AgdaDefinition,
        candidate_nodes: Iterator[AgdaNode],
        edge_type: EdgeType,
        to_with_references: Dict[Node, List[Tuple[Node, EdgeType]]],
        to_rewrite_references: Dict[Node, List[Tuple[Node, EdgeType]]],
    ):
        references = list(AgdaDefinition.named_nodes(candidate_nodes))
        self._init_edges(
            definition, references, edge_type, to_with_references, to_rewrite_references
        )

    def _init_edges_declaration(
        self,
        definition: AgdaDefinition,
        to_with_references: Dict[Node, List[Tuple[Node, EdgeType]]],
        to_rewrite_references: Dict[Node, List[Tuple[Node, EdgeType]]],
    ):
        self._init_edges_of_type(
            definition,
            definition.type_nodes,
            EdgeType.REFERENCE_IN_TYPE,
            to_with_references,
            to_rewrite_references,
        )

    def _init_edges_body(
        self,
        definition: AgdaDefinition,
        to_with_references: Dict[Node, List[Tuple[Node, EdgeType]]],
        to_rewrite_references: Dict[Node, List[Tuple[Node, EdgeType]]],
    ):
        self._init_edges_of_type(
            definition,
            definition.body_nodes,
            EdgeType.REFERENCE_IN_BODY,
            to_with_references,
            to_rewrite_references,
        )

    def _init_edges(
        self,
        definition: AgdaDefinition,
        nodes: List[AgdaNode],
        edge_type: EdgeType,
        to_with_references: Dict[Node, List[Tuple[Node, EdgeType]]],
        to_rewrite_references: Dict[Node, List[Tuple[Node, EdgeType]]],
    ):
        d_name = definition.name
        edge_type_with = edge_type.normal_to_with()
        edge_type_rewrite = edge_type.normal_to_rewrite()
        for node in nodes:
            name = node.name
            if name not in self.graph.nodes:
                raise ValueError(
                    f"No definition of '{name}' (edge '{d_name}'-->'{name}')"
                )
            used_edge = edge_type
            if AgdaDefinition.is_with_definition(name):
                used_edge = edge_type_with
                KnowledgeGraph._maybe_update_to_with_reference(
                    to_with_references, d_name, name, edge_type_with
                )
            elif AgdaDefinition.is_rewrite_definition(name):
                used_edge = edge_type_rewrite
                KnowledgeGraph._maybe_update_to_rewrite_reference(
                    to_rewrite_references, d_name, name, edge_type_rewrite
                )
            # In any case, mark only the original reference, connect on the main level (if necessary) later
            self.add_edge_to_graph(d_name, name, used_edge, w=1)

    @staticmethod
    def _maybe_update_to_with_reference(
        to_with_references: Dict[Node, List[Tuple[Node, EdgeType]]],
        d_name: Node,
        name: Node,
        edge_type_with: EdgeType,
    ):
        if d_name == name:
            LOGGER.warning(
                f"Skipping to-with self-reference {d_name}-{edge_type_with.value}->{name}"
            )
        else:
            if name not in to_with_references:
                to_with_references[name] = []
            to_with_references[name].append((d_name, edge_type_with))

    @staticmethod
    def _maybe_update_to_rewrite_reference(
        to_rewrite_references: Dict[Node, List[Tuple[Node, EdgeType]]],
        d_name: Node,
        name: Node,
        edge_type_rewrite: EdgeType,
    ):
        if d_name == name:
            LOGGER.warning(
                f"Skipping rewrite self-reference {d_name}-{edge_type_rewrite}->{name}"
            )
        else:
            if name not in to_rewrite_references:
                to_rewrite_references[name] = []
            to_rewrite_references[name].append((d_name, edge_type_rewrite))
            if len(to_rewrite_references[name]) > 1:
                LOGGER.error(
                    f"WTF: more than 1 rewrite reference to {name}: {to_rewrite_references[name]}"
                )

    @staticmethod
    def _update_main_definition_of_with(
        d_name: Node,
        to_with_references: Dict[Node, List[Tuple[Node, EdgeType]]],
        with_to_main: Dict[Node, Node],
    ):
        # At least the definition where this "with" is introduced, should already reference d_name
        if d_name not in to_with_references:
            raise ValueError(
                "At least one definition (the one where this with is used) "
                "should be already processed"
            )
        # That definition is also the last one that references d_name
        main_def, _ = to_with_references[d_name][-1]
        # Do not do this, compute only "direct" main":
        # if LibraryEntry.is_with_definition(main_def):
        #     main_def = with_to_main[main_def]
        with_to_main[d_name] = main_def

    def _postprocess_with_nodes(
        self,
        with_to_main: Dict[Node, Node],
        to_from_with_edges: Dict[Node, List[Tuple[Node, EdgeType]]],
    ):
        with_to_root = KnowledgeGraph._compute_with_to_root(with_to_main)
        total_weights = self._compute_total_weights_of_with_nodes(
            with_to_root, to_from_with_edges
        )
        self._postprocess_from_with_references(with_to_root, total_weights)
        self._postprocess_to_with_references(with_to_root, to_from_with_edges)

    def _postprocess_from_with_references(
        self,
        with_to_root: Dict[Node, Node],
        total_weights: Dict[Node, Dict[EdgeType, int]],
    ):
        # "move" references from with
        LOGGER.debug("Postprocess from with")
        for with_node, main_node in with_to_root.items():
            LOGGER.debug(f"  Processing {with_node} (with main {main_node})")
            for ref, edges in self.graph[with_node].items():
                if AgdaDefinition.is_with_definition(ref):
                    continue
                main_of_ref = with_to_root.get(ref, ref)
                total_connection_weight = sum(props["w"] for props in edges.values())
                for with_edge_type, w_tot in total_weights[with_node].items():
                    LOGGER.debug(
                        f"    Updating {ref}: {main_node}-{with_edge_type.value}->{main_of_ref} "
                        f"with w +={total_connection_weight * w_tot}"
                    )
                    normalized_edge_type = with_edge_type.with_to_normal()
                    self.add_edge_to_graph(
                        main_node,
                        main_of_ref,
                        normalized_edge_type,
                        w=total_connection_weight * w_tot,
                    )

    def _postprocess_to_with_references(
        self,
        with_to_root: Dict[Node, Node],
        to_from_with_edges: Dict[Node, List[Tuple[Node, EdgeType]]],
    ):
        # "move" "true" references to with nodes:
        # - skip with->with, because these were already processed in "from with"
        # - skip x->with if main(x) = main(with)
        LOGGER.info("Postprocess to with references")
        _, reversed_with_graph = KnowledgeGraph._create_with_graph(to_from_with_edges)
        for with_node, references in reversed_with_graph.items():
            LOGGER.debug(f"Processing {with_node}")
            for referencing_node, edge_types in references.items():
                LOGGER.debug(f"  Processing WITH <-- {referencing_node}")
                this_root = with_to_root[with_node]
                other_root = with_to_root.get(referencing_node, referencing_node)
                if this_root == other_root or AgdaDefinition.is_with_definition(
                    referencing_node
                ):
                    continue
                for edge_type in edge_types:
                    try:
                        w = self.graph[referencing_node][with_node][edge_type]["w"]
                        LOGGER.debug(
                            f"    Updating {other_root}-{edge_type.value}->{this_root} with w +={w}"
                        )
                        self.add_edge_to_graph(
                            other_root, this_root, edge_type.with_to_normal(), w=w
                        )
                    except KeyError:
                        LOGGER.error(
                            f"{edge_type} not in self.graph[{referencing_node}][{with_node}]: "
                            f"{self.graph[referencing_node][with_node]}"
                        )
                        raise

    def _compute_total_weights_of_with_nodes(
        self,
        with_to_root: Dict[Node, Node],
        to_from_with_edges: Dict[Node, List[Tuple[Node, EdgeType]]],
    ) -> Dict[Node, Dict[EdgeType, int]]:
        # to_from_with_edges is reversed graph, but with repetitions
        with_graph, reversed_with_graph = KnowledgeGraph._create_with_graph(
            to_from_with_edges
        )
        KnowledgeGraph._cycles(with_graph)
        stack = list(filter(lambda n: not reversed_with_graph[n], reversed_with_graph))
        total_weights: Dict[Node, Dict[EdgeType, int]] = {}
        while stack:
            node = stack.pop()
            parents = reversed_with_graph[node]
            if not parents:
                total_weights[node] = {}  # add it to mark it processed
            if parents:
                total_weights[node] = self._compute_total_weights_of_single(
                    total_weights, with_to_root, node, parents
                )
            LOGGER.debug(f"total_weights[{node}] = {total_weights[node]}")
            for child in with_graph[node]:
                if all(
                    parent_of_child in total_weights
                    for parent_of_child in reversed_with_graph[child]
                ):
                    stack.append(child)
        return total_weights

    def _compute_total_weights_of_single(
        self,
        total_weights: Dict[Node, Dict[EdgeType, int]],
        with_to_root: Dict[Node, Node],
        node: Node,
        parents: Dict[Node, Set[EdgeType]],
    ) -> Dict[EdgeType, int]:
        root = with_to_root[node]
        ws: Dict[EdgeType, int] = {}
        for parent, e_types in parents.items():
            if with_to_root.get(parent, parent) != root:
                # this is some other reference
                continue
            if parent == root:
                # take the weights of the edge types that actually exist from parent to this
                for e_type in e_types:
                    ws[e_type] = (
                        ws.get(e_type, 0) + self.graph[parent][node][e_type]["w"]
                    )
            else:
                # compute total weight from parent (with of >= 1st order) to this (>= 2nd order)
                # and multiply it with the total weights of parent (all edges)
                weight_parent_to_node = sum(
                    props["w"] for props in self.graph[parent][node].values()
                )
                for e_type, tot_weight in total_weights[parent].items():
                    ws[e_type] = ws.get(e_type, 0) + tot_weight * weight_parent_to_node
        return ws

    @staticmethod
    def _cycles(with_graph: Dict[Node, Dict[Node, Set[EdgeType]]]):
        graph: nx.DiGraph = nx.DiGraph()
        for source, sinks in with_graph.items():
            for sink in sinks:
                graph.add_edge(source, sink)
        LOGGER.info(
            f"With graph has {len(graph.nodes)} nodes and {len(graph.edges)} edges"
        )
        cycles: List[List[Node]] = list(nx.simple_cycles(graph))
        LOGGER.info(f"With graph contains {len(cycles)} cycles")
        for c in cycles:
            if len(c) > 1:
                LOGGER.error(f"    {c}")

    @staticmethod
    def _compute_with_to_root(with_to_main: Dict[Node, Node]) -> Dict[Node, Node]:
        with_to_root: Dict[Node, Node] = {}
        for node, parent in with_to_main.items():
            while parent in with_to_main:
                parent = with_to_main[parent]
            with_to_root[node] = parent
            LOGGER.debug(f"with_to_root[{node}] = {parent}")
        return with_to_root

    @staticmethod
    def _create_with_graph(
        to_from_with_edges: Dict[Node, List[Tuple[Node, EdgeType]]]
    ) -> Tuple[
        Dict[Node, Dict[Node, Set[EdgeType]]], Dict[Node, Dict[Node, Set[EdgeType]]]
    ]:
        with_graph: Dict[Node, Dict[Node, Set[EdgeType]]] = {}
        reversed_with_graph: Dict[Node, Dict[Node, Set[EdgeType]]] = {}
        all_nodes: Set[Node] = set()
        for sink, sources in to_from_with_edges.items():
            all_nodes.add(sink)
            for source, _ in sources:
                all_nodes.add(source)
        for node in all_nodes:
            with_graph[node] = {}
            reversed_with_graph[node] = {}
        for sink, sources in to_from_with_edges.items():
            for source, e_type in sources:
                KnowledgeGraph._update_with_graph(with_graph, source, sink, e_type)
                KnowledgeGraph._update_with_graph(
                    reversed_with_graph, sink, source, e_type
                )
        return with_graph, reversed_with_graph

    @staticmethod
    def _update_with_graph(
        with_graph: Dict[Node, Dict[Node, Set[EdgeType]]],
        source: Node,
        sink: Node,
        edge_type: EdgeType,
    ):
        if sink not in with_graph[source]:
            with_graph[source][sink] = set()
        with_graph[source][sink].add(edge_type)

    def _postprocess_rewrite_nodes(self, rewrite_to_main: Dict[Node, Node]):
        """
        Moves references rewrite --> something (we know that this is not rewrite)
        to main(rewrite) -> something

        :param rewrite_to_main:
        :return:
        """
        for node in self.graph:
            if not AgdaDefinition.is_rewrite_definition(node):
                continue
            main = rewrite_to_main[node]
            for reference in self.graph[node]:
                for edge_type, props in self.graph[node][reference].items():
                    if not edge_type.is_normal_reference():
                        LOGGER.error(f"Skipping {node} -[{edge_type}]->{reference}")
                        continue
                    self.add_edge_to_graph(main, reference, edge_type, **props)

    @staticmethod
    def _update_main_definition_of_rewrite(
        d_name: Node,
        to_rewrite_references: Dict[Node, List[Tuple[Node, EdgeType]]],
        rewrite_to_main: Dict[Node, Node],
    ):
        # At least the definition where this "rewrite" is introduced, should already reference d_name
        if d_name not in to_rewrite_references:
            raise ValueError(
                "At least one definition (the one where this with is used) "
                "should be already processed"
            )
        # That definition is also the last one that references d_name
        main_def, _ = to_rewrite_references[d_name][-1]
        rewrite_to_main[d_name] = main_def
        if main_def in rewrite_to_main:
            LOGGER.error(
                f"WTF: rewrite contains rewrite: {d_name} <-- {main_def} <-- {rewrite_to_main[main_def]}"
            )

    def _init_meta(self):
        LOGGER.info("Processing meta nodes and edges.")
        library_node = f"library_{self._library}"
        fake_library_node = "external source"
        module_nodes: Set[Node] = set()
        self.add_node_to_graph(library_node, NodeType.LIBRARY)
        for definition in self.id_to_definition.values():
            module_nodes |= self._add_module_chain(
                definition, library_node, fake_library_node, definition.is_internal
            )
        module_name_to_nodes: Dict[str, List[Node]] = {}
        for node in module_nodes:
            name = node  # it used to be _, _, name = node, dirty fix :)
            if name not in module_name_to_nodes:
                module_name_to_nodes[name] = []
            module_name_to_nodes[name].append(node)
        for name, nodes in module_name_to_nodes.items():
            nodes.sort(reverse=True)
            if len(nodes) == 2:
                node_proper, node_temp = nodes
                self._update_node_edges(node_proper, node_temp)
            elif len(nodes) > 2:
                raise ValueError(
                    f"At most two nodes of the same name can be present, but: {nodes}"
                )

    def _add_module_chain(
        self,
        definition: AgdaDefinition,
        library_node: Node,
        fake_library_node: Node,
        is_internal: bool,
    ):
        this = definition.name
        parent = definition.module_name
        edge_type = EdgeType.DEFINES
        module_nodes: Set[Node] = set()
        while True:
            module_nodes.add(parent)
            module_type = NodeType.MODULE if is_internal else NodeType.EXTERNAL_MODULE
            was_added = self.add_node_to_graph(parent, module_type)
            self.graph.add_edge(parent, this, edge_type)
            this = parent
            parent = self._find_parent_name(parent)
            edge_type = EdgeType.CONTAINS
            if not parent:
                break
            elif not was_added:
                return module_nodes
        if is_internal:
            self.graph.add_edge(library_node, this, edge_type)
        else:
            self.add_node_to_graph(fake_library_node, NodeType.EXTERNAL_LIBRARY)
            self.graph.add_edge(fake_library_node, this, edge_type)
        return module_nodes

    @staticmethod
    def _find_parent_name(name: str):
        i = name.rfind(".")
        if i < 0:
            return ""
        else:
            return name[:i]

    def _update_node_edges(self, proper_node: Node, temp_node: Node):
        LOGGER.info(
            f"Moving edges from {temp_node} to {proper_node} and deleting {temp_node}"
        )
        # create new in-edges for proper node
        for source, _ in self.graph.in_edges(temp_node):
            for e_type, properties in self.graph[source][temp_node].items():
                if (source, proper_node, e_type) in self.graph.edges:
                    continue
                self.graph.add_edge(source, proper_node, e_type, **properties)
        # create new out-edges for proper node
        for sink in self.graph[temp_node]:
            for e_type, properties in self.graph[temp_node][sink].items():
                if (proper_node, sink, e_type) in self.graph.edges:
                    continue
                self.graph.add_edge(proper_node, sink, e_type, **properties)
        # remove temp_node and its edges
        self.graph.remove_node(temp_node)

    def dump(self, file: str):
        with open(file, "wb") as f:
            pickle.dump(self, f)

    def dump_pure(self, file: str):
        with open(file, "wb") as f:
            pickle.dump(self.graph, file=f)

    @staticmethod
    def load(file: str) -> "KnowledgeGraph":
        with open(file, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def load_pure(file: str) -> nx.MultiDiGraph:
        with open(file, "rb") as f:
            g = pickle.load(f)
        assert isinstance(g, nx.MultiDiGraph)
        return g

    @staticmethod
    def create_from_definitions_file(
        library: str, definitions_file: str
    ) -> "KnowledgeGraph":
        LOGGER.info(f"Creating graph from '{definitions_file}'")
        return KnowledgeGraph(library, AgdaDefinitionForest.load(definitions_file))

    @staticmethod
    def database_node_id(node_id: Node) -> str:
        return node_id

    @staticmethod
    def database_node_name(node_id: Node) -> str:
        if " " in node_id:
            name, _ = node_id.split(" ")
        else:
            name = node_id
        return name.split(".")[-1]

    def _get_node_label(self, node_id: Node) -> NodeType:
        return self.graph.nodes[node_id]["label"]

    @staticmethod
    def _node_type_to_str(node_type: NodeType) -> str:
        replacements = [
            # ("ω", "_omega"),
            ("-", "_")
        ]
        label = node_type.value
        assert label.startswith(":")
        label = label[1:]
        for bad, good in replacements:
            label = re.sub(bad, good, label)
        return label

    @staticmethod
    def database_edge_type(edge_type: EdgeType):
        return edge_type.value

    def dump_to_database(
        self,
        path_to_neo: str = "bolt://localhost:7687",
        authentication: Tuple[str, str] = ("neo4j", "test"),
    ):
        graph = DatabaseManipulation.create_graph_connection(
            path_to_neo, authentication
        )
        # constraints
        for node_type in list(self.definition_types) + [NodeType.MODULE_LIKE]:
            DatabaseManipulation.create_uniqueness_constraint(
                graph, KnowledgeGraph._node_type_to_str(node_type)
            )
        # nodes
        nodes = []
        for node in self.graph.nodes:
            labels = self._create_node_labels(node)
            properties = self._create_node_properties(node)
            nodes.append((labels, properties))
        DatabaseManipulation.create_nodes(graph, nodes)
        # edges
        edges: List[
            Tuple[Tuple[List[str], str], Tuple[List[str], str], str, Dict[str, Any]]
        ] = []
        for e in self.graph.edges:
            source_id, sink_id, e_type = e
            source_labels = self._create_node_labels(source_id)
            sink_labels = self._create_node_labels(sink_id)
            edges.append(
                (
                    (source_labels, KnowledgeGraph.database_node_id(source_id)),
                    (sink_labels, KnowledgeGraph.database_node_id(sink_id)),
                    self.database_edge_type(e_type),
                    self.graph.edges[e],
                )
            )
        DatabaseManipulation.create_edges(graph, edges)

    def _create_node_properties(self, node: Node) -> Dict[str, float]:
        properties: Dict[str, Any] = {
            p: value for p, value in self.graph.nodes[node].items() if p != "label"
        }
        properties["id"] = KnowledgeGraph.database_node_id(node)
        properties["name"] = KnowledgeGraph.database_node_name(node)
        if node in self.id_to_definition:
            for statistic, values in self.graph_properties.node_statistics.items():
                properties[statistic] = values[node]
        else:
            label = self.graph.nodes[node]["label"]
            assert isinstance(label, NodeType)
            if label == NodeType.LIBRARY:
                for statistic, value in self.graph_properties.graph_statistics.items():
                    properties[statistic] = value
            elif label.is_module() or label.is_external():
                # no statistics
                pass
            else:
                raise ValueError(
                    f"Expected meta/definition but got {label} for {node}."
                )
        return properties

    def _create_node_labels(self, node: Node):
        if node in self.id_to_definition:
            return self._create_labels_definition(node)
        else:
            return self._create_labels_meta(node)

    def _create_labels_definition(self, node: Node):
        labels = [KnowledgeGraph._node_type_to_str(self._get_node_label(node))]
        if not self.id_to_definition[node].is_internal:
            labels.append(KnowledgeGraph._node_type_to_str(NodeType.EXTERNAL))
        return labels

    def _create_labels_meta(self, node: Node):
        base_label = self._get_node_label(node)
        labels = [
            KnowledgeGraph._node_type_to_str(label)
            for label in [NodeType.MODULE_LIKE, base_label]
        ]
        if not node[0]:
            # hash is dummy
            labels.pop()
        if NodeType.is_external(base_label):
            labels.append(self._node_type_to_str(NodeType.EXTERNAL))
        return labels

    def _init_graph_properties(self, debug: bool) -> GraphProperties:
        prop_file = os.path.join(Locations.DUMPS_DIR, self._library + "_prop.pickle")
        if os.path.exists(prop_file):
            LOGGER.info(f"Loading graph properties from '{prop_file}'")
            return GraphProperties.load(prop_file)
        properties = GraphProperties(self.graph, weight="w", debug=debug)
        properties.dump(prop_file)
        return properties

    @property
    def graph_properties(self):
        return self._graph_properties

    @property
    def graph(self):
        return self._graph

    def graph_text_dump(self, file: str):
        with open(file, "w", encoding="utf-8") as f:
            LOGGER.info("Writing nodes")
            lines: List[Tuple[int | float, str]] = []
            for node, node_props in self.graph.nodes(True):
                degree_stats = {}
                degree = 0
                if "degree" in self.graph_properties.node_statistics:
                    if node in self.graph_properties.node_statistics["degree"]:
                        degree_stats = {
                            stat: self.graph_properties.node_statistics[stat][node]
                            for stat in ["in_degree", "out_degree", "degree"]
                        }
                        degree = degree_stats["degree"]
                lines.append((-degree, f"N;{node};{node_props};{degree_stats}"))
            lines.sort()
            for _, line in lines:
                print(line, file=f)
            LOGGER.info("Writing edges")
            for node1 in tqdm.tqdm(self.graph):
                for node2 in self.graph[node1]:
                    for edge_type, edge_props in self.graph[node1][node2].items():
                        print(
                            f"E;{node1};{node2};{edge_type.value};{edge_props}", file=f
                        )


if __name__ == "__main__":
    lib_name = Locations.NAME_AGDA_TEST
    kg = KnowledgeGraph.load(Locations.knowledge_graph(lib_name))
    LOGGER.info("Loaded")
    kg.dump_pure(Locations.knowledge_graph_pure(lib_name))
    # LOGGER.info("Dumped")
    # g = KnowledgeGraph.load_pure(Locations.knowledge_graph_pure(lib_name))
    # print(len(g.edges), len(g.nodes))
    kg.graph_text_dump(Locations.graph_text_dump(lib_name))
    # kg.dump_to_database()
