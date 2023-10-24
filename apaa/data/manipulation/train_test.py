import random
import tqdm
import heapq
from typing import Tuple, List, Dict, Any, Iterator
import networkx as nx

from apaa.data.structures.agda_tree import AgdaDefinition, AgdaNode
from apaa.other.helpers import EdgeType, NodeType
from apaa.other.helpers import Other, MyTypes


LOGGER = Other.create_logger(__file__)
LOGGER_DETAILS = Other.create_logger("details", file="prune_stats.log")
Node = MyTypes.NODE


def get_theorems_and_other(
    definition_ids_in_order: list[Node],
    id_to_definition: dict[Node, AgdaDefinition],
    theorem_like_tag: NodeType,
) -> tuple[list[Node], list[Node]]:
    function_indices: list[Node] = []
    other_indices: list[Node] = []
    for def_id in definition_ids_in_order:
        definition = id_to_definition[def_id]
        if (
            definition.body.node_type == theorem_like_tag
            and AgdaDefinition.is_normal_definition(def_id)
        ):
            # this is the only thing that can go into test
            function_indices.append(def_id)
        else:
            other_indices.append(def_id)
    return function_indices, other_indices


def check_validity_of_train_test_entries(
    id_to_definition: Dict[Node, AgdaDefinition],
    function_ids: list[Node],
    train_entries: list[Node] | None,
    test_entries: list[Node] | None,
    theorem_like_tag: NodeType,
) -> tuple[bool, str]:
    if (train_entries is None) != (test_entries is None):
        raise ValueError("Either both or none of test/train defs must be given.")
    if train_entries is None or test_entries is None:
        return True, "ok"
    for entry in test_entries:
        if entry not in id_to_definition:
            return False, f"missing entry {entry}"
        definition = id_to_definition[entry]
        if definition.body.node_type != theorem_like_tag:
            return False, f"wrong node type {definition.body.node_type} of {entry}"
    if sorted(function_ids) != sorted(test_entries + train_entries):
        return False, "something missing or too much"
    return True, "ok"


def prepare_dataset(
    graph: nx.MultiDiGraph,
    id_to_definition: Dict[Node, AgdaDefinition],
    p_test: float = 0.2,
    p_def_to_keep: float = 0.0,
    seed: int = 123,
    train_defs: List[Node] | None = None,
    test_defs: List[Node] | None = None,
) -> Tuple[
    nx.MultiDiGraph,
    Tuple[Dict[Node, AgdaDefinition], Dict[Node, AgdaDefinition]],
    Tuple[
        List[Tuple[Node, Node, EdgeType]],
        List[Tuple[Node, Node, EdgeType]],
    ],
]:
    """
    1. Split to train and test
    2. Prune test definition trees (and cut the edges)
    3. Return (modified graph, (train defs, test defs), (positive edges, negative edges))
    """
    random.seed(seed)
    theorem_like_tag = NodeType.get_theorem_like_tag(graph)
    LOGGER.info(f"Theorem-like tag: {theorem_like_tag}")
    definitions_ids = sorted(id_to_definition)
    random.shuffle(definitions_ids)
    function_ids, other_ids = get_theorems_and_other(
        definitions_ids, id_to_definition, theorem_like_tag
    )
    n = len(function_ids)
    LOGGER.info(
        f"Found {n} definitions with tag {theorem_like_tag} "
        f"and {len(other_ids)} other definitions."
    )
    is_ok, reason = check_validity_of_train_test_entries(
        id_to_definition, function_ids, train_defs, test_defs, theorem_like_tag
    )
    if not is_ok:
        raise ValueError(reason)
    if test_defs is not None:
        p_test = len(test_defs) / n
    LOGGER.info(f"Dataset {p_test} {p_def_to_keep}")
    if n <= 1:
        raise ValueError("Cannot split less than 2 examples.")
    n_test = max(1, min(n - 1, round(p_test * n)))
    train_test_defs = ({}, {})
    train_graph: nx.MultiDiGraph = graph.copy(False)
    positive_edges = []
    negative_edges = []
    for def_id in other_ids:
        train_test_defs[0][def_id] = id_to_definition[def_id]
    if test_defs is not None:
        # test defs at the beginning
        function_ids = test_defs + function_ids
    for def_id in tqdm.tqdm(function_ids):
        if any(def_id in part for part in train_test_defs):
            # test_defs are present twice
            continue
        definition = id_to_definition[def_id]
        if len(train_test_defs[1]) == n_test:
            train_test_defs[0][def_id] = definition
            continue
        appropriate_for_test, pruned, removed_edges = prune_definition(
            train_graph, definition, p_def_to_keep
        )
        train_test_defs[appropriate_for_test][def_id] = pruned
        if appropriate_for_test:
            positive_edges.extend(removed_edges)
            negative_edges.extend(
                Other.sample_negative_edges(
                    graph, definitions_ids, def_id, len(removed_edges)
                )
            )
    LOGGER.info(
        f"Train : Test split = {len(train_test_defs[0])} : {len(train_test_defs[1])} examples"
    )
    LOGGER.info(f"Edges to test on: {len(positive_edges)} + {len(negative_edges)}")
    return train_graph, train_test_defs, (positive_edges, negative_edges)


def prune_definition(
    current_graph: nx.MultiDiGraph, definition: AgdaDefinition, p_keep: float
) -> Tuple[bool, AgdaDefinition, List[Tuple[Node, Node, EdgeType]]]:
    """
    1) treba je odstraniti povezave, ki predstavljajo približno 1 - p_keep teže
    2) One so posejane malo levo desno po drevesu, zato jih lahko enostavno naključno izberemo
       (Iterativno, če na nekem koraku ni več kandidatov (npr. ws = [4, 3, 3], p = 0.4 in najprej odstranimo
    3) potem končamo (čeprav bi bilo optimalno 4)
       Povezavo odstranimo tako, da v pripadajočem vozlišču (ki jo referencira), izbrišemo ime reference.
       Poleg tega moramo povezavo odstraniti še v grafu.
       Izbris vozlišča = vozlišče.parent <---> vozlišče.otroci
    """
    n_children, referenced_nodes = get_n_children_and_references(definition)
    random.shuffle(n_children)  # so that the choice of leaves later is random
    heapq.heapify(n_children)
    removed_edges = remove_references(
        current_graph, referenced_nodes, p_keep, definition
    )

    did_anything = bool(removed_edges)
    if not did_anything:
        return did_anything, definition, removed_edges
    prune_leaves(n_children, p_keep, definition.name)
    return did_anything, definition, removed_edges


def get_n_children_and_references(definition: AgdaDefinition):
    n_children = []  # (n children present, node)
    referenced_nodes: Dict[Any, List[AgdaNode]] = {}  # reference name: [list of nodes]
    body_node_iterator = definition.body_nodes
    next(body_node_iterator)  # skip root of the body
    for node in body_node_iterator:
        n_children.append((len(node.children), node))
        if node.node_type.is_name():
            reference = node.name
            if reference not in referenced_nodes:
                referenced_nodes[reference] = []
            referenced_nodes[reference].append(node)
    return n_children, referenced_nodes


def get_edge_type(reference_name: str):
    if AgdaDefinition.is_with_definition(reference_name):
        return EdgeType.REFERENCE_IN_BODY_TO_WITH
    elif AgdaDefinition.is_rewrite_definition(reference_name):
        return EdgeType.REFERENCE_IN_BODY_TO_REWRITE
    else:
        return EdgeType.REFERENCE_IN_BODY


def remove_references(
    current_graph, referenced_nodes, p_keep, definition
) -> list[tuple[Node, Node, EdgeType]]:
    referenced_nodes_heap = list(
        (-len(nodes), reference) for reference, nodes in referenced_nodes.items()
    )
    heapq.heapify(referenced_nodes_heap)
    total_degree = sum(-degree for degree, _ in referenced_nodes_heap)
    degree_to_prune = round(
        (1 - p_keep) * sum(-degree for degree, _ in referenced_nodes_heap)
    )
    definition_name = definition.name
    LOGGER_DETAILS.info(
        f"{definition_name}: total degree:{total_degree}, to prune: {degree_to_prune} (p = {p_keep}), edges: {len(referenced_nodes_heap)}"
    )
    removed_edges = []
    if degree_to_prune == 0:
        LOGGER.warning(f"Nothing can be pruned in {definition.name}")
    while degree_to_prune > 0 and referenced_nodes_heap:
        degree, reference_name = heapq.heappop(referenced_nodes_heap)
        degree = -degree
        if degree > degree_to_prune:
            continue
        # the edge might not even exist due to rewrite/with
        edge_type = get_edge_type(reference_name)
        weight = current_graph.get_edge_data(
            definition_name, reference_name, key=edge_type, default={}
        ).get("w", 0)
        message = (
            f"The edge {definition_name}-[ref body]->{reference_name} has weight {weight}, "
            f"but we wound {degree} referencing node(s)"
        )
        if weight < degree - 0.1:
            raise ValueError(message)
        elif weight > degree + 0.1:
            # probably with/rewrite
            LOGGER.warning(message)
        current_graph.remove_edge(definition_name, reference_name, key=edge_type)
        degree_to_prune -= degree
        removed_edges.append((definition_name, reference_name, edge_type))
        for node in referenced_nodes[reference_name]:
            node.node_description = ""
    LOGGER_DETAILS.info(
        f"remaining to prune: {degree_to_prune}, removed edges: {len(removed_edges)}"
    )
    return removed_edges


def prune_leaves(n_children, p_keep, definition_name):
    """
    Odstranimo še preostanek od 1 - p_keep vozlišč: hranimo trenutne liste.
    Naključno izberemo enega od njih in ga odstranimo.
    Tega ne smemo storiti, če pridemo do reference, ki je prisotna! V tem primeru preskočimo.
    """
    # -1: the root of the body should not count
    n_to_prune = round((1 - p_keep) * (len(n_children) - 1))
    while n_to_prune > 0:
        c, node = heapq.heappop(n_children)
        if c > 0:
            LOGGER.warning(f"Only internal nodes left in {definition_name}")
            break
        elif node.node_type.is_name() and node.node_description:
            # reference that is present
            continue
        # prune
        parent = node.parent
        assert isinstance(
            parent, AgdaNode
        )  # even the root of the body is not the root of the definition
        new_children = [child for child in parent.children if child is not node]
        parent.children = new_children
        heapq.heappush(n_children, (len(parent.children), parent))
        node.parent = None
        n_to_prune -= 1


def prepare_internal_cv_dataset(
    train_graph: nx.MultiDiGraph,
    definition_ids: list[Node],
    train_defs: Dict[Node, AgdaDefinition],
    i_fold: int,
    n_folds: int,
    p_def_to_keep: float = 0.0,
    seed: int = 123,
) -> Tuple[
    nx.MultiDiGraph,
    Tuple[Dict[Node, AgdaDefinition], Dict[Node, AgdaDefinition]],
    Tuple[
        List[Tuple[Node, Node, EdgeType]],
        List[Tuple[Node, Node, EdgeType]],
    ],
]:
    # split the definitions into n_folds parts
    rng = random.Random(seed)
    definitions = sorted(definition_ids)  # make a copy ...
    rng.shuffle(definitions)
    validation_defs = definitions[i_fold::n_folds]
    train_train_defs = [d for j, d in enumerate(definitions) if j % n_folds != i_fold]
    return prepare_dataset(
        train_graph,
        train_defs,
        p_test=0.0,
        p_def_to_keep=p_def_to_keep,
        seed=seed,
        train_defs=train_train_defs,
        test_defs=validation_defs,
    )
