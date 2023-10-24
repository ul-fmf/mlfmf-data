from typing import List, Tuple, Dict, Any
import py2neo as pn
import contextlib
import time

from apaa.other.helpers import Other


ID = "id"
LOGGER = Other.create_logger(__file__)


@contextlib.contextmanager
def transaction_execution(graph: pn.Graph):
    transaction = graph.begin()
    yield transaction
    graph.commit(transaction)  # type: ignore


@contextlib.contextmanager
def timer(name: str):
    t0 = time.time()
    yield
    t1 = time.time()
    LOGGER.info(f"Time spent for {name}: {t1 - t0}")


class DatabaseManipulation:
    @staticmethod
    def create_graph_connection(path_to_neo: str, authentication: Tuple[str, str]):
        return pn.Graph(path_to_neo, auth=authentication)

    @staticmethod
    def create_uniqueness_constraint(graph: pn.Graph, label: str):
        schema = pn.Schema(graph)
        existing = schema.get_uniqueness_constraints(label)  # type: ignore
        if ID in existing:
            LOGGER.info(f"Uniqueness constraint for {label}.{ID} already exists.")
        else:
            schema.create_uniqueness_constraint(label, ID)  # type: ignore

    @staticmethod
    def create_nodes(
        graph: pn.Graph, data: List[Tuple[List[str], Dict[str, Any]]], batch: int = 1000
    ):
        i_start = 0
        while i_start < len(data):
            i_end = min(len(data), i_start + batch)
            LOGGER.info(f"Processing nodes [{i_start}, {i_end})")
            with transaction_execution(graph) as te:
                for i in range(i_start, i_end):
                    labels, properties = data[i]
                    te.create(pn.Node(*labels, **properties))  # type: ignore
            i_start = i_end

    @staticmethod
    def create_edges(
        graph: pn.Graph,
        data: List[
            Tuple[Tuple[List[str], str], Tuple[List[str], str], str, Dict[str, Any]]
        ],
        batch: int = 1000,
    ):
        i_start = 0
        while i_start < len(data):
            i_end = min(len(data), i_start + batch)
            LOGGER.info(f"Processing edges [{i_start}, {i_end})")
            with transaction_execution(graph) as te:
                matcher = pn.NodeMatcher(graph)
                for i in range(i_start, i_end):
                    (u_labels, u_id), (v_labels, v_id), r_type, properties = data[i]
                    source = DatabaseManipulation.find_node(  # type: ignore
                        matcher, *u_labels, **{ID: u_id}
                    )
                    sink = DatabaseManipulation.find_node(  # type: ignore
                        matcher, *v_labels, **{ID: v_id}
                    )
                    te.create(  # type: ignore
                        pn.Relationship(source, r_type, sink, **properties)
                    )
            i_start = i_end

    @staticmethod
    def find_node(matcher: pn.NodeMatcher, *labels: Any, **properties: Any) -> Any:
        node = matcher.match(*labels, **properties).first()  # type: ignore
        if node is None:
            raise ValueError(
                f"Cannot find the node with labels {labels} and {properties}"
            )
        return node  # type: ignore
