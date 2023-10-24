from typing import Literal, Protocol, Dict, Tuple, List, Any
from enum import Enum
import numpy as np
import networkx as nx

from apaa.learning.recommendation.base import BaseRecommender
from apaa.other.helpers import MyTypes, NodeType, EdgeType, Other
from apaa.data.structures.agda_tree import AgdaDefinition

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


Node = MyTypes.NODE
array1d = MyTypes.ARRAY_1D
array2d = MyTypes.ARRAY_2D

LOGGER = Other.create_logger(__file__)


class EdgeEmbeddingScheme(Enum):
    CONCATENATION = "concatenation"
    SUM = "sum"
    MEAN = "mean"


class ScikitLikePredictiveModel(Protocol):
    def __init__(self) -> None:
        raise NotImplementedError()

    def fit(self, xs: array2d, y: array1d) -> None:
        raise NotImplementedError()

    def predict_proba(self, xs: array2d) -> array2d:
        raise NotImplementedError()

    @property
    def classes_(self) -> array1d:
        raise NotImplementedError()


class BaseEdgeEmbeddingRecommender(BaseRecommender):
    def __init__(
        self,
        name: str,
        k: Literal["all"] | int,
        predictive_model: ScikitLikePredictiveModel,
        edge_embedding_scheme: EdgeEmbeddingScheme,
        edge_file: str | None = None,
    ) -> None:
        super().__init__(name, k)
        self.node_embeddings: array2d = np.zeros((0, 0))
        self.node_to_index: Dict[Node, int] = {}
        self.predictive_model = predictive_model
        self.class_index = -1
        self.edge_embedding_scheme = edge_embedding_scheme
        self.edge_file = edge_file

    def embed_nodes(
        self,
        graph: nx.MultiDiGraph,
        definitions: Dict[MyTypes.NODE, AgdaDefinition],
    ) -> Tuple[List[Node], array2d]:
        raise NotImplementedError()

    def embed_edge(self, node1: Node, node2: Node) -> array1d:
        """
        Simple concatenation.
        """
        self._check_known(node1, node2)
        xs1 = self.node_embeddings[self.node_to_index[node1]]
        xs2 = self.node_embeddings[self.node_to_index[node2]]
        if self.edge_embedding_scheme == EdgeEmbeddingScheme.CONCATENATION:
            return np.block([xs1, xs2])
        elif self.edge_embedding_scheme == EdgeEmbeddingScheme.SUM:
            return xs1 + xs2
        elif self.edge_embedding_scheme == EdgeEmbeddingScheme.MEAN:
            return (xs1 + xs2) / 2.0
        else:
            raise ValueError(f"Wrong embedding scheme: {self.edge_embedding_scheme}")

    def embed_edges(self, node1: Node, others: List[Node]) -> array2d:
        return np.vstack([self.embed_edge(node1, other) for other in others])

    def _check_known(self, *nodes: Node) -> None:
        for node in nodes:
            if node not in self.node_to_index:
                raise ValueError(f"Unknown node {node}")

    def fit(
        self,
        graph: nx.MultiDiGraph,
        definitions: Dict[MyTypes.NODE, AgdaDefinition],
        **embed_kwargs: Any,
    ):
        nodes, embeddings = self.embed_nodes(graph, definitions, **embed_kwargs)
        self.graph = graph
        self.node_to_index = {node: i for i, node in enumerate(nodes)}
        self.node_embeddings = embeddings
        self.fit_predictive_model(graph)

    def fit_predictive_model(self, graph: nx.MultiDiGraph):
        es_positive, es_negative = BaseEdgeEmbeddingRecommender.sample_edges(
            graph, self.edge_file
        )
        n_plus = len(es_positive)
        n_minus = len(es_negative)
        if not (es_positive and es_negative):
            raise ValueError(
                f"Need at least one positive (got {n_plus}) "
                f"and at least one negative edge (got {n_minus})."
            )
        u0, v0, _ = es_positive[0]
        e_dim = self.embed_edge(u0, v0).shape[0]
        xs = np.zeros((n_plus + n_minus, e_dim))
        row = 0
        for edges in [es_positive, es_negative]:
            for u, v, _ in edges:
                xs[row] = self.embed_edge(u, v)
                row += 1
        y = np.block([np.ones(n_plus), np.zeros(n_minus)])
        self.predictive_model.fit(xs, y)
        class_one = 1.0
        for i, class_value in enumerate(self.predictive_model.classes_):
            if abs(class_value - class_one) < 0.1:
                self.class_index = i
                break
        if self.class_index < 0:
            raise ValueError(
                f"Class {class_one} was not found in {self.predictive_model.classes_}"
            )

    @staticmethod
    def sample_edges(
        graph: nx.MultiDiGraph,
        edge_file: str | None,
    ) -> Tuple[
        List[Tuple[MyTypes.NODE, MyTypes.NODE, EdgeType]],
        List[Tuple[MyTypes.NODE, MyTypes.NODE, EdgeType]],
    ]:
        """
        Consider definition functions only. Then, just randomly sample.
        Might be more intelligent to sample the close ones.
        Will do that later.
        """
        definitions: List[MyTypes.NODE] = []
        theorem_like_label = NodeType.get_theorem_like_tag(graph)
        for node, label in graph.nodes(data="label"):  # type: ignore
            if label == theorem_like_label:
                definitions.append(node)  # type: ignore
        positive_edges: List[Tuple[str, str, EdgeType]] = []
        n_edges_per_node: Dict[Node, int] = {node: 0 for node in definitions}
        for u, v, e_type in graph.edges(nbunch=definitions, keys=True):  # type: ignore
            if e_type == EdgeType.REFERENCE_IN_BODY:
                positive_edges.append((u, v, e_type))  # type: ignore
                n_edges_per_node[u] += 1  # type: ignore
        negative_edges: List[Tuple[str, str, EdgeType]] = []
        # randomly sample
        for node, count in n_edges_per_node.items():
            negative_edges.extend(
                Other.sample_negative_edges(graph, definitions, node, count)  # type: ignore
            )
        if edge_file is not None:
            LOGGER.info(f"Loading false positives from {edge_file}")
            # chose false positives from internal CV
            false_positives = BaseEdgeEmbeddingRecommender._load_false_positives_file(
                edge_file
            )
            for node, count in n_edges_per_node.items():
                if node in false_positives:
                    negative_edges.extend(
                        map(
                            lambda fp, origin=node: (origin, fp, EdgeType.REFERENCE_IN_BODY),
                            false_positives[node][:count],
                        )
                    )
                else:
                    LOGGER.warning(f"No predictions for for node {node}")
        return positive_edges, negative_edges

    @staticmethod
    def _load_false_positives_file(file: str) -> dict[Node, list[Node]]:
        """
        Parses a file whose blocks are

        NODE;<node name>
        <score>,<node name>
        <score>,<node name>
        <score>,<node name>

        NODE;<node name>
        <score>,<node name>
        <score>,<node name>

        ...
        The first line of each block is the node name, the following lines are false positives
        together with their scores (from classification).
        """
        false_positives: dict[Node, list[Node]] = {}
        with open(file, encoding="utf-8") as f:
            node: Node | None = None
            fps: list[Node] = []
            for line in f:
                line = line.strip()
                if not line:
                    assert node is not None
                    false_positives[node] = fps
                    node = None
                    fps = []
                elif line.startswith("NODE;"):
                    node = line[5:]
                else:
                    _, fp_node = line.split(";")
                    fps.append(fp_node)
        return false_positives

    def predict_one(self, example: AgdaDefinition) -> List[Tuple[float, Node]]:
        # This is inefficient (as compared to predict_one_edge)
        assert isinstance(self.graph, nx.MultiDiGraph)
        candidates = [
            node
            for node in self.node_to_index
            if not self.graph.has_edge(example.name, node, EdgeType.REFERENCE_IN_BODY)
        ]
        return self.predict_edges(example.name, candidates)

    def predict_one_edge(
        self,
        example: AgdaDefinition,
        other: AgdaDefinition,
        nearest_neighbours: List[Tuple[float, Node]] | None = None,
    ) -> float:
        # This is efficient (as compared to predict_one)
        return self.predict_edges(example.name, [other.name])[0][0]

    def predict_edges(self, example_name: Node, candidates: List[Node]):
        xs = self.embed_edges(example_name, candidates)
        y = self.predictive_model.predict_proba(xs)[:, self.class_index]
        pairs = list(zip(y, candidates))
        return self.postprocess_predictions(pairs, False, True)

    @staticmethod
    def create_classifier(
        classifier: Literal["knn", "rf"], kwargs: dict[str, Any] | None
    ) -> ScikitLikePredictiveModel:
        if kwargs is None:
            kwargs = {}
        if classifier == "knn":
            return KNeighborsClassifier(**kwargs)
        elif classifier == "rf":
            return RandomForestClassifier(**kwargs)
        else:
            raise ValueError(f"Unsupported classifier {classifier}")
