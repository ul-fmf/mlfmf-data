from apaa.learning.node_embedding.word import TFIDFEmbedder, DeepWordEmbedder
from apaa.learning.node_embedding.graph import NodeToVecEmbedding
from apaa.learning.node_embedding.base import EmbeddingConcatenator
from apaa.learning.edge_prediction.base import (
    BaseEdgeEmbeddingRecommender,
    EdgeEmbeddingScheme,
)
from apaa.data.structures.agda_tree import AgdaDefinition
from apaa.other.helpers import MyTypes

import networkx as nx

from typing import Literal, Any


Node = MyTypes.NODE
array2d = MyTypes.ARRAY_2D


class TFIDFAndNode2VecEmbeddingRecommender(BaseEdgeEmbeddingRecommender):
    def __init__(
        self,
        k: Literal["all"] | int = 5,
        edge_embedding_scheme: EdgeEmbeddingScheme = EdgeEmbeddingScheme.MEAN,
        classifier: Literal["knn", "rf"] = "rf",
        classifier_kwargs: dict[str, Any] | None = None,
        **node_to_vec_kwargs: Any,
    ):
        super().__init__(
            name="tfidf and node2vec edge embedding",
            k=k,
            predictive_model=BaseEdgeEmbeddingRecommender.create_classifier(
                classifier, classifier_kwargs
            ),
            edge_embedding_scheme=edge_embedding_scheme,
        )
        self.embedder = EmbeddingConcatenator(
            [TFIDFEmbedder(), NodeToVecEmbedding(**node_to_vec_kwargs)]
        )

    def embed_nodes(
        self, graph: nx.MultiDiGraph, definitions: dict[Node, AgdaDefinition]
    ) -> tuple[list[Node], array2d]:
        self.embedder.fit(graph, definitions)
        assert self.embedder.node_embeddings is not None
        return self.embedder.nodes, self.embedder.node_embeddings


class Word2VecAndNode2VecEmbeddingRecommender(BaseEdgeEmbeddingRecommender):
    def __init__(
        self,
        k: Literal["all"] | int = 5,
        edge_embedding_scheme: EdgeEmbeddingScheme = EdgeEmbeddingScheme.MEAN,
        classifier: Literal["knn", "rf"] = "rf",
        classifier_kwargs: dict[str, Any] | None = None,
        deep_word_embedder_kwargs: dict[str, Any] | None = None,
        node_to_vec_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(
            "word2vec and node2vec edge embedding",
            k,
            BaseEdgeEmbeddingRecommender.create_classifier(
                classifier, classifier_kwargs
            ),
            edge_embedding_scheme,
        )
        if deep_word_embedder_kwargs is None:
            deep_word_embedder_kwargs = {}
        if node_to_vec_kwargs is None:
            node_to_vec_kwargs = {}
        self.embedder = EmbeddingConcatenator(
            [
                DeepWordEmbedder(**deep_word_embedder_kwargs),
                NodeToVecEmbedding(**node_to_vec_kwargs),
            ]
        )

    def embed_nodes(
        self, graph: nx.MultiDiGraph, definitions: dict[Node, AgdaDefinition]
    ) -> tuple[list[Node], array2d]:
        self.embedder.fit(graph, definitions)
        assert self.embedder.node_embeddings is not None
        return self.embedder.nodes, self.embedder.node_embeddings
