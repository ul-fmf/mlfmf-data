from typing import Dict, Literal, Optional, List, Any, Set, Tuple
from apaa.data.structures.agda_tree import AgdaDefinition

import networkx as nx
import numpy as np
from scipy.spatial import distance as ssd

from apaa.learning.node_embedding.word import DeepWordEmbedder, WordFrequencyWeight
from apaa.learning.recommendation.embedding.base import KNNNodeEmbeddingRecommender
from apaa.other.helpers import EdgeType, Other, MyTypes


array1d = MyTypes.ARRAY_1D
array2d = MyTypes.ARRAY_2D
Node = MyTypes.NODE

LOGGER = Other.create_logger(__file__)


class WordEmbeddingRecommender(KNNNodeEmbeddingRecommender):
    """
    Nearest neighbours in the embedding space of definitions
    represented as flat sequences of words.
    """

    def __init__(
        self,
        k: Literal["all"] | int = 5,
        metric: str = "cosine",
        words: Optional[List[str]] = None,
        word_embeddings: Optional[MyTypes.ARRAY_2D] = None,
        word_frequency_weight: WordFrequencyWeight = WordFrequencyWeight.COUNT,
        **metric_kwargs: Any,
    ):
        super().__init__(
            DeepWordEmbedder(
                words=words,
                word_embeddings=word_embeddings,
                word_frequency_weight=word_frequency_weight,
            ),
            k=k,
            metric=metric,
            **metric_kwargs,
        )


class EmbeddingAnalogiesRecommender(WordEmbeddingRecommender):
    """
    For a given node N finds it best fit via analogies
    with existing pairs N' --> R', so that the diagram

    N  -----> solution
    |         |
    |         |
    N' -----> R'

    approximately commutes (-N + N' + R' = cca. solution)
    """

    def __init__(
        self,
        k: Literal["all"] | int = 5,
        metric: str = "cosine",
        words: Optional[List[str]] = None,
        word_embeddings: Optional[array2d] = None,
        word_frequency_weight: WordFrequencyWeight = WordFrequencyWeight.COUNT,
        analogy_alpha: float = 0.0,
        **metric_kwargs: Any,
    ):
        super().__init__(
            k=k,
            metric=metric,
            words=words,
            word_embeddings=word_embeddings,
            word_frequency_weight=word_frequency_weight,
            **metric_kwargs,
        )
        self.analogy_alpha = analogy_alpha
        self.train_kg: Optional[nx.MultiDiGraph] = None
        self.kg: Optional[nx.MultiDiGraph] = None
        self.direct_candidates: Optional[
            Dict[MyTypes.NODE, List[Tuple[Node, Node]]]
        ] = None
        self.node_to_module: Optional[Dict[Node, Node]] = None
        self.dim: Optional[int] = None

    def fit(
        self,
        graph: nx.MultiDiGraph,
        definitions: Dict[Node, AgdaDefinition],
        **kwargs: Any,
    ):
        # do standard document embeddings
        self.embed_documents(graph, definitions, **kwargs)
        assert self.embeddings is not None
        self.dim = self.embeddings.shape[1]
        # and store the knowledge graph
        assert self.examples is not None
        training_nodes = set(self.examples)
        n2m, m2n = self._compute_node_to_module(training_nodes)
        self._compute_analogy_candidates(training_nodes, m2n)
        self.node_to_module = {node: module for node, (_, module) in n2m.items()}

    def predict_one(self, example: AgdaDefinition) -> List[Tuple[float, Node]]:
        embedding, distances = self.compute_distances(example)
        # 1. Find all the candidates
        assert self.node_to_module is not None
        assert self.direct_candidates is not None
        assert self.embeddings is not None
        assert self.dim is not None
        all_candidates: List[Tuple[Node, Node]] = []
        current_module = example.name  # a module only after entering the loop
        while current_module in self.node_to_module:
            current_module = self.node_to_module[current_module]
            all_candidates.extend(self.direct_candidates[current_module])
        if not all_candidates:
            error_values = np.inf * np.ones(self.embeddings.shape[0])
        else:
            # 2. compute target vectors: -ref + def + example
            target_vectors = np.zeros((len(all_candidates), self.dim))
            for i, (definition, reference) in enumerate(all_candidates):
                target_vectors[i] = (
                    self.embeddings[self.example_to_i[reference]]
                    - self.embeddings[self.example_to_i[definition]]
                )
            target_vectors = target_vectors + embedding
            # 3. compute all the distances and choose the best one
            fitness_values = np.min(
                ssd.cdist(
                    self.embeddings,
                    target_vectors,
                    metric=self.metric,
                    **self.metric_kwargs,
                ),
                axis=1,
            )
            # 4. compute final values
            error_values = fitness_values + self.analogy_alpha * distances
        return self.postprocess_predictions(
            self.distances_to_tuples(error_values), True, False
        )

    def _compute_node_to_module(
        self, training_nodes: Set[Node]
    ) -> tuple[dict[Node, tuple[EdgeType, Node]], dict[Node, list[Node]]]:
        """
        Computes a module that contains given node
        (which is either a definition or a (sub)module).
        Then, it also computes all the definitions that are present in a given module.
        :return: (n2m, m2n) where
            n2m = {node: (edge_type, module), ...} where
                  edge type is the edge between module --> node
                  in the training graph.
            m2n= {module: [def, ...], ...} where defs are directly defined in module
        """
        n2m: dict[Node, tuple[EdgeType, Node]] = {}
        reversed_g = nx.reverse(self.graph)
        for source, sink, e_type in reversed_g.edges(keys=True):
            if e_type == EdgeType.DEFINES or e_type == EdgeType.CONTAINS:
                n2m[source] = (e_type, sink)
        m2n: dict[Node, list[Node]] = {}  # module: definitions directly in this module
        for node, (e_type, module) in n2m.items():
            if module not in m2n:
                m2n[module] = []
            if e_type == EdgeType.DEFINES and node in training_nodes:
                m2n[module].append(node)
        return n2m, m2n

    def _compute_analogy_candidates(
        self,
        training_nodes: Set[Node],
        m2n: Dict[Node, List[Node]],
    ):
        """
        :param training_nodes:
        :param m2n: contains only training definitions
        :return:
        """
        self.direct_candidates = {}
        for module, direct_definitions in m2n.items():
            self.direct_candidates[module] = []
            for direct_def in direct_definitions:
                for ref, edges_to_ref in self.graph[direct_def].items():
                    if (
                        ref in training_nodes
                        and EdgeType.REFERENCE_IN_BODY in edges_to_ref
                    ):
                        self.direct_candidates[module].append((direct_def, ref))
