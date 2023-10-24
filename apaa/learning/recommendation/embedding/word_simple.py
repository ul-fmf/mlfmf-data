import bisect
from typing import Any, Dict, List, Literal, Tuple

import networkx as nx
import numpy as np
import tqdm

from apaa.data.structures.agda_tree import AgdaDefinition
from apaa.learning.node_embedding.word import BagOfWordsEmbedder, TFIDFEmbedder
from apaa.learning.recommendation.base import KNNRecommender, Node
from apaa.learning.recommendation.embedding.base import KNNNodeEmbeddingRecommender
from apaa.learning.recommendation.embedding.numba_distance import jaccard
from apaa.other.helpers import MyTypes

int_array = MyTypes.INT_ARRAY_1D
array1d = MyTypes.ARRAY_1D


class BagOfWordsRecommender(KNNRecommender):
    def __init__(self, k: Literal["all"] | int = 5):
        super().__init__(k=k)
        self.vectorizer = BagOfWordsEmbedder()
        self.vectors: List[Tuple[int_array, int_array]] = []

    def fit(
        self,
        graph: nx.MultiDiGraph,
        definitions: Dict[MyTypes.NODE, AgdaDefinition],
        **kwargs: Any,
    ):
        self.initialize_examples_and_distance_matrix(list(graph.nodes))
        # assert self.distance_matrix is not None
        self.graph = graph
        self.definitions = definitions
        self.vectorizer.fit(graph, definitions, **kwargs)
        self.examples = self.vectorizer.nodes
        matrix = self.vectorizer.sparse_embeddings()
        n = len(self.examples)
        rows_columns: tuple[int_array, int_array] = matrix.nonzero()
        rows, columns = rows_columns
        data: array1d = matrix.data
        for i in range(n):
            i_start = bisect.bisect_left(rows, i)  # n log n instead of n
            i_end = bisect.bisect_right(rows, i)  # but who cares
            self.vectors.append((columns[i_start:i_end], data[i_start:i_end]))
        if self.distance_matrix is not None:
            for i in tqdm.trange(n):
                words1, counts1 = self.vectors[i]
                for j in range(i + 1, n):
                    words2, counts2 = self.vectors[j]
                    d = jaccard(words1, counts1, words2, counts2)
                    self.distance_matrix[i][j] = d
                    self.distance_matrix[j][i] = d
        else:
            print("Warning: distance matrix not initialized")

    def predict_one(self, example: AgdaDefinition) -> List[Tuple[float, Node]]:
        node = example.name
        if node not in self.example_to_i:
            raise ValueError(f"Unknown example {node}")
        i = self.example_to_i[node]
        words, counts = self.vectors[i]
        distances = np.array([jaccard(words, counts, *other) for other in self.vectors])
        return self.postprocess_predictions(
            self.distances_to_tuples(distances), True, False
        )


class TFIDFRecommender(KNNNodeEmbeddingRecommender):
    def __init__(self, k: int = 5, metric: str = "cityblock", **metric_kwargs: Any):
        super().__init__(TFIDFEmbedder(), k=k, metric=metric, **metric_kwargs)
