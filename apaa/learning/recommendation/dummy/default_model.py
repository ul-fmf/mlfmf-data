from typing import Literal, Union, List, Dict, Any

import numpy as np
import networkx as nx

from apaa.data.structures import AgdaDefinition
from apaa.learning.recommendation.base import BaseRecommender
from apaa.other.helpers import Other, MyTypes


Node = MyTypes.NODE


class DummyRecommender(BaseRecommender):
    def __init__(self, k: Literal["all"] | int = 5, random_seed: int = 12345):
        super().__init__("default model", k)
        self.prediction: Node | None = None
        self.other_train_examples: List[MyTypes.NODE] = []
        self.seed = random_seed

    def fit(
        self,
        graph: nx.MultiDiGraph,
        definitions: Dict[MyTypes.NODE, AgdaDefinition],
        **kwargs: Any
    ):
        # find the node with the largest in-degree
        in_degrees = DummyRecommender.compute_in_degrees(graph)
        max_degree = 0
        # when computing arg max (in degree), we assume that most popular node
        # will not change if we remove the edges from test nodes
        for node, degree in in_degrees.items():
            if degree > max_degree:
                self.prediction = node
                max_degree = degree
        for node in graph:
            if node != self.prediction:
                self.other_train_examples.append(node)

    @staticmethod
    def compute_in_degrees(graph: nx.MultiDiGraph):
        in_degrees: dict[Node, float] = {}
        for _, sink, weight in graph.edges(data="w", default=1.0):
            in_degrees[sink] = weight + in_degrees.get(sink, 0.0)
        return in_degrees

    def predict(self, example_s: Union[AgdaDefinition, List[AgdaDefinition]]):
        if self.prediction is None:
            raise ValueError("Fit the model first! Most popular node is unknown.")
        return super().predict(example_s)

    def predict_one(self, example: AgdaDefinition):
        """
        Similarity with the most popular node is 0.75, and 0.25 otherwise.

        :param example:
        :return:
        """
        assert self.prediction is not None
        answer = [(0.75, self.prediction)]
        n_other_predictions = self.n_predictions(1 + len(self.other_train_examples)) - 1
        for i in range(n_other_predictions):
            answer.append((0.25 + np.random.rand() * 0.25, self.other_train_examples[i]))
        return self.postprocess_predictions(answer, False, True)

    @staticmethod
    def load(file: str) -> "DummyRecommender":
        return Other.class_checker(BaseRecommender.unpickle(file), DummyRecommender)
