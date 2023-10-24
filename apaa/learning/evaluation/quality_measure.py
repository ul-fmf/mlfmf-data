from apaa.other.helpers import EdgeType
from apaa.other.helpers import Other, MyTypes
from apaa.data.structures import AgdaDefinition

from typing import Any, List, Tuple, Dict, Union
import numpy as np
import networkx as nx
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_auc_score,
)

LOGGER = Other.create_logger(__file__)
Node = MyTypes.NODE


class QualityMeasureRecommender:
    P_PREDICTIONS_IN_ACTUAL = "p predictions in actual"
    RANKS_OF_TRUE = "ranks of true"
    SMOOTH_RANKS_OF_TRUE = "smooth ranks of true"

    def __init__(
        self,
        total_graph: nx.MultiDiGraph,
        train_graph: nx.MultiDiGraph,
        test_definitions: dict[Node, Any],
        k: int,
    ):
        self.kg = QualityMeasureRecommender.compute_test_graph(
            total_graph, train_graph, test_definitions
        )
        self.k = k
        self.n_added = 0
        self.n_fully_added = 0
        self.scores: dict[str, list[list[float]]] = {
            QualityMeasureRecommender.P_PREDICTIONS_IN_ACTUAL: [],
            QualityMeasureRecommender.RANKS_OF_TRUE: [],
            QualityMeasureRecommender.SMOOTH_RANKS_OF_TRUE: [],
        }

    def update(self, fact: MyTypes.NODE, predictions: List[Tuple[float, MyTypes.NODE]]):
        actual_neighbours, top_neighbours = self._find_actual_neighbours(fact)
        if not actual_neighbours:
            LOGGER.warning(
                f"{fact} has no neighbours and it will be skipped when computing stats"
            )
            return False, top_neighbours
        p = 0.0
        ranks_actual: list[float] = []
        smoothed_actual: list[float] = []
        similarity_of_kth = predictions[self.k - 1][0]
        for i, (similarity, neighbour) in enumerate(predictions):
            if neighbour in actual_neighbours:
                if i < self.k:
                    p += 1.0
                ranks_actual.append(i)
                if similarity > 0.0:
                    smoothed_actual.append(max(1.0, similarity_of_kth / similarity))
        if len(ranks_actual) != len(actual_neighbours):
            raise ValueError(
                f"fact: {fact}\n"
                f"ranks actual: {ranks_actual}\n"
                f"neigh. actual: {actual_neighbours}\npred: {predictions}"
            )
        self.scores[QualityMeasureRecommender.P_PREDICTIONS_IN_ACTUAL].append(
            [p / self.k]
        )
        self.scores[QualityMeasureRecommender.RANKS_OF_TRUE].append(ranks_actual)
        if smoothed_actual:
            self.scores[QualityMeasureRecommender.SMOOTH_RANKS_OF_TRUE].append(
                smoothed_actual
            )
            self.n_fully_added += 1
        else:
            LOGGER.warning(
                f"Smooth stats skipped. The {self.k}-th neighbour of {fact}"
                " and/or all its actual neighborshas have similarity 0."
            )
        self.n_added += 1
        return True, top_neighbours

    def _find_actual_neighbours(self, fact: MyTypes.NODE):
        actual_neighbours: Dict[MyTypes.NODE, float] = {}
        n_top = 20
        top_neighbours = [(0.0, ":)") for _ in range(n_top)]
        n_updates = 0
        if fact not in self.kg:
            raise ValueError(f"Fact {fact} not in the graph: {sorted(self.kg.nodes())}.")
        for node, edges in self.kg[fact].items():
            if EdgeType.REFERENCE_IN_BODY in edges and AgdaDefinition.is_normal_definition(node):
                w = edges[EdgeType.REFERENCE_IN_BODY]["w"]
                actual_neighbours[node] = w
                if w > top_neighbours[-1][0]:
                    n_updates += 1
                    # E[n_updates] = n_top (1 + ln (n_train / n_top)) times.
                    top_neighbours[-1] = (w, node)
                    top_neighbours.sort(reverse=True)
        return actual_neighbours, top_neighbours[:n_updates]

    def __str__(self):
        if self.n_added == 0:
            return "Recommender quality measures:\nNo test examples --> no statistics"
        p_actual = np.mean(self.scores[QualityMeasureRecommender.P_PREDICTIONS_IN_ACTUAL])
        min_rank = np.mean(
            [
                np.min(scores)
                for scores in self.scores[QualityMeasureRecommender.RANKS_OF_TRUE]
            ]
        )
        mean_rank = np.mean(
            [
                np.mean(scores)
                for scores in self.scores[QualityMeasureRecommender.RANKS_OF_TRUE]
            ]
        )
        if self.n_fully_added > 0:
            min_smooth = np.mean(
                [
                    np.min(scores)
                    for scores in self.scores[
                        QualityMeasureRecommender.SMOOTH_RANKS_OF_TRUE
                    ]
                ]
            )
            mean_smooth = np.mean(
                [
                    np.mean(scores)
                    for scores in self.scores[
                        QualityMeasureRecommender.SMOOTH_RANKS_OF_TRUE
                    ]
                ]
            )
        else:
            min_smooth = np.nan
            mean_smooth = np.nan
        lines = [
            "Recommender quality measures:",
            f"  - Number of test examples: {self.n_added} (fully added: {self.n_fully_added})",
            f"  - Proportion of predictions that are in actual neighbours: {p_actual}",
            f"  - The average minimal rank (>= 0) of the actual neighbour: {min_rank}",
            f"  - The average minimal smooth rank (>= 1.0) of the actual neighbour: {min_smooth}",
            f"  - The average mean rank (>= 0) of the actual neighbour: {mean_rank}",
            f"  - The average mean smooth rank (>= 1.0) of the actual neighbour: {mean_smooth}",
        ]
        return "\n".join(lines)

    @staticmethod
    def compute_test_graph(
        total_graph: nx.MultiDiGraph,
        train_graph: nx.MultiDiGraph,
        test_definitions: dict[Node, Any],
    ) -> nx.MultiDiGraph:
        """
        Test graph is the graph that contains
        - test definitions,
        - edges from test definitions that are not present in the train graph.

        (and sinks of the edges above).
        """
        test_graph = nx.MultiDiGraph()
        for u, v, e_type, w in total_graph.edges(keys=True, data="w"):
            if u not in test_definitions or e_type != EdgeType.REFERENCE_IN_BODY:
                continue
            if not train_graph.has_edge(u, v, e_type):
                test_graph.add_edge(u, v, e_type, w=w)
        return test_graph


class QualityMeasureClassification:
    def __init__(
        self,
        thresholds: Union[str, float, List[float]] = "default",
        zero_division: Union[str, int] = 0,
    ):
        self.true_values = []
        self.predicted_values = []
        if thresholds == "default":
            thresholds = [0.5, 0.1, 0.25, 0.75, 0.9]
        self.theta = [thresholds] if isinstance(thresholds, float) else thresholds
        self.zd = zero_division

    def update(self, true_value, predicted_value):
        self.true_values.append(true_value)
        self.predicted_values.append(predicted_value)

    def __str__(self):
        n = len(self.true_values)
        if n == 0:
            return (
                "Classification quality measures:\nNo test examples --> no statistics"
            )
        labels = sorted(set(self.true_values))  # one can never be too sure
        lines = [
            "Classification quality measures:",
            f"* Number of examples: {len(self.true_values)}",
            "* Threshold independent measures:",
            f"   - area under ROC: {roc_auc_score(self.true_values, self.predicted_values)}",
            "* Threshold dependent measures:",
        ]
        for theta in self.theta:
            lines.append(f"  threshold = {theta}")
            thresholded_values = [int(p >= theta) for p in self.predicted_values]
            conf_matrix = confusion_matrix(
                self.true_values, thresholded_values, labels=labels
            ).tolist()
            lines.extend(
                [
                    f"   - accuracy: {accuracy_score(self.true_values, thresholded_values)}",
                    f"   - precision: {precision_score(self.true_values, thresholded_values, zero_division=self.zd, pos_label=1)}",
                    f"   - recall: {recall_score(self.true_values, thresholded_values, zero_division=self.zd, pos_label=1)}",
                    f"   - F1: {f1_score(self.true_values, thresholded_values, zero_division=self.zd, pos_label=1)}",
                    f"   - confusion matrix: {conf_matrix} (labels: {labels})",
                ]
            )
        return "\n".join(lines)
