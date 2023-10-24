import math
from enum import Enum
from typing import Any, Dict, Literal, Optional

import networkx as nx

from apaa.data.structures.agda_tree import AgdaDefinition
from apaa.learning.recommendation.base import BaseRecommender
from apaa.other.helpers import MyTypes

Node = MyTypes.NODE


class NodeWeightScheme(Enum):
    CONSTANT = "constant"
    INVERSE_DEGREE = "inverse degree"


class EdgeWeightScheme(Enum):
    CONSTANT = "constant"
    DEGREE = "degree"
    LOG2_DEGREE = "log2 degree"


class TwoHops(BaseRecommender):
    """
    Recommendations of this model are motivated by the fact that

    'Two similar methods should reference the same stuff.'

    """

    def __init__(
        self,
        k: Literal["all"] | int = 5,
        node_weights: NodeWeightScheme = NodeWeightScheme.CONSTANT,
        edge_weights: EdgeWeightScheme = EdgeWeightScheme.DEGREE,
    ):
        super().__init__("two hops", k)
        self.reversed_graph: Optional[nx.MultiDiGraph] = None
        self.middle_node_weights: Optional[Dict[str, float]] = None
        self.node_weighting_scheme: NodeWeightScheme = node_weights
        self.edge_weighting_scheme: EdgeWeightScheme = edge_weights

        self.node_degrees: dict[Node, float] = {}

    def fit(
        self, graph: nx.MultiDiGraph, definitions: Dict[Node, AgdaDefinition], **kwargs
    ):
        # The forbidden edges are:
        #
        # train node  -- declaration or body --> test node
        # train node <-- some parts of body -- test node
        # The graph should already be train graph!
        self.graph = graph
        self.definitions = definitions
        self.reversed_graph = self.graph.copy().reverse()  # for faster look-ups
        degrees = self.reversed_graph.degree()
        for train_node in self.reversed_graph:
            self.node_degrees[train_node] = degrees(train_node, "w")

    def predict_one(self, example: AgdaDefinition):
        """
        Distance between example and training node T is 1 / the weight of paths of form
        example - reference -> x <- reference - T, for some x in training set.
        The total weight of paths is defined as

        sum_{path = example - r1 -> x <- r2 - T} r1.w * r2.w

        where r_i.w is the weight on the edge r_i,
        and r1 and x meet the constraints mentioned above.

        If there is no path between example and T,
        the score of T is 2 (since 1 / weight < 1).
        The nodes at the same distance are sorted alphabetically.

        :param example:
        :return:
        """
        w_paths: dict[Node, float] = {node: 0.0 for node in self.graph}
        the_name = example.name
        for x, edges1 in self.graph[the_name].items():
            w_x = self.node_weight(x)
            w_to_x = self.edge_weight(TwoHops._get_total_weight_of_edges(edges1))
            for t, edges2 in self.reversed_graph[x].items():
                for edge_params in edges2.values():
                    w_from_x = self.edge_weight(edge_params.get("w", 1.0))
                    w_paths[t] += self.path_weight(w_x, w_to_x, w_from_x)
        candidates = [(1 / w if w > 0 else 2.0, node) for node, w in w_paths.items()]
        return self.postprocess_predictions(candidates, True, False)

    @staticmethod
    def _get_total_weight_of_edges(edges: Dict[Any, Dict[str, Any]]):
        return sum(e_props.get("w", 1.0) for e_props in edges.values())

    def node_weight(self, node: Node) -> float:
        if self.node_weighting_scheme == NodeWeightScheme.CONSTANT:
            return 1.0
        elif self.node_weighting_scheme == NodeWeightScheme.INVERSE_DEGREE:
            return 1.0 / max(1.0, self.node_degrees[node])
        else:
            raise ValueError(f"Unknown option: {self.node_weighting_scheme}")

    def edge_weight(self, weight: float) -> float:
        if self.edge_weighting_scheme == EdgeWeightScheme.CONSTANT:
            return 1.0
        elif self.edge_weighting_scheme == EdgeWeightScheme.DEGREE:
            return weight
        elif self.edge_weighting_scheme == EdgeWeightScheme.LOG2_DEGREE:
            return math.log(weight, 2.0) + 1.0  # otherwise: log(1) = 0
        else:
            raise ValueError(f"Unknown option: {self.edge_weighting_scheme}")

    def path_weight(self, w_node: float, w_e1: float, w_e2: float) -> float:
        if self.edge_weighting_scheme == EdgeWeightScheme.LOG2_DEGREE:
            return w_node * (w_e1 + w_e2)
        else:
            return w_node * w_e1 * w_e2
