import networkx as nx
import pickle
from typing import Optional, Dict, List, Set, Any
from apaa.other.helpers import NetworkxWrappers, Other, MyTypes

LOGGER = Other.create_logger(__file__)
Node = MyTypes.NODE


class GraphProperties:
    IN_DEGREE = "in_degree"

    def __init__(
        self, graph: nx.MultiDiGraph, weight: Optional[str] = None, debug: bool = False
    ):
        self.graph = graph
        self.weight = weight if weight is not None else "bla"
        self._node_statistics: Dict[
            str, Dict[Node, float]
        ] = {}  # {measure: {node: value, ...}, ...}
        self._graph_statistics: Dict[str, float] = {}  # {measure: value, ...}
        self._weakly_connected_components: List[List[str]] = []
        self._strongly_connected_components: List[Set[Any]] = []
        self._max_cycles: List[List[str]] = []
        self.compute_properties(debug)

    @property
    def node_statistics(self):
        return self._node_statistics

    @property
    def graph_statistics(self):
        return self._graph_statistics

    @property
    def max_cycles(self):
        return self._max_cycles

    @property
    def strong_components(self):
        return self._strongly_connected_components

    @property
    def weak_components(self):
        return self._weakly_connected_components

    def compute_properties(self, debug: bool):
        if debug:
            LOGGER.warning("Graph properties will be computed in debug mode.")
            for i in range(2):
                self._node_statistics[f"dummy{i + 1}"] = {
                    node: 0.12345
                    for node in NetworkxWrappers.graph_nodes(self.graph)
                }
                self._graph_statistics[f"g_dummy{i + 1}"] = 3.1415
            return
        self.compute_degree_related()
        self.compute_eigen_related()
        self.compute_betweenness()
        self.compute_connected_components()
        self.compute_cycles()
        self.correct_statistics()

    def correct_statistics(self):
        for values in self.node_statistics.values():
            GraphProperties.correct_values(values)
        GraphProperties.correct_values(self.graph_statistics)

    @staticmethod
    def correct_values(values: Dict[Any, float]):
        for node, value in values.items():
            values[node] = max(0.0, float(value))
            if values[node] < 10**-16:
                values[node] = 0.0

    @staticmethod
    def prepare_measure_dict(
        graph: nx.Graph, default: float = 0.0
    ) -> Dict[Node, float]:
        return {node: default for node in NetworkxWrappers.graph_nodes(graph)}

    def compute_degree_related(self):
        """
        Computes in-, out- and total degree.
        """
        LOGGER.info("Computing degree-related centrality measures.")
        in_degrees = GraphProperties.prepare_measure_dict(self.graph)
        out_degrees = GraphProperties.prepare_measure_dict(self.graph)
        degrees = GraphProperties.prepare_measure_dict(self.graph)
        highest_degree: int = 0
        for a, b, w in self.graph.edges(data=self.weight, default=1.0):
            out_degrees[a] += w  # type: ignore
            in_degrees[b] += w  # type: ignore
            highest_degree = max(highest_degree, w)  # type: ignore
        for node in degrees:
            degrees[node] = in_degrees[node] + out_degrees[node]
        self._node_statistics[GraphProperties.IN_DEGREE] = in_degrees
        self._node_statistics["out_degree"] = out_degrees
        self._node_statistics["degree"] = degrees
        self._graph_statistics[
            "star_shape_level"
        ] = GraphProperties._compute_stare_shape_level(
            list(degrees.values()), highest_degree
        )
        self._graph_statistics[
            "tendency_to_make_hub"
        ] = GraphProperties._compute_tendency_to_make_hub(list(degrees.values()))
        LOGGER.info("Computed degree-related centrality measures.")

    @staticmethod
    def _compute_stare_shape_level(degrees: List[int | float], alfa: int):
        """
        The traditional star level is computed in simple graphs (no parallel edges, i.e., no weights of the edges).
        In that case, the normalisation factor is H(n) = (n - 1)(n - 2), which is the score of the
        star-graph (V, E) = (range(n), {(0, i) for i in range(1, n)}).

        Instead, we use the factor H(n, alfa) = alfa (n - 1)(n - 2),
        which is the maximal possible value of the multi-graphs on n nodes,
        where the highest weight is alpha.

        :param degrees:
        :return:
        """
        if not degrees:
            return 0.0
        max_degree = max(degrees)
        n_nodes = len(degrees)
        h = alfa * (n_nodes - 1) * (n_nodes - 2)
        if h == 0:
            return float(alfa > 0)
        return sum(max_degree - d for d in degrees) / h

    @staticmethod
    def _compute_tendency_to_make_hub(degrees: List[int | float]):
        if not degrees:
            return 0.0
        return sum(d**2 for d in degrees) / sum(degrees)

    def compute_eigen_related(self):
        LOGGER.info("Computing eigen-related centrality measures 1/2.")
        self._node_statistics["pagerank"] = nx.pagerank(self.graph, weight=self.weight)  # type: ignore
        LOGGER.info("Computing eigen-related centrality measures 2/2.")
        self._node_statistics["eigen_centrality"] = nx.eigenvector_centrality_numpy(  # type: ignore
            self.graph, weight=self.weight
        )
        LOGGER.info("Computed eigen-related centrality measures.")

    def compute_connected_components(self):
        LOGGER.info("Computing connected components 1/2.")
        self._weakly_connected_components = list(
            nx.weakly_connected_components(self.graph)  # type: ignore
        )
        LOGGER.info("Computing connected components 2/2.")
        self._strongly_connected_components = list(
            nx.strongly_connected_components(self.graph)  # type: ignore
        )
        LOGGER.info("Computed connected components.")

    def compute_betweenness(self):
        LOGGER.info("Computing betweenness-related centrality measures 1/2.")
        self._node_statistics["betweenness"] = nx.betweenness_centrality(  # type: ignore
            self.graph, weight=self.weight
        )
        LOGGER.info("Computing betweenness-related centrality measures 2/2.")
        undirected_version = self.graph.to_undirected()  # type: ignore
        current_flow_betweenness = {n: -1.0 for n in self.graph}  # {}
        # for component in nx.connected_components(undirected_version):
        #     if len(component) == 1:
        #         LOGGER.warning(f"Default betweenness for the component {component}")
        #         fresh = {n: 0.0 for n in component}
        #     else:
        #         subgraph = nx.subgraph(undirected_version, component)
        #         fresh = nx.current_flow_betweenness_centrality(subgraph, weight=self.weight)
        #     current_flow_betweenness = {**current_flow_betweenness, **fresh}
        self._node_statistics["current_flow_betweenness"] = current_flow_betweenness
        LOGGER.info("Computed betweenness-related centrality measures.")

    def compute_cycles(self):
        LOGGER.info("Computing cycles")
        max_cycle_length = 0
        cycles_per_node = GraphProperties.prepare_measure_dict(self.graph)
        for cycle in nx.simple_cycles(self.graph):  # type: ignore
            for node in cycle:  # type: ignore
                cycles_per_node[node] += 1.0  # type: ignore
            if len(cycle) > max_cycle_length:  # type: ignore
                self._max_cycles = [cycle]
                max_cycle_length = len(cycle)  # type: ignore
            elif len(cycle) == max_cycle_length:  # type: ignore
                self._max_cycles.append(cycle)  # type: ignore
        self._node_statistics["n_cycles"] = cycles_per_node
        self._graph_statistics["n_cycles"] = sum(cycles_per_node.values())
        LOGGER.info("Finished cycles")

    def dump(self, file: str):
        with open(file, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file: str) -> "GraphProperties":
        with open(file, "rb") as f:
            return pickle.load(f)
