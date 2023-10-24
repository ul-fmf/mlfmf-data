import networkx as nx
import numpy as np
import numba

from apaa.other.helpers import Other

LOGGER = Other.create_logger(__file__)


def efficient_transiction_matrix(graph: nx.Graph, weight_key: str = "w"):
    """
    Computes the sparse transition matrix of the graph. Every row in the list
    is a list of cumulative probabilities.

    Node name must be an integer 0 <= name < |nodes|.
    """
    matrix: list[tuple[list[int], list[float]]] = []
    for node in range(len(graph.nodes)):
        neighbours = list(graph[node])
        weights = [graph.edges[(node, n)][weight_key] for n in neighbours]
        total = sum(weights)
        cumulative = np.cumsum(weights) / total
        matrix.append((neighbours, cumulative))
    return matrix


@numba.njit
def bisect_left(values: np.array, x: float):
    i = 0
    j = len(values)
    while i < j:
        h = (i + j) // 2
        if values[h] < x:
            i = h + 1
        else:
            j = h
    return i


@numba.njit
def get_walks(transition_matrix: list[tuple[list[int], list[float]]], n_walks: int, walk_length: int, rng: np.random.Generator):
    """
    Returns a list of random walks of length walk_length.
    """
    walks = []
    n_nodes = len(transition_matrix)
    for i_node in range(n_nodes):
        if i_node % 1000 == 0:
            print("Walking from node", i_node, "; total nodes:", n_nodes)
        for _ in range(n_walks):
            walk = -np.ones(walk_length, dtype=np.int32)
            walk[0] = i_node
            random_steps = rng.random(walk_length - 1)
            for i_step in range(walk_length - 1):
                i_neighbour = bisect_left(
                    transition_matrix[walk[i_step]][1],
                    random_steps[i_step]
                )
                walk[i_step + 1] = transition_matrix[walk[i_step]][0][i_neighbour]
            walks.append(walk)
    return walks


class Walker:
    def __init__(self, n_walks: int = 100, walk_length: int = 100, seed: int = 1234):
        self.n_walks = n_walks
        self.walk_length = walk_length
        self.rng = np.random.default_rng(seed)
    
    def get_walks(self, graph: nx.Graph):
        transition_matrix = efficient_transiction_matrix(graph)
        LOGGER.debug("Transition matrix computed.")
        walks = get_walks(transition_matrix, self.n_walks, self.walk_length, self.rng)
        walks = [Walker._postprocess_walk(walk) for walk in walks]
        return walks
    
    @staticmethod
    def _postprocess_walk(walk):
        a = walk.tolist()
        i0 = i = len(a) - 1
        while a[i] < -0.5:
            i -= 1
        if i < i0:
            return a[:i + 1]
        else:
            return a


def test_bisection():
    import bisect
    for n in range(2, 18):
        a = np.arange(n)
        for x in np.random.rand(100):
            assert bisect_left(a, x) == bisect.bisect_left(a, x)
        for x in a:
            assert bisect_left(a, x) == bisect.bisect_left(a, x)
    print("alles gut")


if __name__ == "__main__":
    test_bisection()
    g0 = nx.Graph()
    g0.add_edge(0, 1, w=1)
    g0.add_edge(1, 2, w=2)
    g0.add_edge(0, 2, w=3)
    m = efficient_transiction_matrix(g0)
    for w in get_walks(m, 2, 6, np.random.default_rng(1234)):
        print(w)
