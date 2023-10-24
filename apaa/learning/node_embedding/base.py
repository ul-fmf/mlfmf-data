from platform import node
from tkinter import X
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import networkx as nx
from apaa.data.structures.agda_tree import AgdaDefinition

from apaa.other.helpers import MyTypes


Node = MyTypes.NODE
array2d = MyTypes.ARRAY_2D


class NodeEmbeddingBase:
    def __init__(self, name: str):
        self._nodes: List[Node] = []
        self._node_to_index: Dict[Node, int] = {}
        self._node_embeddings: Optional[array2d] = np.zeros((0, 0))
        self.name = name

    @property
    def node_embeddings(self):
        return self._node_embeddings

    @property
    def nodes(self):
        return self._nodes

    def update_fields(self, node_order: List[Node], vectors: array2d):
        assert len(node_order) == vectors.shape[0]
        sorted_nodes = sorted(enumerate(node_order), key=lambda pair: pair[1])
        self._nodes = [n for _, n in sorted_nodes]
        self._node_to_index = {n: i for i, n in enumerate(self._nodes)}
        permutation = [i for i, _ in sorted_nodes]
        self._node_embeddings = vectors[permutation]

    def fit(
        self,
        graph: nx.MultiDiGraph,
        definitions: Dict[Node, AgdaDefinition],
        **kwargs: Any,
    ) -> None:
        nodes, embeddings = self.embed(graph, definitions, **kwargs)
        self.update_fields(nodes, embeddings)

    def embed(
        self,
        graph: nx.MultiDiGraph,
        definitions: Dict[Node, AgdaDefinition],
        **kwargs: Any,
    ) -> Tuple[List[Node], array2d]:
        raise NotImplementedError()


class EmbeddingConcatenator(NodeEmbeddingBase):
    def __init__(self, embedders: Optional[List[NodeEmbeddingBase]] = None):
        if embedders is None or len(embedders) <= 1:
            raise ValueError("You need to specify at least two embedders.")

        super().__init__(" & ".join([e.name for e in embedders]))
        self.embedders = embedders

    def embed(
        self,
        graph: nx.MultiDiGraph,
        definitions: Dict[Node, AgdaDefinition],
        **kwargs: Any,
    ) -> Tuple[List[Node], array2d]:
        for e in self.embedders:
            e.fit(graph, definitions, **kwargs)

        the_node_order = self._get_node_order(self.embedders[0])
        the_nodes = sorted(self.embedders[0].nodes)
        embeddings: List[array2d] = []
        for e in self.embedders:
            sorted_e_nodes = sorted(self.embedders[0].nodes)
            if sorted_e_nodes != the_nodes:
                raise ValueError(f"Different nodes:\n{sorted_e_nodes}\n{the_nodes}")
            node_order = self._get_node_order(e)
            if node_order != the_node_order:
                raise ValueError(f"Different node order:\n{node_order}\n{the_node_order}")
            assert e.node_embeddings is not None
            embeddings.append(e.node_embeddings[node_order])
        nodes = self.embedders[0].nodes
        ordered_nodes = [nodes[i] for i in the_node_order]
        return ordered_nodes, np.hstack(embeddings)

    @staticmethod
    def _get_node_order(e: NodeEmbeddingBase):
        n = len(e.nodes)
        return sorted(range(n), key=lambda i: e.nodes[i])
