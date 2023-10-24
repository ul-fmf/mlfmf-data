from typing import List, Dict, Tuple, Any
import multiprocessing

import numpy as np
import networkx as nx
from node2vec import Node2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from gensim.models import Word2Vec
from apaa.data.structures.agda_tree import AgdaDefinition
from apaa.learning.node_embedding.base import NodeEmbeddingBase, Node, array2d
from apaa.learning.node_embedding.walk_generation import Walker

from apaa.other.helpers import MyTypes, Other


LOGGER = Other.create_logger(__file__)
array1d = MyTypes.ARRAY_1D


class NodeToVecEmbedding(NodeEmbeddingBase):
    def __init__(
        self,
        # walk-generating parameters
        num_walks: int = 100,
        walk_length: int = 100,
        p: float = 1.0,
        q: float = 1.0,
        # embedding parameters
        vector_size: int = 64,
        window: int = 5,
        epochs: int = 10,
        workers=-1,
    ):
        super().__init__("node2vec")
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.p = p
        self.q = q
        self.workers = NodeToVecEmbedding._get_workers(workers)
        LOGGER.info(f"Using {self.workers} workers")
        self.embedding_parameters = {
            "vector_size": vector_size,
            "window": window,
            "epochs": epochs,
            "sg": 1,
            "workers": self.workers,
        }
    
    @staticmethod
    def _get_workers(workers: int):
        if workers <= 0:
            available = multiprocessing.cpu_count()
            return max(1, available - 1)
        return workers

    def embed(
        self,
        graph: nx.MultiDiGraph,
        definitions: Dict[Node, AgdaDefinition],
        **kwargs: Any,
    ) -> Tuple[List[Node], array2d]:
        is_big = len(graph.nodes()) >= 10**5
        u_graph, node_dictionary = NodeToVecEmbedding.to_undirected(
            graph, use_int_names=is_big
        )
        embedding_file = self.get_vector_file_name(
            graph, kwargs, self.num_walks, self.walk_length, self.p, self.q
        )
        if os.path.exists(embedding_file):
            LOGGER.info(f"Loading embeddings from {embedding_file}")
            model = Word2Vec.load(embedding_file)
        elif not is_big:
            node2vec = Node2Vec(
                u_graph,
                num_walks=self.num_walks,
                walk_length=self.walk_length,
                p=self.p,
                q=self.q,
                workers=self.workers,
            )
            model: Word2Vec = node2vec.fit(**self.embedding_parameters)
            # LOGGER.info(f"Saving embeddings to {embedding_file}")
            # model.save(embedding_file)
        else:
            LOGGER.info(
                f"Graph is large: {len(graph.nodes())} nodes. "
                "Computing walks more efficiently."
            )
            if max(abs(self.p - 1), abs(self.q - 1.0)) > 1e-5:
                raise ValueError("p and q must be 1.0 for large graphs.")
            model = self.compute_efficient_walks(u_graph)
        node_order: List[str] = model.wv.index_to_key  # type: ignore
        vectors: array2d = model.wv.vectors  # type: ignore
        assert isinstance(vectors, np.ndarray)
        original_names = [node_dictionary[node] for node in node_order]
        # filtering names (keeping only entries) saves up some % of space
        # but we migth face some problems with missing embeddings later
        return original_names, vectors

    def compute_efficient_walks(self, u_graph):
        walks = Walker(self.num_walks, self.walk_length).get_walks(u_graph)
        model = Word2Vec(walks, **self.embedding_parameters)
        return model

    @staticmethod
    def get_vector_file_name(graph: nx.MultiDiGraph, kwargs: Any, *args) -> str:
        library_identifier = f"n{len(graph.nodes())}e{len(graph.edges())}"
        return f"n2v{library_identifier}{NodeToVecEmbedding.kwarg_to_string(kwargs, *args)}.pkl"

    @staticmethod
    def kwarg_to_string(kwargs: Any, *args: Any) -> str:
        parts = [str(a) for a in args]
        if isinstance(kwargs, dict):
            for key, value in sorted(kwargs.items()):
                value_str = NodeToVecEmbedding.kwarg_to_string(value)
                parts.append(f"{key}={value_str}")
        elif isinstance(kwargs, list):
            for value in kwargs:
                value_str = NodeToVecEmbedding.kwarg_to_string(value)
                parts.append(f"{value_str}")
        elif isinstance(kwargs, str):
            parts.append(kwargs)
        elif isinstance(kwargs, (int, float)):
            parts.append(str(kwargs))
        else:
            raise ValueError(f"Unknown type: {type(kwargs)}, {kwargs}")
        return "@".join(parts)

    @staticmethod
    def to_undirected(graph: nx.MultiDiGraph, use_int_names: bool = False):
        """
        Weights on the edges are converted via tfidf score of the target node:
        1. convert every node to a word (its name)
        2. convert a "u" (source) node to a document:
           - join its "v" nodes (as words) to a string
        3. compute tfidf: edge weight of the edge (u, v) is
           tfidf(u, v) where u is a document and v word in it

        :param graph:
        :return:
        """
        new_to_old = NodeToVecEmbedding.convert_node_names(graph, use_int_names)
        old_to_new = {old: new for new, old in new_to_old.items()}
        documents: Dict[Node, List[str]] = {}
        u_graph = nx.Graph()
        for node in graph.nodes:
            u_graph.add_node(old_to_new[node])
            documents[node] = []
        for u, v, e_type, w in graph.edges(keys=True, data="w", default=1):
            u_new = old_to_new[u]
            v_new = old_to_new[v]
            if not e_type.is_normal():
                continue
            # update documents
            if u not in documents:
                documents[u] = []
            for _ in range(round(w)):
                documents[u].append(str(v_new))  # for joining ...
            # update graph
            if v_new not in u_graph[u_new]:
                u_graph.add_edge(u_new, v_new, w=0.0)
            else:
                u_graph[u_new][v_new]["w"] += w
        # the loop above skips the nodes with no edges
        for u in graph.nodes:
            u_new = old_to_new[u]
            if u not in documents:
                documents[u] = ["empty document"]
            if u_new not in u_graph:
                u_graph.add_node(u_new)

        node_order = sorted(old_to_new)
        vectorizer = TfidfVectorizer(stop_words=[], token_pattern=r"(?u)\b\w+\b")
        vectorizer.fit_transform(  # type: ignore
            [" ".join(documents[node]) for node in node_order]  # type: ignore
        )  # .todense()
        words: List[str] = vectorizer.get_feature_names_out()  # type: ignore
        idf_values = np.log2(vectorizer.idf_)  # type: ignore
        inverse_document_frequencies = dict(zip(words, idf_values))
        for u, v, e_type, w in graph.edges(keys=True, data="w", default=1):  # type: ignore
            source_new = old_to_new[u]  # type: ignore
            sink_new = old_to_new[v]  # type: ignore
            if not e_type.is_normal():  # type: ignore
                continue
            weight_increase = w * inverse_document_frequencies[str(sink_new)]  # type: ignore
            u_graph[source_new][sink_new]["w"] += weight_increase
        return u_graph, new_to_old

    @staticmethod
    def convert_node_names(graph: nx.Graph, use_int_names: bool) -> Dict[str, Node]:
        # fancy stuff like re.sub("[ .',()]", "_", str(node)).lower()
        # is dangerous (since the definition of a word is rather tricky)
        dictionary: Dict[str, Node] = {}
        for node in graph.nodes:
            s_node = len(dictionary) if use_int_names else f"word{len(dictionary)}"
            if s_node in dictionary:
                raise ValueError(
                    f"{node} and {dictionary[s_node]} "
                    f"have the same representation: {s_node}"
                )
            dictionary[s_node] = node
        return dictionary


class MetapathToVec(NodeEmbeddingBase):
    def __init__(
        self,
    ):
        super().__init__("metapath2vec")
        raise NotImplementedError(
            "Apparently, noone has done this in the last 4 years ..."
        )
