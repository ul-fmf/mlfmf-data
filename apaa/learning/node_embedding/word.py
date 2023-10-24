from collections import Counter
from typing import Any, Dict, List, Tuple, Optional
import networkx as nx
import numpy as np
from apaa.data.structures.agda_tree import AgdaDefinition
from apaa.learning.node_embedding.base import Node, NodeEmbeddingBase
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from enum import Enum

from apaa.other.helpers import MyTypes


array1d = MyTypes.ARRAY_1D
array2d = MyTypes.ARRAY_2D


class WordEmbedding(NodeEmbeddingBase):
    def __init__(self, name: str):
        super().__init__(name)

    @staticmethod
    def definition_to_document(
        graph: nx.MultiDiGraph,
        definitions: Dict[Node, AgdaDefinition],
        definition: AgdaDefinition,
    ) -> str:
        return " ".join(definition.to_words(graph, definitions))

    @staticmethod
    def get_embeddings(
        graph: nx.MultiDiGraph,
        definitions: Dict[Node, AgdaDefinition],
        nodes: List[Node],
        vectorizer: CountVectorizer | TfidfVectorizer,
    ) -> array2d:
        definitions_as_words = []
        for node in nodes:
            if node in definitions:
                doc = WordEmbedding.definition_to_document(graph, definitions, definitions[node])
            else:
                doc  = "not a definition"
            definitions_as_words.append(doc)
        return vectorizer.fit_transform(definitions_as_words)  # type: ignore

    def sparse_embeddings(self):
        raise NotImplementedError()


class BagOfWordsEmbedder(WordEmbedding):
    def __init__(self):
        super().__init__("BagOfWords")

    def embed(
        self,
        graph: nx.MultiDiGraph,
        definitions: Dict[Node, AgdaDefinition],
        **kwargs: Any,
    ) -> Tuple[List[Node], array2d]:
        nodes = list(graph.nodes)
        vectorizer = CountVectorizer(**kwargs)
        embeddings = WordEmbedding.get_embeddings(graph, definitions, nodes, vectorizer)
        return nodes, embeddings

    @property
    def node_embeddings(self) -> array2d:
        assert self._node_embeddings is not None
        return self._node_embeddings.todense()

    def sparse_embeddings(self):
        assert self._node_embeddings is not None
        return self._node_embeddings


class TFIDFEmbedder(WordEmbedding):
    def __init__(self):
        super().__init__("tfidf")

    def embed(
        self,
        graph: nx.MultiDiGraph,
        definitions: Dict[Node, AgdaDefinition],
        **kwargs: Any,
    ) -> Tuple[List[Node], array2d]:
        nodes = list(graph.nodes)
        vectorizer = TfidfVectorizer(**kwargs)
        embeddings = WordEmbedding.get_embeddings(
            graph, definitions, nodes, vectorizer
        ).todense()
        if isinstance(embeddings, np.matrix):
            embeddings = np.asarray(embeddings)
        return nodes, embeddings


class WordFrequencyWeight(Enum):
    COUNT = "count"
    COUNT_LOG2 = "count log2"
    TFIDF = "tfidf"
    TFIDF_LOG2_COUNT = "tfidf log2 count"
    CONSTANT = "constant"

    def needs_tfidf(self):
        return self in [WordFrequencyWeight.TFIDF, WordFrequencyWeight.TFIDF_LOG2_COUNT]


class DeepWordEmbedder(WordEmbedding):
    def __init__(
        self,
        words: Optional[List[str]] = None,
        word_embeddings: Optional[array2d] = None,
        word_frequency_weight: WordFrequencyWeight = WordFrequencyWeight.COUNT,
    ):
        super().__init__("word2vec")
        if words is None or word_embeddings is None:
            raise ValueError("Need pretrained word embeddings!")
        elif len(words) != word_embeddings.shape[0]:
            raise ValueError("Number of words != number of embedding vectors")
        self.words = {word: i for i, word in enumerate(words)}
        self.word_embeddings = word_embeddings
        self.dim = self.word_embeddings.shape[1]
        self.word_frequency_weight = word_frequency_weight
        self.inverse_document_frequency: Optional[Dict[str, float]] = None

    def _get_default_vector(self) -> array1d:
        return np.zeros(self.dim)

    def get_word_embedding(self, original_word: str) -> array1d:
        word = original_word.lower()
        if word not in self.words:
            # ignore this word
            return self._get_default_vector()
        else:
            return self.word_embeddings[self.words[word]]

    def word_weight(self, word: str, count: float) -> float:
        if self.word_frequency_weight == WordFrequencyWeight.COUNT:
            return count
        elif self.word_frequency_weight == WordFrequencyWeight.COUNT_LOG2:
            return 1.0 + np.log2(count)
        elif self.word_frequency_weight.needs_tfidf():
            assert self.inverse_document_frequency is not None
            canonic = word.lower()
            if canonic not in self.inverse_document_frequency:
                return 0.0  # ignore it: unknown word
            else:
                w_count = count
                if self.word_frequency_weight == WordFrequencyWeight.TFIDF_LOG2_COUNT:
                    w_count = 1.0 + np.log2(count)
                return w_count * self.inverse_document_frequency[canonic]
        elif self.word_frequency_weight == WordFrequencyWeight.CONSTANT:
            return 1.0
        else:
            raise ValueError(f"Wrong WordFrequencyWeight: {self.word_frequency_weight}")

    def definition_vector(
        self,
        graph: nx.MultiDiGraph,
        definitions: dict[Node, AgdaDefinition],
        definition: AgdaDefinition,
    ):
        word_counts = Counter(definition.to_words(graph, definitions))
        vector = self._get_default_vector()
        for word, count in word_counts.items():
            vector = vector + self.word_weight(word, count) * self.get_word_embedding(
                word
            )
        return vector

    def embed(
        self,
        graph: nx.MultiDiGraph,
        definitions: Dict[Node, AgdaDefinition],
        **kwargs: Any,
    ) -> Tuple[List[Node], array2d]:
        nodes = list(graph.nodes)
        if self.word_frequency_weight.needs_tfidf():
            vectorizer = TfidfVectorizer(**kwargs)
            WordEmbedding.get_embeddings(graph, definitions, nodes, vectorizer)
            words: List[str] = vectorizer.get_feature_names_out()
            idf_values: array1d = np.log2(vectorizer.idf_)  # type: ignore
            self.inverse_document_frequency = dict(zip(words, idf_values))
        embeddings = np.zeros((len(nodes), self.dim))
        for i, example in enumerate(nodes):
            if example not in definitions:
                embeddings[i] = self._get_default_vector()
            else:
                embeddings[i] = self.definition_vector(
                    graph, definitions, definitions[example]
                )
        return nodes, embeddings
