from typing import Literal, Any
from apaa.learning.node_embedding.base import EmbeddingConcatenator
from apaa.learning.node_embedding.word import (
    TFIDFEmbedder,
    DeepWordEmbedder,
    WordFrequencyWeight,
)
from apaa.learning.recommendation.embedding.base import KNNNodeEmbeddingRecommender
from apaa.other.helpers import MyTypes


array2d = MyTypes.ARRAY_2D


class TFIDFAndWord2VecEmbeddingRecommender(KNNNodeEmbeddingRecommender):
    def __init__(
        self,
        k: int | Literal["all"] = 5,
        metric: str = "cityblock",
        words: list[str] | None = None,
        word_embeddings: array2d | None = None,
        word_frequency_weight: WordFrequencyWeight = WordFrequencyWeight.TFIDF,
        **metric_kwargs: Any,
    ):
        super().__init__(
            vectorizer=EmbeddingConcatenator(
                [
                    TFIDFEmbedder(),
                    DeepWordEmbedder(
                        words=words,
                        word_embeddings=word_embeddings,
                        word_frequency_weight=word_frequency_weight,
                    ),
                ]
            ),
            k=k,
            metric=metric,
            **metric_kwargs
        )
