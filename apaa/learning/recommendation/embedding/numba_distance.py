from typing import Dict

import numpy as np
import tqdm
from numba import jit


@jit(nopython=True)
def merge(words1: np.ndarray, words2):
    n1 = words1.size
    n2 = words2.size
    union = np.zeros(n1 + n2, dtype=np.int64)
    position = 0
    left = 0
    right = 0
    while left < n1 and right < n2:
        w_left = words1[left]
        w_right = words2[right]
        if w_left == w_right:
            union[position] = w_left
            left += 1
            right += 1
        elif w_left < w_right:
            union[position] = w_left
            left += 1
        else:
            union[position] = w_right
            right += 1
        position += 1
    while left < n1:
        union[position] = words1[left]
        position += 1
        left += 1
    while right < n2:
        union[position] = words2[right]
        position += 1
        right += 1
    return union[:position]


@jit(nopython=True)
def find_with_bisection(words: np.ndarray, word: np.int64):
    left = 0
    right = words.size
    while right - left > 0:
        middle_position = (left + right) // 2
        middle = words[middle_position]
        if middle == word:
            return middle_position
        elif middle < word:
            left = middle_position + 1
        else:
            right = middle_position
    return -1


@jit(nopython=True)
def jaccard(words1, counts1, words2, counts2) -> float:
    """
    :param words1: sorted
    :param counts1:
    :param words2:
    :param counts2: sorted
    :return:
    """
    union_words = merge(words1, words2)
    n = union_words.size
    union_size = 0.0
    intersection_size = 0.0
    for i in range(n):
        position1 = find_with_bisection(words1, union_words[i])
        position2 = find_with_bisection(words2, union_words[i])
        if position1 >= 0:
            c1 = counts1[position1]
        else:
            c1 = 0.0
        if position2 >= 0:
            c2 = counts2[position2]
        else:
            c2 = 0.0
        if c1 >= c2:
            union_size += c1
            intersection_size += c2
        else:
            union_size += c2
            intersection_size += c1
    if union_size == 0.0:
        union_size = 1.0
    return 1.0 - intersection_size / union_size


def test_find_with_bisection():
    for n in tqdm.trange(1, 12):
        words = np.arange(0, n, dtype=np.int64)
        for w in range(-3, n + 3):
            expected = w if w in words else -1
            actual = find_with_bisection(words, np.int64(w))
            if expected != actual:
                raise ValueError(f"Does not work: {words}, {w}, {expected} != {actual}")


def test_jaccard():
    cases = [
        ([], [], [], [], 1.0),
        ([0, 1, 2], [3, 4, 5], [1, 1, 1], [1, 1, 1], 1.0),
        ([0, 1, 2], [0, 1, 2], [1, 1, 1], [1, 1, 1], 0.0),
        ([0, 1, 2], [1, 2, 3], [0, 3, 4], [3, 4, 0], 0.0),
        ([0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 2], 1 - 7 / 11),
        ([1, 2], [1, 2], [2, 3], [3, 2], 1 - 4 / 6),
        ([0, 1, 2], [], [1, 1, 1], [], 1.0)
    ]
    for words1, words2, counts1, counts2, d_exptected in tqdm.tqdm(cases):
        actual = jaccard(
            np.array(words1, dtype=np.int64),
            np.array(counts1, dtype=np.int64),
            np.array(words2, dtype=np.int64),
            np.array(counts2, dtype=np.int64),
        )
        if abs(actual - d_exptected) > 10**-5:
            raise ValueError(f"Does not work: {words1}, {counts1}, {words2}, {counts2}:"
                             f"{d_exptected} != {actual}")


#################################################################

def manhattan():
    # never used
    pass


if __name__ == "__main__":
    test_find_with_bisection()
    test_jaccard()


class Jaccard:
    @staticmethod
    def distance(fact1_words: Dict[str, float], fact2_words: Dict[str, float]):
        union = fact1_words
        for word, value in fact2_words.items():
            if word not in union:
                union[word] = value
            else:
                union[word] += value
        union_size = sum(union.values())
        intersection_size = 0
        for key in union:
            intersection_size += min(fact1_words.get(key, 0.0), fact2_words.get(key, 0.0))
        return 1 - intersection_size / union_size

    @staticmethod
    def distance_numba(
            words1: np.ndarray, counts1: np.ndarray,
            words2: np.ndarray, counts2: np.ndarray
    ):
        return jaccard(words1, counts1, words2, counts2)

