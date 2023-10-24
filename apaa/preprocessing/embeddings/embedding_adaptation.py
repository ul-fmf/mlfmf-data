import os
import numpy as np
import sys
from apaa.other.helpers import Embeddings, Other, Locations


LOGGER = Other.create_logger(__file__)


ADDITIONAL_TRANSLATIONS = {
    "ω": "infinity"  # to be consistent with setω -> set + infinity
}


def load_translations(file: str):
    """
    Hacky format, every line either

    word;count;cumulative count;word1;word2;...;wordN

    or

    word;count;cumulative count;word1;word2;...;wordN;weight1;...;weightN

    If no words, skip this later.
    If no weights, they are uniform.

    :param file:
    :return:
    """
    dictionary = {}
    with open(file, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip(";\n").split(";")
            word = parts[0]
            # parts[1:3] = count, cumulative count
            other = parts[3:]
            if not other:
                dictionary[word] = ([], [])
                continue
            try:
                last_weight = float(other[-1])  # might be 1.0
                has_numbers = 0.0 < last_weight < 1.0
            except ValueError:
                has_numbers = False
            if has_numbers:
                assert len(other) % 2 == 0, line
                n_other = len(other) // 2
                description = other[:n_other]
                weights = list(map(float, other[n_other:]))
            else:
                description = other
                weights = [1 / len(other) for _ in description]
            dictionary[word] = (description, weights)
    # sanity checks
    for word, (replacement_words, _) in dictionary.items():
        assert " " not in word, word
        assert all(" " not in w for w in replacement_words), (word, replacement_words)
    return dictionary


def filter_embeddings(
        embedding_file: str,
        library: str, library_word_mappings_file: str,
        filtered_embeddings_file: str
):
    words_library = Embeddings.load_words(Locations.vocabulary_file(library, True), separator="\t")
    LOGGER.info(f"Loaded {len(words_library)} words from library")
    additional_words_library = load_translations(library_word_mappings_file)
    words_library += [w for words, _ in additional_words_library.values() for w in words]
    words_library = sorted(set(words_library))
    LOGGER.info(f"Extended to {len(words_library)} words from library")
    words_all, matrix_all = Embeddings.load_embedding(embedding_file)
    LOGGER.info(f"Loaded '{embedding_file}'")
    words_all_indices = {word: i for i, word in enumerate(words_all)}
    chosen_indices = []
    for word in words_library:
        canonic = word.lower()
        if canonic in words_all_indices:
            chosen_indices.append(words_all_indices[canonic])
        else:
            LOGGER.warning(f"The word {canonic} ({word}) missing.")
    with open(filtered_embeddings_file, "w", encoding="utf-8") as f:
        print(f"{len(chosen_indices)} {matrix_all.shape[1]}", file=f)
        for i in chosen_indices:
            word = words_all[i]
            vector = ' '.join(map(str, matrix_all[i]))
            print(f"{word} {vector}", file=f)
        # new words (defined in terms of previous words)
        for new_word, (existing_words, weights) in additional_words_library.items():
            new_vector = np.zeros(matrix_all.shape[1])
            for existing_word, weight in zip(existing_words, weights):
                if existing_word not in words_all_indices:
                    raise ValueError(f"{existing_word} not found (defines {new_word})")
                new_vector = new_vector + weight * matrix_all[words_all_indices[existing_word]]
            str_vector = ' '.join(map(str, new_vector))
            print(f"{new_word} {str_vector}", file=f)


if __name__ == "__main__":
    do_filtering = True
    the_embeddings = os.path.join(Locations.EMBEDDINGS_DIR, "pretrained", "crawl-300d-2M-subword.txt")
    additional_translations = os.path.join(
        Locations.EMBEDDINGS_DIR,
        "translations_for_stdlib_in_crawl-300d-2M-subword_words.csv"
    )
    if do_filtering:
        filter_embeddings(
            the_embeddings,
            Locations.NAME_STDLIB,
            additional_translations,
            os.path.join(Locations.EMBEDDINGS_DIR, "pretrained", f"{Locations.NAME_STDLIB}_crawl-300d-2M-subword2.txt")
        )


