from apaa.data.structures import AgdaDefinitionForest, AgdaFact
from apaa.other.helpers import Other, Locations, Embeddings, TextManipulation

from collections import Counter
import tqdm
# import fasttext as ft
# import fasttext.util as ftu
import os
import traceback
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import re
from typing import Set
import matplotlib.pyplot as plt
import random
import bisect


LOGGER = Other.create_logger(__file__)


def find_vocabulary(library: str):
    LOGGER.info(f"Loading {library}")
    out_file_names = Locations.vocabulary_file(library, False)
    out_file_all = Locations.vocabulary_file(library, True)
    if os.path.exists(out_file_all) and os.path.exists(out_file_names):
        LOGGER.info(f"Already done: find vocabulary for {library}")
        return
    lib_entries: AgdaDefinitionForest = AgdaDefinitionForest.load(Locations.definitions_pickled(library))
    counts_names = Counter()
    counts_all = Counter()
    forbidden = {""}
    for entry in tqdm.tqdm(lib_entries):
        forbidden1, forbidden2, name = entry.name
        counts_names.update(TextManipulation.name_to_parts(name))
        counts_all.update(AgdaFact("irrelevant", tree=entry).words)
        forbidden.add(forbidden1)
        forbidden.add(forbidden2)
    LOGGER.info(f"Library {library} contains {len(counts_names)} words in definition names.")
    LOGGER.info(f"Library {library} contains {len(counts_all)} words in total.")
    save_to_vocabulary_file(counts_names, out_file_names, forbidden)
    save_to_vocabulary_file(counts_all, out_file_all, forbidden)


def save_to_vocabulary_file(counter: Counter, out_file: str, forbidden: Set[str]):
    with open(out_file, "w", encoding="utf-8") as f:
        print("WORD\tCOUNT", file=f)
        for word, count in sorted(counter.items(), key=lambda pair: -pair[1]):
            if word in forbidden:
                continue
            print(f"{word}\t{count}", file=f)


def prepare_embeddings_in_txt(bin_file: str, word_file: str, txt_file: str):
    # try:
    #     model = ft.load_model(bin_file)
    #     if not os.path.exists(word_file):
    #         with open(word_file, "w", encoding="utf-8") as f:
    #             for word in model.words:
    #                 print(word, file=f)
    #     if not os.path.exists(txt_file):
    #         with open(txt_file, "w", encoding="utf-8") as f:
    #             for word in tqdm.tqdm(model.words):
    #                 vec = model.get_word_vector(word)
    #                 vec_str = '\t'.join([str(x) for x in vec])
    #                 print(f"{word}\t{vec_str}", file=f)
    # except ValueError:
    #     print(f"fasttext: could not load '{bin_file}', here is the full traceback:")
    #     traceback.print_exc()
    try:
        if not os.path.exists(txt_file):
            model = KeyedVectors.load_word2vec_format(bin_file, binary=True)
            model.save_word2vec_format(txt_file, binary=False)
        if not os.path.exists(word_file):
            embedding_vocabulary(txt_file, word_file)
    except ValueError:
        LOGGER.error(f"Could not load '{bin_file}' with gensim.")
        traceback.print_exc()


def embedding_vocabulary(text_file: str, word_file: str):
    LOGGER.info(f"Creating word file from {text_file}")
    n_errors = 0
    with open(word_file, "w", encoding="utf-8") as f_out:
        with open(text_file, encoding="utf-8") as f_in:
            for line in tqdm.tqdm(f_in):
                match = re.match("([^ \t]+)[ \t]-?\\d+\\.\\d+", line)
                if match is None:
                    n_errors += 1
                    LOGGER.warning(f"Could not extract a word from the line: '{line}'")
                    if n_errors > 10:
                        LOGGER.warning(f"Existing, n_errors = {n_errors}")
                        return
                    continue
                word = match.group(1)
                print(word, file=f_out)


def prepare_all_embeddings():
    vec_location = os.path.join(Locations.EMBEDDINGS_DIR, "pretrained")
    bin_files = [os.path.join(vec_location, f) for f in os.listdir(vec_location) if f.endswith(".bin")]
    for file in bin_files:
        LOGGER.info(f"Processing '{file}'")
        bare_name = file[:file.rfind(".")]
        word_file = bare_name + "_words.txt"
        text_file = bare_name + ".txt"
        prepare_embeddings_in_txt(file, word_file, text_file)


def n_missing():
    LOGGER.info("Computing missing words")
    libs = [Locations.NAME_STDLIB, Locations.NAME_UNIMATH, Locations.NAME_AGDA_TEST]
    vec_location = os.path.join(Locations.EMBEDDINGS_DIR, "pretrained")
    vec_word_files = [os.path.join(vec_location, f) for f in os.listdir(vec_location) if "_words.txt" in f]
    scores = []
    for vec_word_file in vec_word_files:
        vec_words = load_vec_words(vec_word_file)
        vec_word_file_name = os.path.basename(vec_word_file)
        vec_word_file_name = vec_word_file_name[:vec_word_file_name.rfind(".")]
        for lib in libs:
            lib_words = load_lib_words(Locations.vocabulary_file(lib, True))
            missing_words = score(lib_words, vec_words)
            penalty = (len(missing_words), sum(missing_words.values()))
            scores.append((lib, penalty, os.path.basename(vec_word_file)))
            report_file = os.path.join(Locations.EMBEDDINGS_DIR, f"missing_for_{lib}_in_{vec_word_file_name}.txt")
            with open(report_file, "w", encoding="utf-8") as f:
                for word, count in sorted(missing_words.items(), key=lambda t: -t[1]):
                    print(f"{word}\t{count}", file=f)
    scores.sort()
    for s in scores:
        LOGGER.info(s)


def load_words(file: str, is_library):
    words = {}
    with open(file, encoding="utf-8") as f:
        if is_library:
            f.readline()  # header
        for line in f:
            parts = line.rstrip().split("\t")
            if is_library:
                assert len(parts) == 2, (parts, file)
                word = parts[0]
                count = int(parts[1])
            else:
                assert len(parts) == 1, parts
                word = parts[0]
                count = 1
            words[word] = count
    return words


def load_vec_words(file: str):
    return load_words(file, False)


def load_lib_words(file: str):
    return load_words(file, True)


def score(words_lib, words_vec):
    missing = {}
    for word, count in tqdm.tqdm(words_lib.items()):
        if word not in words_vec:
            missing[word] = count
    return missing


def embedding_distribution(file: str):
    _, matrix = Embeddings.load_embedding(file)
    # norms
    euklid_norm = np.sum(np.square(matrix), axis=1)
    plt.hist(euklid_norm, bins=20)
    plt.title("Euklid norm of vectors")
    plt.show()
    # max / min
    max_components = np.max(matrix, axis=0)
    min_components = np.min(matrix, axis=0)
    mean_components = np.mean(matrix, axis=0)
    std_components = np.std(matrix, axis=0)
    for what, name in zip(
            [max_components, min_components, mean_components, std_components],
            ["max", "min", "mean", "std"]
    ):
        plt.hist(what, bins=20)
        plt.title(f"{name} of components")
        plt.show()


def embedding_quality(file: str):
    def eu_norm(vec):
        return np.sqrt(np.sum(np.square(vec)))

    def distance(vec1, vec2):
        if DIST_VERSION == 0:
            return np.mean(np.abs(vec1 - vec2))
        elif DIST_VERSION == 1:
            return 1.0 - vec1 @ vec2 / (eu_norm(vec1) * eu_norm(vec2))
        else:
            raise ValueError(":)")

    def distance_multi(matrix, vec):
        if DIST_VERSION == 0:
            return np.mean(np.abs(matrix - vec), axis=1)
        elif DIST_VERSION == 1:
            dot_products = matrix @ vec
            norms = np.sqrt(np.sum(np.square(matrix), axis=1))
            return 1.0 - dot_products / norms / eu_norm(vec)
        else:
            raise ValueError(":)")

    def find_embedding(w: str):
        i_word_group = groups[w.lower()]
        for i_word in i_word_group:
            if words[i_word] == w:
                return i_word, embeddings[i_word]
        raise ValueError(f"Cannot find '{w}'")

    def compute_quantile(values, element):
        return bisect.bisect_left(values, element) / len(values)

    DIST_VERSION = 1

    # similarity between random words vs. similarity of lower : upper-case
    words, embeddings = Embeddings.load_embedding(file, normalize=True)
    n_words, n_dim = embeddings.shape

    groups = {}  # lowercase: all cases
    for i, word in enumerate(words):
        canonic = word.lower()
        if canonic not in groups:
            groups[canonic] = [i]
        else:
            groups[canonic].append(i)
    chosen = [(word, variations) for word, variations in groups.items() if len(variations) > 1]
    chosen.sort(key=lambda pair: -len(pair[1]))
    chosen_distances = []
    n_top = min(100, len(chosen))
    for i in range(n_top):
        canonic, variations = chosen[i]
        x0 = embeddings[variations[0]]
        distances = []
        for i_other in variations[1:]:
            x1 = embeddings[i_other]
            distances.append(distance(x0, x1))  # man hat tan / n dim
        chosen_distances.append(distances)
    n_random = 10**3
    other_distances = []
    for _ in range(n_random):
        i0 = int(random.random() * n_words)
        i1 = int(random.random() * n_words)
        other_distances.append(distance(embeddings[i0], embeddings[i1]))
    other_distances.sort()
    for i in range(n_top):
        word, variations = chosen[i]
        distances = chosen_distances[i]
        quantiles = sorted([(compute_quantile(other_distances, d), words[w]) for d, w in zip(distances, variations)])
        LOGGER.info(f"Quantiles for {word}: {quantiles}")

    # some analogies:
    # 1)
    # -  man - woman
    # -  king - queen
    # -  waiter - waitress
    # 2)
    # -  Paris - France
    # -  London - England
    # 3) Paris - paris
    #    Home - home
    #    Boy - boy
    pair_groups = [
        [("man", "woman"), ("king", "queen"), ("waiter", "waitress")],
        [("Paris", "France"), ("London", "England"), ("Tokio", "Japan")],
        [("Paris", "paris"), ("Home", "home"), ("Boy", "boy")]
    ]
    # man - woman + king = queen?
    for group in pair_groups:
        LOGGER.info(f"Analyzing {group}")
        x0, y0 = group[0]
        diff = find_embedding(x0)[1] - find_embedding(y0)[1]
        for x1, y1 in group[1:]:
            target = diff + find_embedding(y1)[1]
            i_actual, actual = find_embedding(x1)
            distances_to_actual = list(enumerate(distance_multi(embeddings, target)))
            distances_to_actual.sort(key=lambda pair: pair[1])
            nns = [(words[i], f"{d:.2e}") for i, d in distances_to_actual[:10]]
            distances_only = [d for _, d in distances_to_actual]
            distances_only.sort()
            d_actual = distance(target, actual)
            quantile = compute_quantile(distances_only, d_actual)
            LOGGER.info(f"    {x0} - {y0} + {y1} = {nns} [quantile/dist: {quantile}, {d_actual:.2e}]")

    # some synonyms
    pairs = [
        (["happy"], ["thrilled"]),
        (["omega"], ["ω"]),
        (["lambda"], ["λ"]),
        (["omega"], ["lambda"]),
        (["ω"], ["λ"]),
        (["l"], ["ell"]),
        (["co", "prime"], ["coprime"]),
        (["un", "pleasant"], ["unpleasant"])
    ]
    for xs0, xs1 in pairs:
        v0 = np.zeros(n_dim)
        v1 = np.zeros(n_dim)
        for x0 in xs0:
            v0 = find_embedding(x0)[1] + v0
        for x1 in xs1:
            v1 = find_embedding(x1)[1] + v1
        d = distance(v0, v1)
        q = compute_quantile(other_distances, d)
        LOGGER.info(f"{xs0} and {xs1}: d = {d:.2e}, q = {q:.2e}")


if __name__ == "__main__":
    do_lib_vocab = False
    do_vec_vocab = False
    do_missing = False
    do_analisis = False
    do_quality = False
    if do_lib_vocab:
        for lib_name in [Locations.NAME_STDLIB, Locations.NAME_UNIMATH, Locations.NAME_AGDA_TEST]:
            find_vocabulary(lib_name)
    if do_vec_vocab:
        prepare_all_embeddings()
    if do_missing:
        n_missing()

    the_embeddings = os.path.join(Locations.EMBEDDINGS_DIR, "pretrained", "crawl-300d-2M-subword.txt")
    if do_analisis:
        embedding_distribution(the_embeddings)
    if do_quality:
        embedding_quality(the_embeddings)
