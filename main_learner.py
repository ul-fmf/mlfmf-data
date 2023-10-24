import gc
import itertools
import os
import pickle
import time
from typing import Any, Dict, List, SupportsFloat, Tuple, Type, Optional

import networkx as nx
import tqdm

from apaa.data.structures import KnowledgeGraph
from apaa.data.structures.agda_tree import AgdaDefinition
from apaa.learning.edge_prediction import (
    BaseEdgeEmbeddingRecommender,
    Node2VecEdgeEmbeddingRecommender,
    EdgeEmbeddingScheme,
)
from apaa.learning.evaluation.quality_measure import (
    QualityMeasureClassification,
    QualityMeasureRecommender,
)
from apaa.learning.node_embedding import WordFrequencyWeight
from apaa.learning.recommendation.base import BaseRecommender
from apaa.learning.recommendation.dummy import DummyRecommender
from apaa.learning.recommendation.embedding import (
    BagOfWordsRecommender,
    TFIDFRecommender,
    WordEmbeddingRecommender,
    EmbeddingAnalogiesRecommender,
)
from apaa.other.helpers import Embeddings, Locations, MyTypes, Other, EdgeType


Config = tuple[str, dict[str, Any]]
Configs = list[Config]


LOGGER = Other.create_logger(__file__)

Node = MyTypes.NODE
Dataset = tuple[
    nx.MultiDiGraph,
    tuple[dict[Node, AgdaDefinition], dict[Node, AgdaDefinition]],
    tuple[
        List[Tuple[Node, Node, EdgeType]],
        List[Tuple[Node, Node, EdgeType]],
    ],
]


def learn_and_predict(
    library_path: str,
    dataset: Dataset,
    model_class: Type[BaseRecommender],
    model_args: Optional[Dict[str, Any]] = None,
    fit_args: Optional[Dict[str, Any]] = None,
    actual_k: int = 5,
    eval_as_recommender: bool = True,
    eval_as_classification: bool = False,
    file_appendix: str = "",
    force: bool = False,
):
    library = os.path.basename(library_path)
    temp_learning_file = Locations.temp_learning_file(
        library, model_class, file_appendix
    )
    model_dump_file = Locations.model_file(library, model_class, file_appendix)
    results_file = Locations.predictions_file(library, model_class, file_appendix)
    meta_file = Locations.experiment_meta_file(library, model_class, file_appendix)
    kg_pure_file = os.path.join(library_path, "graph.pkl")
    if force:
        LOGGER.info("Forcing to run ... Deleting previous results ...")
        for file in [model_dump_file, results_file, meta_file]:
            if os.path.exists(file):
                os.remove(file)
    if model_args is None:
        model_args = {}
    if fit_args is None:
        fit_args = {}
    if not check_learn_predict(
        results_file, model_args, kg_pure_file, temp_learning_file
    ):
        return
    LOGGER.info(f"For evaluation using k = {actual_k}")
    model_args["k"] = "all"
    kg = KnowledgeGraph.load_pure(kg_pure_file)
    LOGGER.info("Loaded graph")
    train_graph, defs, edges = dataset
    train_defs, test_defs = defs
    positive_edges, negative_edges = edges
    definitions_for_training = {**train_defs, **test_defs}
    should_continue, model, t_learn, did_learn = learn(
        model_dump_file,
        model_class,
        model_args,
        train_graph,
        definitions_for_training,
        fit_args,
        meta_file,
    )
    if not should_continue:
        LOGGER.info("Stopping the experiment ...")
        return
    assert model is not None
    assert t_learn is not None
    assert did_learn is not None
    create_experiment_meta_file(
        meta_file,
        model_class,
        model_args,
        fit_args,
        did_learn,
        t_learn,
    )
    predict_and_evaluate(
        kg,
        train_graph,
        definitions_for_training,
        test_defs,
        positive_edges,
        negative_edges,
        actual_k,
        model,
        eval_as_recommender,
        eval_as_classification,
        results_file,
    )
    if not os.path.exists(temp_learning_file):
        LOGGER.error(f"Temp learning file {temp_learning_file} was deleted.")
    else:
        os.remove(temp_learning_file)


def check_learn_predict(
    results_file: str, model_args: dict[str, Any], kg_file: str, temp_learning_file: str
):
    if os.path.exists(results_file):
        LOGGER.info(f"Results '{results_file}' exist, nothing to do.")
        return False
    if "k" in model_args:
        LOGGER.warning("Overwriting k = 'all' in model args")
    else:
        LOGGER.info("Setting k = 'all' in model args")
    if not os.path.exists(kg_file):
        raise ValueError(f"Missing kg file '{kg_file}'")
    if os.path.exists(temp_learning_file):
        LOGGER.info(f"Somebody else is doing {temp_learning_file}, skipping ...")
        return False
    else:
        with open(temp_learning_file, "w", encoding="utf-8"):
            pass
    return True


def learn(
    model_dump_file: str,
    model_class: Type[BaseRecommender],
    model_args: Dict[str, Any],
    train_graph: nx.MultiDiGraph,
    id_to_def: Dict[Node, AgdaDefinition],
    fit_args: Dict[str, Any],
    meta_file: str,
) -> tuple[bool, Optional[BaseRecommender], Optional[float], Optional[bool]]:
    will_learn = not os.path.exists(model_dump_file)
    if will_learn:
        LOGGER.info("Learning models")
        t0_learn = time.time()
        model: BaseRecommender = model_class(**model_args)
        model.fit(train_graph, id_to_def, **fit_args)
        t1_learn = time.time()
        # model.dump(model_dump_file)
        LOGGER.warning("Not dumping the model, since it is too big.")
    else:
        if os.path.exists(meta_file):
            LOGGER.error(
                f"Delete {meta_file} first to prevent any confusion about meta data."
            )
            return False, None, None, None
        LOGGER.info("Loading models")
        t0_learn = time.time()
        model = model_class.load(model_dump_file)
        t1_learn = time.time()
    t_learn = t1_learn - t0_learn
    return True, model, t_learn, will_learn


def predict_and_evaluate(
    kg: nx.MultiDiGraph,
    train_graph: nx.MultiDiGraph,
    defs_for_training: dict[Node, AgdaDefinition],
    test_definitions: dict[Node, AgdaDefinition],
    positive_edges: List[Tuple[Node, Node, EdgeType]],
    negative_edges: List[Tuple[Node, Node, EdgeType]],
    actual_k: int,
    model: BaseRecommender,
    compute_recommender_style: bool,
    compute_link_prediction_style: bool,
    results_file: str,
):
    LOGGER.info("Predict and evaluate")
    is_edge_embedding = isinstance(model, BaseEdgeEmbeddingRecommender)
    predictions_recommender: Dict[Node, List[Tuple[float, Node]]] = {}
    actual_neighbours_recommender: Dict[Node, List[Tuple[float, Node]]] = {}
    measures_recommender = QualityMeasureRecommender(
        kg, train_graph, test_definitions, actual_k
    )

    results_classification: Dict[
        Tuple[Node, Node], Tuple[SupportsFloat, SupportsFloat]
    ] = {}
    measures_classification = QualityMeasureClassification()
    if compute_recommender_style:
        evaluate_recommender_style(
            test_definitions,
            model,
            measures_recommender,
            actual_neighbours_recommender,
            predictions_recommender,
            actual_k,
        )
    if compute_link_prediction_style:
        evaluate_classification_style(
            defs_for_training,   # ok: pruned test definitons and no positive edges
            positive_edges,
            negative_edges,
            model,
            measures_classification,
            results_classification,
            predictions_recommender,
            compute_recommender_style and not is_edge_embedding,
        )
    report_results(
        results_file,
        predictions_recommender,
        actual_neighbours_recommender,
        measures_recommender,
        results_classification,
        measures_classification,
    )


def evaluate_recommender_style(
    test_defs: dict[Node, AgdaDefinition],
    model: BaseRecommender,
    measures_recommender: QualityMeasureRecommender,
    actual_neighbours_recommender: dict[Node, list[tuple[float, Node]]],
    predictions_recommender: dict[Node, list[tuple[float, Node]]],
    actual_k: int,
):
    for name, definition in tqdm.tqdm(test_defs.items()):
        neighbours: list[tuple[float, Node]] = model.predict(definition)
        _, true_neighbours = measures_recommender.update(name, neighbours)
        actual_neighbours_recommender[name] = true_neighbours
        predictions_recommender[name] = neighbours[:actual_k]


def evaluate_classification_style(
    defs_for_training: dict[Node, AgdaDefinition],
    positive_edges: List[Tuple[Node, Node, EdgeType]],
    negative_edges: List[Tuple[Node, Node, EdgeType]],
    model: BaseRecommender,
    measures_classification: QualityMeasureClassification,
    results_classification: dict[
        tuple[Node, Node], tuple[SupportsFloat, SupportsFloat]
    ],
    predictions_recommender: dict[Node, list[tuple[float, Node]]],
    use_recommender_candidates: bool,
):
    for edges, true_value in zip([positive_edges, negative_edges], [1, 0]):
        for source, sink, _ in tqdm.tqdm(edges):
            if not AgdaDefinition.is_normal_definition(sink):
                LOGGER.warning(
                    f"Skipping source-sink, since sink ({sink}) is not normal."
                )
                continue
            source_def = defs_for_training[source]
            sink_def = defs_for_training[sink]
            if use_recommender_candidates:
                candidates = predictions_recommender[source_def.name]
            else:
                candidates = None
            prediction = model.predict_one_edge(
                source_def, sink_def, nearest_neighbours=candidates
            )
            results_classification[(source, sink)] = (true_value, prediction)
    for true_value, prediction in results_classification.values():
        measures_classification.update(true_value, prediction)


def report_results(
    results_file: str,
    predictions_recommender: Dict[Node, List[Tuple[float, Node]]],
    actual_neighbours_recommender: Dict[Node, List[Tuple[float, Node]]],
    measures_recommender: QualityMeasureRecommender,
    results_classification: Dict[
        Tuple[Node, Node], Tuple[SupportsFloat, SupportsFloat]
    ],
    measures_classification: QualityMeasureClassification,
):
    with open(results_file, "w", encoding="utf-8") as f:
        print(str(measures_recommender), file=f)
        print(str(measures_classification), file=f)
        print("", file=f)
        print("RECOMMENDER EVALUATION:", file=f)
        for name in actual_neighbours_recommender:
            print("DEF NAME:", name, file=f)
            print("PREDICTED NEIGHBOURS:", file=f)
            for d, neighbour in predictions_recommender[name]:
                print(f"NEIGHBOUR;{d};{neighbour}", file=f)
            print("TRUE NEIGHBOURS [top 20, ordered by weight]:", file=f)
            for w, neighbour in actual_neighbours_recommender[name]:
                print(f"ACTUAL;{w};{neighbour}", file=f)
            print("", file=f)
        print("CLASSIFICATION EVALUATION:", file=f)
        for (source, sink), (true_value, prediction) in results_classification.items():
            print(f"EDGE;{source};{sink};{true_value};{prediction}", file=f)


def create_experiment_meta_file(
    meta_file: str,
    model_class: Type[BaseRecommender],
    model_args: dict[str, Any],
    fit_args: dict[str, Any],
    model_was_learned: bool,
    t_learn: float,
):
    with open(meta_file, "w", encoding="utf-8") as f:
        print("Meta data for the experiment", file=f)
        print(f"model class: {model_class.__name__}", file=f)
        print(f"model args: {model_args}", file=f)
        print(f"fit args: {fit_args}", file=f)
        verb = "Learning" if model_was_learned else "Loading"
        print(f"{verb} time: {t_learn} seconds", file=f)


def learn_recommender_models(
    library_path: str,
    dummy: bool = True,
    bow: bool = True,
    tfidf: bool = True,
    word_embedding: bool = True,
    analogies: bool = True,
    node_to_vec: bool = True,
    p_def_to_keep: float = 0.0,
    force: bool = False,
):
    LOGGER.info(f"Learning for {library_path}")
    dataset_file = os.path.join(library_path, "dataset.pkl")
    with open(dataset_file, "rb") as f:
        dataset: Dataset = pickle.load(f)
    if dummy:
        LOGGER.info("Dummy models ...")
        dummy_configs = create_no_arg_configs()
        learn_one_group(
            DummyRecommender, library_path, dataset, p_def_to_keep, dummy_configs, force
        )
    if bow:
        LOGGER.info("BOW models ...")
        bow_configs = create_bow_configs()
        learn_one_group(
            BagOfWordsRecommender, library_path, dataset, p_def_to_keep, bow_configs, force
        )
    if tfidf:
        LOGGER.info("TFIDF models ...")
        tfidf_configs = create_tfidf_configs()
        learn_one_group(
            TFIDFRecommender, library_path, dataset, p_def_to_keep, tfidf_configs, force
        )
    if word_embedding:
        LOGGER.info("Word embedding models ...")
        we_configs = create_word_embedding_configs()
        learn_one_group(
            WordEmbeddingRecommender,
            library_path,
            dataset,
            p_def_to_keep,
            we_configs,
            force,
        )
    if analogies:
        LOGGER.info("Analogies models ...")
        ana_configs = create_analogy_configs()
        learn_one_group(
            EmbeddingAnalogiesRecommender,
            library_path,
            dataset,
            p_def_to_keep,
            ana_configs,
            force,
        )
    if node_to_vec:
        LOGGER.info("Node to vec ...")
        node_to_vec_configs = create_node_to_vec_configs()
        learn_one_group(
            Node2VecEdgeEmbeddingRecommender,
            library_path,
            dataset,
            p_def_to_keep,
            node_to_vec_configs,
            force,
        )
    LOGGER.info("\n\n")


def create_no_arg_configs() -> Configs:
    return [("empty", {})]


def create_node_to_vec_configs() -> Configs:
    configs: Configs = []
    edge_schemess = [EdgeEmbeddingScheme.CONCATENATION, EdgeEmbeddingScheme.MEAN]
    ps = [1.0, 2.0]
    qs = [1.0, 2.0]
    vector_sizes = [32, 64]
    windows = [2, 4, 5]
    walk_lengths = [50, 100]
    epochss = [10]
    classifiers = {
        "rf": [{"max_features": 1.0, "n_jobs": -1}],
        # "knn": [{"metric": "cosine"}, {"metric": "cityblock"}],
    }
    for i, combination in enumerate(
        itertools.product(
            edge_schemess, ps, qs, vector_sizes, windows, walk_lengths, epochss
        )
    ):
        if i != 4:
            # only the best so far ...
            continue
        edge_scheme, p, q, vector_size, window, walk_length, epochs = combination
        basic = {
            "edge_embedding_scheme": edge_scheme,
            "p": p,
            "q": q,
            "vector_size": vector_size,
            "window": window,
            "walk_length": walk_length,
            "epochs": epochs,
        }
        for model, options in classifiers.items():
            for classifier_kwargs in options:
                extended = {
                    **basic,
                    "classifier": model,
                    "classifier_kwargs": classifier_kwargs,
                }
                metric = (
                    "" if model != "knn" else "_metric_" + classifier_kwargs["metric"]
                )
                configs.append(
                    (
                        f"edge_{edge_scheme.value}_p{p}_q{q}_vec_size{vector_size}_"
                        f"window{window}_walk_len{walk_length}_epo{epochs}"
                        f"model_{model}{metric}",
                        extended,
                    )
                )
    return configs


def create_word_embedding_configs():
    options: Configs = []
    words, word_embeddings = Embeddings.load_embedding(
        os.path.join(
            Locations.EMBEDDINGS_DIR, "pretrained", "stdlib_crawl-300d-2M-subword2.txt"
        )
    )
    for frequency_weight, metric in itertools.product(
        WordFrequencyWeight, ["cosine", "cityblock"]
    ):
        name = f"{frequency_weight}-{metric}"
        args = {
            "words": words,
            "word_embeddings": word_embeddings,
            "word_frequency_weight": frequency_weight,
            "metric": metric,
        }
        options.append((name, args))
    return options[7:8] + options[-1:]  # only the best: 7 for word2vec, 9 for analogies


def create_analogy_configs():
    # needs the same as word embedding
    return create_word_embedding_configs()


def create_tfidf_configs() -> Configs:
    options: Configs = []
    for metric in ["cosine", "cityblock"]:
        name = f"{metric}"
        args = {"metric": metric}
        options.append((name, args))
    return options


def create_bow_configs():
    return create_no_arg_configs()


def create_tfidf_word_embedding_configs():
    # tfidf needs no additional ones
    return create_word_embedding_configs()


def learn_one_group(
    model_type: Type[BaseRecommender],
    library_path: str,
    dataset: Dataset,
    p_def_to_keep: float,
    configs: Configs,
    force: bool,
):
    for i_config, (name, config) in enumerate(configs):
        LOGGER.info(f"Processing config {name}  #{i_config + 1}/{len(configs)}")
        learn_and_predict(
            library_path,
            dataset,
            model_type,
            model_args=config,
            actual_k=5,
            eval_as_recommender=True,
            eval_as_classification=True,
            force=force,
            file_appendix=f"_{i_config}_p_to_keep_{p_def_to_keep}_{name}",
        )
        gc.collect()


if __name__ == "__main__":
    learn_recommender_models(
        "your/path/to/mathlib",  # change this
        dummy=True,
        bow=True,
        tfidf=True,
        word_embedding=True,
        analogies=True,
        node_to_vec=True,
        p_def_to_keep=0.1,
        force=False,
    )
    LOGGER.info("Done")
