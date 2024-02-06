import logging
from mteb import MTEB
import json
import os

import time

from utils.constants import BATCH_SIZE

# Basically pulled from mteb, but turned into a function to make it easier to use for my needs
def mteb_meta(results_folder, output_folder):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


    # results_folder = sys.argv[1].rstrip("/")
    model_name = results_folder.split("/")[-1]

    all_results = {}

    for file_name in os.listdir(results_folder):
        if not file_name.endswith(".json"):
            logger.info(f"Skipping non-json {file_name}")
            continue
        with open(os.path.join(results_folder, file_name), "r", encoding="utf-8") as f:
            results = json.load(f)
            all_results = {**all_results, **{file_name.replace(".json", ""): results}}

    # Use "train" split instead
    TRAIN_SPLIT = ["DanishPoliticalCommentsClassification"]
    # Use "validation" split instead
    VALIDATION_SPLIT = ["AFQMC", "Cmnli", "IFlyTek", "TNews", "MSMARCO", "MultilingualSentiment", "Ocnli"]
    # Use "dev" split instead
    DEV_SPLIT = ["CmedqaRetrieval", "CovidRetrieval", "DuRetrieval", "EcomRetrieval", "MedicalRetrieval", "MMarcoReranking", "MMarcoRetrieval", "MSMARCO", "T2Reranking", "T2Retrieval", "VideoRetrieval"]

    MARKER = "---"
    TAGS = "tags:"
    MTEB_TAG = "- mteb"
    HEADER = "model-index:"
    MODEL = f"- name: {model_name}"
    RES = "  results:"

    META_STRING = "\n".join([MARKER, TAGS, MTEB_TAG, HEADER, MODEL, RES])


    ONE_TASK = "  - task:\n      type: {}\n    dataset:\n      type: {}\n      name: {}\n      config: {}\n      split: {}\n      revision: {}\n    metrics:"
    ONE_METRIC = "    - type: {}\n      value: {}"
    SKIP_KEYS = ["std", "evaluation_time", "main_score", "threshold"]

    for ds_name, res_dict in sorted(all_results.items()):
        mteb_desc = (
            MTEB(tasks=[ds_name.replace("CQADupstackRetrieval", "CQADupstackAndroidRetrieval")]).tasks[0].description
        )
        hf_hub_name = mteb_desc.get("hf_hub_name", mteb_desc.get("beir_name"))
        if "CQADupstack" in ds_name:
            hf_hub_name = "BeIR/cqadupstack"
        mteb_type = mteb_desc["type"]
        revision = res_dict.get("dataset_revision")  # Okay if it's None
        split = "test"
        if (ds_name in TRAIN_SPLIT) and ("train" in res_dict):
            split = "train"
        elif (ds_name in VALIDATION_SPLIT) and ("validation" in res_dict):
            split = "validation"
        elif (ds_name in DEV_SPLIT) and ("dev" in res_dict):
            split = "dev"
        elif "test" not in res_dict:
            logger.info(f"Skipping {ds_name} as split {split} not present.")
            continue
        res_dict = res_dict.get(split)
        for lang in mteb_desc["eval_langs"]:
            mteb_name = f"MTEB {ds_name}"
            mteb_name += f" ({lang})" if len(mteb_desc["eval_langs"]) > 1 else ""
            # For English there is no language key if it's the only language
            test_result_lang = res_dict.get(lang) if len(mteb_desc["eval_langs"]) > 1 else res_dict
            # Skip if the language was not found but it has other languages
            if test_result_lang is None:
                continue
            META_STRING += "\n" + ONE_TASK.format(
                mteb_type, hf_hub_name, mteb_name, lang if len(mteb_desc["eval_langs"]) > 1 else "default", split, revision
            )
            for metric, score in test_result_lang.items():
                if not isinstance(score, dict):
                    score = {metric: score}
                for sub_metric, sub_score in score.items():
                    if any([x in sub_metric for x in SKIP_KEYS]):
                        continue
                    META_STRING += "\n" + ONE_METRIC.format(
                        f"{metric}_{sub_metric}" if metric != sub_metric else metric,
                        # All MTEB scores are 0-1, multiply them by 100 for 3 reasons:
                        # 1) It's easier to visually digest (You need two chars less: "0.1" -> "1")
                        # 2) Others may multiply them by 100, when building on MTEB making it confusing what the range is
                        # This happend with Text and Code Embeddings paper (OpenAI) vs original BEIR paper
                        # 3) It's accepted practice (SuperGLUE, GLUE are 0-100)
                        sub_score * 100,
                    )

    output_filename = output_folder.rstrip("/") + "/" + "mteb_metadata.md"
    META_STRING += "\n" + MARKER
    if os.path.exists(output_filename):
        logger.warning("Overwriting mteb_metadata.md")
    with open(output_filename, "w") as f:
        f.write(META_STRING)


TASK_LIST_CLASSIFICATION = [
    # "AmazonCounterfactualClassification",
    # "AmazonPolarityClassification",
    # "AmazonReviewsClassification",
    # "Banking77Classification",
    # "EmotionClassification",
    # "ImdbClassification",
    # "MassiveIntentClassification",
    # "MassiveScenarioClassification",
    # "MTOPDomainClassification",
    # "MTOPIntentClassification",
    # "ToxicConversationsClassification",
    # "TweetSentimentExtractionClassification",
]

TASK_LIST_CLUSTERING = [
    # "ArxivClusteringP2P",
    # "ArxivClusteringS2S",
    # "BiorxivClusteringP2P",
    # "BiorxivClusteringS2S",
    # "MedrxivClusteringP2P",
    # "MedrxivClusteringS2S",
    # "RedditClustering",
    # "RedditClusteringP2P",
    # "StackExchangeClustering",
    # "StackExchangeClusteringP2P",
    # "TwentyNewsgroupsClustering",
]

TASK_LIST_PAIR_CLASSIFICATION = [
    # "SprintDuplicateQuestions",
    # "TwitterSemEval2015",
    # "TwitterURLCorpus",
]

TASK_LIST_RERANKING = [
    # "AskUbuntuDupQuestions",
    # "MindSmallReranking",
    # "SciDocsRR",
    # "StackOverflowDupQuestions",
]

TASK_LIST_RETRIEVAL = [
    # "ArguAna",
    # "ClimateFEVER",
    # "CQADupstackAndroidRetrieval",
    # "CQADupstackEnglishRetrieval",
    # "CQADupstackGamingRetrieval",
    # "CQADupstackGisRetrieval",
    # "CQADupstackMathematicaRetrieval",
    # "CQADupstackPhysicsRetrieval",
    # "CQADupstackProgrammersRetrieval",
    # "CQADupstackStatsRetrieval",
    # "CQADupstackTexRetrieval",
    # "CQADupstackUnixRetrieval",
    # "CQADupstackWebmastersRetrieval",
    # "CQADupstackWordpressRetrieval",
    # "DBPedia",
    # "FEVER",
    # "FiQA2018",
    # "HotpotQA",
    # "MSMARCO",
    # "NFCorpus",
    # "NQ",
    # "QuoraRetrieval",
    "SCIDOCS",
    # "SciFact",
    # "Touche2020",
    # "TRECCOVID",
]

TASK_LIST_STS = [
    # "BIOSSES",
    # "SICK-R",
    # "STS12",
    # "STS13",
    # "STS14",
    # "STS15",
    # "STS16",
    # "STS17",
    # "STS22",
    # "STSBenchmark",
    # "SummEval",
]


TASK_LIST = (
    TASK_LIST_CLASSIFICATION
    + TASK_LIST_CLUSTERING
    + TASK_LIST_PAIR_CLASSIFICATION
    + TASK_LIST_RERANKING
    + TASK_LIST_RETRIEVAL
    + TASK_LIST_STS
)

def test_all_mteb(model, output_folder, verbosity=0):  
    beginning_start = time.time()
    with open('./mteb/times.txt', 'a') as times:
        times.write('\n')
        times.write(output_folder)
        times.write('\n\n')
    for task in TASK_LIST:
        start_time = time.time()
        print(f"Running task: {task}")
        eval_splits = ["dev"] if task == "MSMARCO" else ["test"]
        evaluation = MTEB(tasks=[task])
        evaluation.run(model, output_folder=output_folder, verbosity=verbosity, eval_splits=eval_splits, batch_size=BATCH_SIZE)
        with open('./mteb/times.txt', 'a') as times:
            times.write(f"{task}: {time.time() - start_time}\n")
    with open('./mteb/times.txt', 'a') as times:
            times.write(f"Total time: {time.time() - beginning_start}\n")