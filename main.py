# file modified from original Apache-2.0 licensed code from https://github.com/Understanding-Visual-Datasets/VisDiff
# see LICENSE and NOTICE files in the root directory for details

import logging
from typing import Dict, List, Tuple, Set
import os
import glob
import json
import sys

import click
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm
from numpy import random

import wandb
from components.evaluator import LLMEvaluator, NullEvaluator, ExportEvaluator, EvaluatorEvaluator
from components.proposer import (
    LLMProposer,
    VLMProposer,
    VLMMultipleImageProposer,
    NullProposer,
)
try:
    from components.proposer import VLMQwenFeatureProposer
except ImportError:
    pass

from components.ranker import CLIPRanker, NullRanker

class TooFewImagesError(Exception):
    pass

repo_root = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(repo_root, "data_dir.json"), "r") as f:
    json_data = json.load(f)
data_dir = json_data["ad_data_dir"]
config_dir = json_data["config_dir"]
if not data_dir:
    data_dir = os.path.join(repo_root, "data", "ad-datasets")
else:
    data_dir = os.path.expanduser(data_dir)
if not config_dir:
    config_dir = os.path.join(repo_root, "configs")
else:
    config_dir = os.path.expanduser(config_dir)

def abs_path(p, type="data"):
    if type == "config":
        base_dir = config_dir
    elif type == "data":
        base_dir = data_dir
    else:
        raise ValueError(f"Unknown type {type} for abs_path")
    if os.path.isabs(p):
        return p
    return os.path.abspath(os.path.join(base_dir, p))

def load_config(config: str) -> Dict:
    base_cfg = OmegaConf.load("configs/base.yaml")
    cfg = OmegaConf.load(config)
    final_cfg = OmegaConf.merge(base_cfg, cfg)
    args = OmegaConf.to_container(final_cfg)
    args["config"] = config
    sweep_name = args.get("sweep_name", None)
    sweep_name = None if sweep_name is None else sweep_name[:128]  # wandb group name max length is 128 chars
    if args["wandb"]:
        dilution_dataset = args["data"].get("dilution_dataset", "None")
        dilution_dataset = os.path.splitext(os.path.basename(dilution_dataset))[0].replace("_train","").replace("_val","").replace("_test","").replace("_patches","")
        os.environ['WANDB_INIT_TIMEOUT'] = '36000'  # 10 hours
        wandb.init(
            project=args["project"],
            name=f'{args["data"].get("difference", args["data"]["group1"].replace(" ", "_")+"-"+args["data"]["group2"].replace(" ", "_"))} (p{args["data"].get("purity", 1)}/c{args["data"].get("concentration", 1)}+{dilution_dataset})',
            group=sweep_name,
            job_type=f'{args["data"]["category"]}'[:64], # wandb job_type max length is 64 chars
            notes=f'{args["data"]["group1"]} - {args["data"]["group2"]} (p{args["data"].get("purity", 1)}/c{args["data"].get("concentration", 1)}) of {sweep_name if sweep_name is not None else "single run"}',
            config=args,
            settings=wandb.Settings(init_timeout=36000,
                                    login_timeout=36000,
                                    summary_timeout=36000),
        )
    return args


def _load_data_csv(data_args: Dict) -> Tuple[List[Dict], List[Dict], List[str]]:
    if data_args.get("concentration", 1) != 1:
        raise ValueError("concentration parameter is not supported for csv mode")
    df = pd.read_csv(f"{data_args['root']}/{data_args['name']}.csv")

    if data_args["subset"]:
        old_len = len(df)
        df = df[df["subset"] == data_args["subset"]]
        print(
            f"Taking {data_args['subset']} subset (dataset size reduced from {old_len} to {len(df)})"
        )

    dataset1 = df[df["group_name"] == data_args["group1"]].to_dict("records")
    dataset2 = df[df["group_name"] == data_args["group2"]].to_dict("records")
    group_names = [data_args["group1"], data_args["group2"]]
    return dataset1, dataset2, group_names, [], []

def _load_data_dirs(data_args: Dict) -> Tuple[List[Dict], List[Dict], List[str]]:
    if data_args.get("concentration", 1) != 1:
        raise ValueError("concentration parameter is not supported for dir mode")
    dir1 = abs_path(data_args["group1_dir"])
    dir2 = abs_path(data_args["group2_dir"])
    dataset_group_num_images_to_sample = data_args["num_images_per_group"]
    if not os.path.exists(dir1) or not os.path.exists(dir2):
        raise ValueError(f"One of the specified directories does not exist: {dir1}, {dir2}, resolving to absolute paths: {os.path.abspath(dir1)}, {os.path.abspath(dir2)}")

    image_paths1 = glob.glob(os.path.join(dir1, "*.jpg"))
    image_paths2 = glob.glob(os.path.join(dir2, "*.jpg"))
    if len(image_paths1) < dataset_group_num_images_to_sample or len(image_paths2) < dataset_group_num_images_to_sample:
        raise TooFewImagesError(f"Not enough images in one of the directories to sample {dataset_group_num_images_to_sample} images. Dir1 has {len(image_paths1)} images, Dir2 has {len(image_paths2)} images.")
    group_names = [data_args["group1"], data_args["group2"]]
    dataset1 = [{"path": path, "group_name": data_args["group1"]} for path in random.choice(image_paths1, size=dataset_group_num_images_to_sample, replace=False)]
    dataset2 = [{"path": path, "group_name": data_args["group2"]} for path in random.choice(image_paths2, size=dataset_group_num_images_to_sample, replace=False)]
    return dataset1, dataset2, group_names, [], []

def _sample(rng: random.Generator, image_paths: List[str] | Set[str], num_samples: int) -> List[str]:
    if len(image_paths) < num_samples:
        raise TooFewImagesError(f"Not enough images to sample {num_samples} images. Only {len(image_paths)} images available. First few images: {list(image_paths)[:5]}")
    return rng.choice(list(image_paths), size=num_samples, replace=False).tolist()

def _sample_list(rng: random.Generator, seq: List[Dict], num_samples: int) -> List[Dict]:
    if len(seq) < num_samples:
        raise TooFewImagesError(f"Not enough items to sample {num_samples}. Only {len(seq)} available.")
    idx = rng.choice(len(seq), size=num_samples, replace=False).tolist()
    return [seq[i] for i in idx]

def _load_data_jsons(data_args: Dict) -> Tuple[List[Dict], List[Dict], List[str]]:
    group1_json = abs_path(data_args['group1_dir'])
    group2_json = abs_path(data_args['group2_dir'])
    dilution_dataset_json = abs_path(data_args['dilution_dataset']) if "dilution_dataset" in data_args else None
    if not os.path.exists(group1_json) or not os.path.exists(group2_json) or (dilution_dataset_json is not None and not os.path.exists(dilution_dataset_json)):
        raise ValueError(f"One of the specified json files does not exist: {group1_json}, {group2_json}, {dilution_dataset_json}, resolving to absolute paths: {os.path.abspath(group1_json)}, {os.path.abspath(group2_json)}, {os.path.abspath(dilution_dataset_json) if dilution_dataset_json is not None else 'N/A'}")
    num_images_per_group = data_args["num_images_per_group"]
    with open(group1_json, "r") as f:
        dataset1 = json.load(f)
        image_paths1 = {item["patchpath"] for item in dataset1["patches"]}
    with open(group2_json, "r") as f:
        dataset2 = json.load(f)
        image_paths2 = {item["patchpath"] for item in dataset2["patches"]}

    if data_args["remove_overlaps_A"]:
        image_paths1_no_overlap = image_paths1 - image_paths2
    if data_args["remove_overlaps_B"]:
        image_paths2_no_overlap = image_paths2 - image_paths1
    if data_args["remove_overlaps_A"] or data_args["remove_overlaps_B"]:
        logging.info(f"Removed overlapping images between groups, new sizes: Group1: {len(image_paths1_no_overlap) if data_args['remove_overlaps_A'] else len(image_paths1)}, Group2: {len(image_paths2_no_overlap) if data_args['remove_overlaps_B'] else len(image_paths2)}")

    # For desired concentration c (0 < c <= 1), the number of dilution images to add per group is:
    # x = N * (1/c - 1), where N is num_images_per_group.
    concentration = data_args.get("concentration", 1.0)
    assert 0.0 < concentration <= 1.0, "concentration must be in (0.0, 1.0]"
    if concentration < 1.0:
        with open(dilution_dataset_json, "r") as f:
            dilution_dataset = json.load(f)
            image_paths_dilution = {item["patchpath"] for item in dilution_dataset["patches"]}
            if data_args["remove_overlaps_dilution_A"]:
                image_paths_dilution -= image_paths1
            if data_args["remove_overlaps_dilution_B"]:
                image_paths_dilution -= image_paths2
            if data_args["remove_overlaps_dilution_A"] or data_args["remove_overlaps_dilution_B"]:
                logging.info(f"Removed overlapping images from dilution dataset, new size: {len(image_paths_dilution)}")

    image_paths1_sampled = _sample(data_args["rng"], image_paths1_no_overlap if data_args["remove_overlaps_A"] else image_paths1, num_images_per_group)
    image_paths2_sampled = _sample(data_args["rng"], image_paths2_no_overlap if data_args["remove_overlaps_B"] else image_paths2, num_images_per_group)
    dataset1 = [{"path": abs_path(path), "group_name": data_args["group1"]} for path in image_paths1_sampled]
    dataset2 = [{"path": abs_path(path), "group_name": data_args["group2"]} for path in image_paths2_sampled]

    if concentration < 1.0:
        # Number of dilution images to add per group to achieve target concentration
        dilution_images_per_group = int(round(num_images_per_group * (1.0 / concentration - 1.0)))
        # sample dilution images for each group, avoiding duplicate images
        image_paths_dilution1 = image_paths_dilution - set(image_paths1_sampled)
        image_paths_dilution_sampled1 = _sample(data_args["rng"], image_paths_dilution1, dilution_images_per_group)

        image_paths_dilution2 = (image_paths_dilution - set(image_paths2_sampled)) - set(image_paths_dilution_sampled1)
        image_paths_dilution_sampled2 = _sample(data_args["rng"], image_paths_dilution2, dilution_images_per_group)
    else:
        image_paths_dilution_sampled1 = []
        image_paths_dilution_sampled2 = []

    group_names = [data_args["group1"], data_args["group2"]]
    return dataset1, dataset2, group_names, image_paths_dilution_sampled1, image_paths_dilution_sampled2

def load_data(args: Dict) -> Tuple[List[Dict], List[Dict], List[str]]:
    data_args = args["data"] | {"context_mode": args.get("context_mode", "no_context")}
    csv_args_set = ("root" in data_args and "name" in data_args)
    dir_args_set = ("group1_dir" in data_args and "group2_dir" in data_args and "num_images_per_group" in data_args)
    json_args_set = ("group1" in data_args and "group2" in data_args and "num_images_per_group" in data_args and "group1_dir" in data_args and "group2_dir" in data_args)

    if "csv" in data_args["mode"].lower():
        if not csv_args_set:
            raise ValueError("For csv mode, data.root and data.name must be set in the config")
        dataset1, dataset2, group_names, dilution_paths1, dilution_paths2 = _load_data_csv(data_args)
    elif "dir" in data_args["mode"].lower():
        if not dir_args_set:
            raise ValueError("For dirs mode, data.group1_dir, data.group2_dir, and data.num_images_per_group must be set in the config")
        dataset1, dataset2, group_names, dilution_paths1, dilution_paths2 = _load_data_dirs(data_args)
    elif "json" in data_args["mode"].lower():
        if not json_args_set:
            raise ValueError("For jsons mode, data.group1, data.group2, data.num_images_per_group, data.group1_dir, and data.group2_dir must be set in the config")
        # If using dilution (concentration < 1), require dilution_dataset to be set
        if data_args.get("concentration", 1.0) < 1.0 and "dilution_dataset" not in data_args:
            raise ValueError("For jsons mode with concentration < 1, data.dilution_dataset must be set in the config")
        dataset1, dataset2, group_names, dilution_paths1, dilution_paths2 = _load_data_jsons(data_args)
    else:
        raise ValueError("data.mode must be one of 'csv', 'dirs' or 'jsons'")

    # 0% purity in the original VisDiff paper corresponds to purity=0.5 here
    assert 0.5 <= data_args["purity"] <= 1, "Purity must be between 0.5 and 1; 0.5 means completely random groups and no shift, 1 means pure groups"
    if data_args["purity"] < 1:
        logging.info(f"Purity is set to {data_args['purity']}. Swapping groups.")
        min_size = min(len(dataset1), len(dataset2))
        dataset1 = _sample_list(data_args["rng"], dataset1, min_size)
        dataset2 = _sample_list(data_args["rng"], dataset2, min_size)
        n_swap = int((1 - data_args["purity"]) * min_size)
        orig1, orig2 = dataset1, dataset2
        dataset1 = orig1[n_swap:] + orig2[:n_swap]
        dataset2 = orig2[n_swap:] + orig1[:n_swap]
    if dilution_paths1 or dilution_paths2:
        c = float(data_args.get('concentration', 1.0))
        logging.info(f"Adding dilution images to datasets: {len(dilution_paths1)} to group 1, {len(dilution_paths2)} to group 2; target concentration c={c}")
        dilution_dataset1 = [{"path": abs_path(path), "group_name": "dilution"} for path in dilution_paths1]
        dilution_dataset2 = [{"path": abs_path(path), "group_name": "dilution"} for path in dilution_paths2]

        if c >= 1.0:
            logging.info("Concentration >= 1.0, skipping dilution mixing.")
        else:
            N1, N2 = len(dataset1), len(dataset2)
            M1, M2 = len(dilution_dataset1), len(dilution_dataset2)

            # Compute the maximum feasible final size per group preserving target concentration c
            # S_i <= floor(N_i / c) and S_i <= floor(M_i / (1-c))
            def max_final_size(N, M, c):
                if c <= 0 or c >= 1:
                    return N  # degenerate cases already handled outside
                return max(0, min(int(N / c), int(M / (1.0 - c))))

            S1_max = max_final_size(N1, M1, c)
            S2_max = max_final_size(N2, M2, c)
            final_size = min(S1_max, S2_max)

            if final_size <= 0:
                logging.warning(
                    f"Unable to achieve target concentration c={c} due to insufficient dilution images (M1={M1}, M2={M2}). "
                    f"Resulting datasets may be empty; downstream TooFewImagesError may occur."
                )

            # Compute per-group keeps given final_size
            base1_keep = min(N1, int(final_size * c))
            dil1_keep = min(M1, max(0, final_size - base1_keep))
            base2_keep = min(N2, int(final_size * c))
            dil2_keep = min(M2, max(0, final_size - base2_keep))

            # Sample accordingly (preserve dict structure)
            if base1_keep > 0:
                dataset1 = _sample_list(data_args['rng'], dataset1, base1_keep)
            else:
                dataset1 = []
            if dil1_keep > 0:
                dilution_dataset1 = _sample_list(data_args['rng'], dilution_dataset1, dil1_keep)
            else:
                dilution_dataset1 = []

            if base2_keep > 0:
                dataset2 = _sample_list(data_args['rng'], dataset2, base2_keep)
            else:
                dataset2 = []
            if dil2_keep > 0:
                dilution_dataset2 = _sample_list(data_args['rng'], dilution_dataset2, dil2_keep)
            else:
                dilution_dataset2 = []

            # Merge and shuffle
            dataset1 = (dataset1 + dilution_dataset1)
            dataset2 = (dataset2 + dilution_dataset2)
            if dataset1:
                dataset1 = data_args['rng'].permutation(dataset1).tolist()
            if dataset2:
                dataset2 = data_args['rng'].permutation(dataset2).tolist()
    logging.info(f"Final loaded dataset sizes: Group 1: {len(dataset1)}, Group 2: {len(dataset2)}")
    if len(dataset1) < 5 or len(dataset2) < 5:
        raise TooFewImagesError(f"Not enough images in one of the datasets. Dataset 1 has {len(dataset1)} images, Dataset 2 has {len(dataset2)} images.")
    return dataset1, dataset2, group_names


def propose(args: Dict, dataset1: List[Dict], dataset2: List[Dict]) -> List[str]:
    proposer_args = args["proposer"] | args["hostnames"] | {"context_mode": args.get("context_mode", "no_context")}
    proposer_args["captioner"] = args["captioner"]

    proposer = eval(proposer_args["method"])(proposer_args, proposer_args["rng"])
    hypotheses, logs, images = proposer.propose(dataset1, dataset2)
    if args["wandb"]:
        wandb.log({"logs": wandb.Table(dataframe=pd.DataFrame(logs))})
        for i in range(len(images)):
            wandb.log(
                {
                    f"group 1 images ({dataset1[0]['group_name']})": images[i][
                        "images_group_1"
                    ],
                    f"group 2 images ({dataset2[0]['group_name']})": images[i][
                        "images_group_2"
                    ],
                }
            )
    return hypotheses


def rank(
    args: Dict,
    hypotheses: List[str],
    dataset1: List[Dict],
    dataset2: List[Dict],
    group_names: List[str],
) -> List[str]:
    ranker_args = args["ranker"] | args["hostnames"] | {"context_mode": args.get("context_mode", "no_context")}

    ranker = eval(ranker_args["method"])(ranker_args, ranker_args["rng"])

    scored_hypotheses = ranker.rerank_hypotheses(hypotheses, dataset1, dataset2)
    if args["wandb"]:
        table_hypotheses = wandb.Table(dataframe=pd.DataFrame(scored_hypotheses))
        wandb.log({"scored hypotheses": table_hypotheses})
        for i in range(5):
            wandb.summary[f"top_{i + 1}_difference"] = scored_hypotheses[i].get(
                "hypothesis", "no hypothesis"
            ).replace('"', "")
            wandb.summary[f"top_{i + 1}_score"] = scored_hypotheses[i].get("auroc", "no auroc")

    scored_groundtruth = ranker.rerank_hypotheses(
        group_names,
        dataset1,
        dataset2,
    )
    if args["wandb"]:
        table_groundtruth = wandb.Table(dataframe=pd.DataFrame(scored_groundtruth))
        wandb.log({"scored groundtruth": table_groundtruth})

    return [hypothesis["hypothesis"] for hypothesis in scored_hypotheses]


def evaluate(args: Dict, ranked_hypotheses: List[str], group_names: List[str], manual_labels = None) -> Dict:
    evaluator_args = args["evaluator"] | args["hostnames"] | {"context_mode": args.get("context_mode", "no_context")}

    evaluator = eval(evaluator_args["method"])(evaluator_args)
    evaluate_args = (
        ranked_hypotheses,
        group_names[0],
        group_names[1],
    )
    if manual_labels is not None:
        evaluate_args = evaluate_args + (manual_labels,)
    metrics, evaluated_hypotheses = evaluator.evaluate(*evaluate_args)

    if args["wandb"] and evaluator_args["method"] != "NullEvaluator":
        table_evaluated_hypotheses = wandb.Table(
            dataframe=pd.DataFrame(evaluated_hypotheses)
        )
        wandb.log({"evaluated hypotheses": table_evaluated_hypotheses})
        wandb.log(metrics)
    return metrics


@click.command()
@click.option("--config", help="config file")
@click.option("--clip-hostname", default="localhost", help="CLIP server hostname")
@click.option("--vlm-hostname", default="localhost", help="VLM server hostname")
@click.option("--proposer-llm-hostname", default="localhost", help="LLM for proposer server hostname")
@click.option("--eval-llm-hostname", default="localhost", help="LLM for evaluator server hostname")
@click.option("--accumulation-file-path", default=None, help="File path to accumulate stats across multiple runs, requires import_ranked_hypotheses_file to be set in config to load labeled hypotheses")
@click.option("--context-mode", default=None, help="Override context mode in config, must be one of 'no_context', 'red_bbox', or 'centered', and must match the data directories specified in the config")
def main(config, clip_hostname, vlm_hostname, proposer_llm_hostname, eval_llm_hostname, accumulation_file_path, context_mode):
    logging.info("Loading config...")
    args = load_config(config)

    prompt_context_mode = args["context_mode"] = context_mode if context_mode is not None else args.get("context_mode", "no_context")
    cfg_path = args.get("cfg_path", "")
    if prompt_context_mode == "no_context":
        assert ("patches_with_padding" not in cfg_path) and ("bbox" not in cfg_path), f"context_mode is not set but config {cfg_path} contains context-specific data directories, please set context_mode in config to match the data directories"
    else:
        assert "patches_with_padding" in cfg_path, f"context_mode is set to {prompt_context_mode} but config {cfg_path} does not contain patches_with_padding data directories, please set context_mode in config to match the data directories"
        if prompt_context_mode == "red_bbox":
            assert "bbox" in cfg_path, f"context_mode is set to {prompt_context_mode} but config {cfg_path} does not contain bbox data directories, please set context_mode in config to match the data directories"
        elif prompt_context_mode == "centered":
            assert cfg_path and "bbox" not in cfg_path, f"context_mode is set to {prompt_context_mode} but config {cfg_path} contains bbox data directories, please set context_mode in config to match the data directories"
        else:
            raise ValueError(f"Invalid context_mode: {prompt_context_mode}")

    hostnames = {
        "clip_hostname": clip_hostname,
        "vlm_hostname": vlm_hostname,
        "proposer_llm_hostname": proposer_llm_hostname,
        "eval_llm_hostname": eval_llm_hostname,
    }
    args["hostnames"] = hostnames
    if accumulation_file_path is not None:
        args["evaluator"]["stats_accumulation_file_path"] = accumulation_file_path

    # guarantees proper statistically independent runs, see https://numpy.org/doc/2.1/reference/random/parallel.html#sequence-of-integer-seeds
    seed_sequence = random.SeedSequence() if args.get("root_seed", None) is None else random.SeedSequence([args.get("comparison_count_for_rng", 0), args["root_seed"]])
    args["proposer"]["rng"], args["ranker"]["rng"], args["evaluator"]["rng"], args["data"]["rng"] = [random.Generator(random.Philox(s)) for s in seed_sequence.spawn(4)]

    # print(args)

    logging.info("Loading data...")
    try:
        dataset1, dataset2, group_names = load_data(args)
    except TooFewImagesError as e:
        logging.error(f"TooFewImagesError: {e}")
        sys.exit(3)
    # print(dataset1, dataset2, group_names)

    logging.info("Proposing hypotheses...")
    hypotheses = propose(args, dataset1, dataset2)
    # print(hypotheses)

    if accumulation_file_path is None:
        logging.info("Ranking hypotheses...")
        ranked_hypotheses = rank(args, hypotheses, dataset1, dataset2, group_names)
        manual_labels = None
        # print(ranked_hypotheses)
    else:
        with open(args["import_ranked_hypotheses_file"], "r") as f:
            filtered_entries = [l for l in (json.loads(line) for line in f) if l["group_a"] == group_names[0] and l["group_b"] == group_names[1]]
            ranked_hypotheses = [entry["hypothesis"] for entry in filtered_entries]
            manual_labels = [entry.get("manual_label", None) for entry in filtered_entries]
            if all(l is None for l in manual_labels):
                print("no labeled hypotheses in image set pair, exiting")
                sys.exit(1)

    logging.info("Evaluating hypotheses...")
    metrics = evaluate(args, ranked_hypotheses, group_names, manual_labels=manual_labels)
    # print(metrics)


if __name__ == "__main__":
    main()
