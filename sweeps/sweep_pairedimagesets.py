# file modified from original Apache-2.0 licensed code from https://github.com/Understanding-Visual-Datasets/VisDiff
# see LICENSE and NOTICE files in the root directory for details

import json
import os
import secrets
import datetime

import click
from numpy import random


@click.command()
@click.option("--root-seed", type=int, help="root seed for the random number generators, no seed will pull system entropy")
@click.option("--purity", default=1.0, type=float)
@click.option("--clip-hostname", default="localhost", help="CLIP server hostname")
@click.option("--vlm-hostname", default="localhost", help="VLM server hostname")
@click.option("--proposer-llm-hostname", default="localhost", help="LLM for proposer server hostname")
@click.option("--eval-llm-hostname", default="localhost", help="LLM for evaluator server hostname")
@click.option("--accumulation-file-prefix", default=None, help="File path prefix to accumulate stats across multiple runs")
@click.option("--no-abort-on-skipped", is_flag=True, default=False, help="Do not abort the sweep if a run is skipped")
@click.option("--bench-root", default="data/AD-Diff_Bench", help="Root directory for Benchmark data")
@click.option("--benchmark-name", default="AD-Diff_Bench", help="Name of the benchmark to run")
@click.option("--run-name", type=str, help="Name of this sweep, prefix for wandb group name")
@click.option("--continue-from-index", type=int, default=None, help="Index to continue the sweep from, skipping all prior indices")
@click.option("--sweep-name", type=str, default=None, help="Name of the sweep to continue from, to match wandb group name")
def main(purity: float, root_seed: int, clip_hostname: str, vlm_hostname: str, proposer_llm_hostname: str, eval_llm_hostname: str, accumulation_file_prefix: str, no_abort_on_skipped: bool, bench_root: str, benchmark_name: str, run_name: str, continue_from_index: int, sweep_name: str):
    if continue_from_index is not None:
        if sweep_name is None:
            raise ValueError("sweep_name must be provided when continue_from_index is set")
        print(f"Continuing sweep \"{sweep_name}\" from index {continue_from_index} over {bench_root} with purity {purity}")
        sweep_name = sweep_name
        if root_seed is None:
            raise ValueError("root_seed must be provided when continue_from_index is set")
        start_index = continue_from_index
    else:
        if sweep_name is not None:
            raise ValueError("sweep_name should not be provided when not continuing from an index. Set run_name instead.")
        if run_name is not None and len(run_name) > 118:
            raise ValueError("run_name must be at most 118 characters to fit within wandb group name limit (so that the timestamp can be appended)")
        sweep_name = f"{run_name}_{datetime.datetime.now().strftime('%y_%m_%d-%H_%M_%S')}"
        print(f"Starting sweep \"{sweep_name}\" over {bench_root} with purity {purity}")
        if root_seed is None:
            root_seed = secrets.randbits(128)
        else:
            if root_seed < 1000:
                raise ValueError("entropy should be a large integer")
            print(f"WARNING: Using provided root seed: {root_seed}. Prefer to not set root_seed to pull from OS unless you intend to reproduce a previous run.")

        print(f"Using root seed: {root_seed}. To reproduce this run, set root_seed to this value.")

        with open(f"sweep_meta_{sweep_name.strip().replace(' ', '-').replace('/', '-').replace(',', '-')}.txt", "w") as f:
            f.write(f"benchmark_name: {benchmark_name}\n")
            f.write(f"sweep_name: {sweep_name}\n")
            f.write(f"timestamp: {datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}\n")
            f.write(f"bench_root: {bench_root}\n")
            f.write(f"purity: {purity}\n")
            f.write(f"root_seed: {root_seed}\n")
        start_index = 0


    easy = [json.loads(line) for line in open(f"{bench_root}/easy.jsonl")]
    medium = [json.loads(line) for line in open(f"{bench_root}/medium.jsonl")]
    hard = [json.loads(line) for line in open(f"{bench_root}/hard.jsonl")]
    data = easy + medium + hard

    accumulation_file_path = None if accumulation_file_prefix is None else f"{accumulation_file_prefix}_{datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}.json"
    for idx in range(start_index, len(data)):
        item = data[idx]
        cfg = f"""
project: {benchmark_name}
root_seed: {root_seed}
comparison_count_for_rng: {idx}
sweep_name: {sweep_name}
data:
  name: {benchmark_name}
  group1: "{item['set1']}"
  group2: "{item['set2']}"
  difference: "{item['difference']}"
  category: "{item.get('category', 'No category')}"
  purity: {purity}
"""

        difficulty = (
            "easy"
            if idx < len(easy)
            else "medium"
            if idx < len(easy) + len(medium)
            else "hard"
        )
        cfg_dir = f"configs/sweep_visdiffbench_purity{purity}_rootseed{root_seed}"
        if not os.path.exists(cfg_dir):
            os.makedirs(cfg_dir)
        cfg_file = f"{cfg_dir}/{idx}_{difficulty}.yaml"
        with open(cfg_file, "w") as f:
            f.write(cfg)
        print(f"python main.py --config {cfg_file} --context-mode no_context --clip-hostname {clip_hostname} --vlm-hostname {vlm_hostname} --proposer-llm-hostname {proposer_llm_hostname} --eval-llm-hostname {eval_llm_hostname} {'--accumulation-file-path ' + accumulation_file_path if accumulation_file_path is not None else ''}")
        status_code = os.system(f"python main.py --config {cfg_file} --context-mode no_context --clip-hostname {clip_hostname} --vlm-hostname {vlm_hostname} --proposer-llm-hostname {proposer_llm_hostname} --eval-llm-hostname {eval_llm_hostname} {'--accumulation-file-path ' + accumulation_file_path if accumulation_file_path is not None else ''}")
        if os.WIFEXITED(status_code):
            exit_code = os.WEXITSTATUS(status_code)
            if exit_code == 1:
                if no_abort_on_skipped:
                    print(f"Run for index {idx} was skipped, continuing sweep as per flag.")
                    continue
                print(f"Run for index {idx} signaled to stop sweep.")
                break
            elif exit_code != 0:
                print(f"Run for index {idx} failed with exit code {exit_code}. Stopping sweep.")
                break
        else:
            print(f"Run for index {idx} terminated abnormally. Stopping sweep.")
            break

if __name__ == "__main__":
    main()
