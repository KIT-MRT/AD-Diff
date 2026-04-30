# file modified from original Apache-2.0 licensed code from https://github.com/Understanding-Visual-Datasets/VisDiff
# see LICENSE and NOTICE files in the root directory for details

import json
import os
import secrets
import datetime
import glob
import pathlib
import string
import random as rnd

import click
from numpy import random
import yaml

repo_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
with open(os.path.join(repo_root, "data_dir.json"), "r") as f:
    json_data = json.load(f)
sweep_validity_warnings_log = json_data["sweep_validity_warnings_log"]
if not sweep_validity_warnings_log:
    sweep_validity_warnings_log = "/dev/null"
else:
    sweep_validity_warnings_log = os.path.expanduser(sweep_validity_warnings_log)

@click.command()
@click.option("--root-seed", type=int, help="root seed for the random number generators, no seed will pull system entropy")
@click.option("--purity", default=1.0, type=float)
@click.option("--concentration", default=1.0, type=float, help="Concentration of the main groups vs dilution images, 1.0 means no dilution images. Configs with a dilution dataset with too few images will be skipped if the concentration is too low.")
@click.option("--clip-hostname", default="localhost", help="CLIP server hostname")
@click.option("--vlm-hostname", default="localhost", help="VLM server hostname")
@click.option("--proposer-llm-hostname", default="localhost", help="LLM for proposer server hostname")
@click.option("--eval-llm-hostname", default="localhost", help="LLM for evaluator server hostname")
@click.option("--no-abort-on-skipped", is_flag=True, default=False, help="Do not abort the sweep if a run is skipped")
@click.option("--bench-root", default="configs/metadata_filtered/splits_with_ground_truth", help="Root directory for Benchmark configs, set this to subdir to run a subset of the benchmark")
@click.option("--run-name", type=str, help="Name of this sweep, prefix for wandb group name")
@click.option("--continue-from-index", type=int, default=None, help="Index to continue the sweep from, skipping all prior indices")
@click.option("--sweep-name", type=str, default=None, help="Name of the sweep to continue from, to match wandb group name")
@click.option("--project-name", type=str, default="ad_datasets", help="Name of the project to set in the config files for this sweep")
def main(root_seed: int, purity: float, concentration: float, clip_hostname: str, vlm_hostname: str, proposer_llm_hostname: str, eval_llm_hostname: str, no_abort_on_skipped: bool, bench_root: str, run_name: str, continue_from_index: int, sweep_name: str, project_name: str):
    if continue_from_index is not None:
        if sweep_name is None:
            raise ValueError("sweep_name must be provided when continue_from_index is set")
        print(f"Continuing sweep \"{sweep_name}\" from index {continue_from_index} over {bench_root} with purity {purity} and concentration {concentration}")
        if root_seed is None:
            raise ValueError("root_seed must be provided when continue_from_index is set")
        start_index = continue_from_index
    else:
        if sweep_name is not None:
            raise ValueError("sweep_name should not be provided when not continuing from an index. Set run_name instead.")
        if run_name is not None and len(run_name) > 107:
            run_name = run_name[-107:]
            print(f"WARNING: run_name is too long, truncating to last 107 characters: {run_name}")
        sweep_name = f"{run_name}_{datetime.datetime.now().strftime('%y_%m_%d-%H_%M_%S')}-{''.join(rnd.choices(string.ascii_lowercase + string.ascii_uppercase + string.digits, k=2))}"
        print(f"Starting sweep \"{sweep_name}\" over {bench_root} with purity {purity}")
        if root_seed is None:
            root_seed = secrets.randbits(128)
        else:
            if root_seed < 1000:
                raise ValueError("entropy should be a large integer")
            print(f"WARNING: Using provided root seed: {root_seed}. Prefer to not set root_seed to pull from OS unless you intend to reproduce a previous run.")
        start_index = 0

    print(f"Using root seed: {root_seed}. To reproduce this run, set root_seed to this value.")

    yaml_list = glob.glob(f"{bench_root}/**/*.yaml", recursive=True)
    data = sorted(yaml_list)
    print(f"Found {len(data)} config files to run.")

    with open(f"sweep_{project_name.replace(' ', '_')}_summary_{sweep_name.replace(' ', '_').replace('/', '-').replace(',', '-')}.txt", "w") as summary_f:
        summary_f.write(f"Sweep Name: {sweep_name}\n")
        summary_f.write(f"Root Seed: {root_seed}\n")
        summary_f.write(f"Purity: {purity}\n")
        summary_f.write(f"Concentration: {concentration}\n")
        summary_f.write(f"Benchmark Root: {bench_root}\n")
        summary_f.write(f"Configs:\n")
        for cfg in data:
            summary_f.write(f"{cfg}\n")

    for i, cfg in enumerate(data[start_index:], start=start_index):
        print(f"Running config {i}/{len(data)-1}: {cfg}")
        cfg_dir = f"configs/sweep_{project_name.replace(' ', '_')}_purity{purity}_concentration{concentration}_rootseed{root_seed}"
        if not os.path.exists(cfg_dir):
            os.makedirs(cfg_dir)
        cfg_file = f"{cfg_dir}/{pathlib.Path(cfg).stem}_purity{purity}_concentration{concentration}.yaml"
        with open(cfg_file, "w") as f:
            with open(cfg, "r") as original_f:
                cfg_data = yaml.safe_load(original_f)
                cfg_data["data"]["purity"] = purity
                cfg_data["data"]["concentration"] = concentration
                cfg_data["data"]["category"] = f"{pathlib.Path(cfg).parents[1].stem} - {pathlib.Path(cfg).parents[0].stem}"
                cfg_data["project"] = project_name
                cfg_data["root_seed"] = root_seed
                cfg_data["comparison_count_for_rng"] = i
                cfg_data["sweep_name"] = sweep_name
                cfg_data["cfg_path"] = cfg
                yaml.dump(cfg_data, f)


        print(f"python main.py --config {cfg_file} --clip-hostname {clip_hostname} --vlm-hostname {vlm_hostname} --proposer-llm-hostname {proposer_llm_hostname} --eval-llm-hostname {eval_llm_hostname}")
        status_code = os.system(f"python main.py --config {cfg_file} --clip-hostname {clip_hostname} --vlm-hostname {vlm_hostname} --proposer-llm-hostname {proposer_llm_hostname} --eval-llm-hostname {eval_llm_hostname}")
        if os.WIFEXITED(status_code):
            exit_code = os.WEXITSTATUS(status_code)
            if exit_code == 1:
                if no_abort_on_skipped:
                    with open(sweep_validity_warnings_log, "a") as log_f:
                        log_f.write(f"WARNING: {sweep_name}: FLAG: run for config {cfg_file} (original config {cfg}, run number{i}) was skipped due to general error, continuing sweep as per flag.\n")
                    print(f"Sweep {sweep_name}: Run for config {cfg_file} (original config {cfg}, run number{i}) was skipped, continuing sweep as per flag.")
                    continue
                print(f"Sweep {sweep_name}: Run for config {cfg_file} (original config {cfg}, run number{i}) signaled to stop sweep.")
                break
            elif exit_code == 2:
                if no_abort_on_skipped:
                    with open(sweep_validity_warnings_log, "a") as log_f:
                        log_f.write(f"WARNING: {sweep_name}: CLI-ARGUMENT-ERROR: run for config {cfg_file} (original config {cfg}, run number{i}) was skipped due to invalid command line arguments, continuing sweep as per flag.\n")
                    continue
                print(f"Sweep {sweep_name}: Run for config {cfg_file} (original config {cfg}, run number{i}) signaled to stop sweep due to invalid dataset.")
                break
            elif exit_code == 3:
                with open(sweep_validity_warnings_log, "a") as log_f:
                    log_f.write(f"ERROR:   {sweep_name}: INSUFFICIENT-IMAGES: run for config {cfg_file} (original config {cfg}, run number{i}) was skipped due to insufficient images given concentration {concentration}, continuing sweep.\n")
                print(f"Sweep {sweep_name}: Run for config {cfg_file} (original config {cfg}, run number{i}) signaled to skip run due to insufficient images. Continuing sweep.")
            elif exit_code != 0:
                print(f"Sweep {sweep_name}: Run for config {cfg_file} (original config {cfg}, run number{i}) failed with exit code {exit_code}. Stopping sweep.")
                break
        else:
            print(f"Sweep {sweep_name}: Run for config {cfg_file} (original config {cfg}, run number{i}) terminated abnormally. Stopping sweep.")
            break

if __name__ == "__main__":
    main()
