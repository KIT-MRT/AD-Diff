import os
import json
import yaml
import subprocess

from tqdm import tqdm

DATA_ROOT = "data/ad-datasets"
KEYWORD_JSON_PATH = "clip_filtered_bench_keywords.json"
INDICE_FOLDER = "clip_index"
OUTPUT_DATA_PATH = "clip_filtered"
OUTPUT_CONFIG_PATH = "configs/clip_filtered"
EXTRACTED_PATCHES_ROOT = "extracted_patches"
IMAGES_CONTEXT_PATH_COMPONENT = "images_pad0-5_clip_bbox255-0-0_th5"
NUM_SAMPLES_PER_KEYWORD = 100

def main():

    with open(os.path.join(DATA_ROOT, KEYWORD_JSON_PATH), "r") as f:
        keyword_data = json.load(f)

    for dataset_name, keywords in tqdm(keyword_data["datasets"].items()):
        dataset_indice_folder = os.path.join(INDICE_FOLDER, dataset_name, "train")
        if not os.path.exists(os.path.join(DATA_ROOT, dataset_indice_folder)):
            print(f"Warning: Indice folder for dataset {dataset_name} does not exist at path {os.path.join(DATA_ROOT, dataset_indice_folder)}. Skipping this dataset.")
            continue

        output_dataset_path = os.path.join(OUTPUT_DATA_PATH, dataset_name, "train")
        output_config_dataset_path = os.path.join(OUTPUT_CONFIG_PATH, dataset_name)
        os.makedirs(os.path.join(DATA_ROOT, output_dataset_path), exist_ok=True)
        os.makedirs(os.path.join(DATA_ROOT, output_config_dataset_path), exist_ok=True)

        for keyword_category, keyword_list in tqdm(keywords.items(), desc=f"Processing dataset '{dataset_name}'"):
            yaml_dir = os.path.join(output_config_dataset_path, keyword_category)
            os.makedirs(yaml_dir, exist_ok=True)
            for keyword in tqdm(keyword_list, desc=f"Processing keywords in category '{keyword_category}' for dataset '{dataset_name}'", leave=False):
                indice_folder = os.path.join(dataset_indice_folder, keyword_category, IMAGES_CONTEXT_PATH_COMPONENT)
                if not os.path.exists(os.path.join(DATA_ROOT, indice_folder)):
                    print(f"Warning: Indice folder for keyword '{keyword}' in category '{keyword_category}' for dataset '{dataset_name}' does not exist at path {os.path.join(DATA_ROOT, indice_folder)}. Skipping this keyword.")
                    continue
                output_folder = os.path.join(output_dataset_path, keyword_category, IMAGES_CONTEXT_PATH_COMPONENT, keyword.replace(" ", "_"))
                os.makedirs(os.path.join(DATA_ROOT, output_folder), exist_ok=True)
                current_dir = os.path.dirname(os.path.abspath(__file__))
                original_dir = os.getcwd()
                try:
                    os.chdir(os.path.join(current_dir, "ad-datasets"))
                    subprocess.run(["clip-retrieval", "filter", "--query", keyword, "--output_folder", output_folder, "--indice_folder", indice_folder, "--num_results", str(NUM_SAMPLES_PER_KEYWORD)], check=True)
                finally:
                    os.chdir(original_dir)
                # Create a config file for this keyword
                config_data = {
                    "data": {
                        "mode": "dirs",
                        "group1": keyword,
                        "group2": keyword_category,
                        "group1_dir": output_folder,
                        "group2_dir": os.path.join(EXTRACTED_PATCHES_ROOT, dataset_name, "train", keyword_category, IMAGES_CONTEXT_PATH_COMPONENT),
                        "num_images_per_group": NUM_SAMPLES_PER_KEYWORD
                    }
                }
                config_file_path = os.path.join(yaml_dir, f"{keyword.replace(' ', '_')}.yaml")
                with open(config_file_path, "w") as config_file:
                    yaml.dump(config_data, config_file, default_flow_style=False)

if __name__ == "__main__":
    main()