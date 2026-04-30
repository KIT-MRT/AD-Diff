# file modified from original Apache-2.0 licensed code from https://github.com/Understanding-Visual-Datasets/VisDiff
# see LICENSE and NOTICE files in the root directory for details

from glob import glob

path_to_group_1 = "./Ampeln/gelb_vs_schwarz/gelb/"
path_to_group_2 = "./Ampeln/gelb_vs_schwarz/schwarz/"
experiment_name = "TrafficLightsYellowBlack"
description_group_1 = "Traffic light with yellow housing"
description_group_2 = "Traffic light with black housing"


def main():
    yaml_file_text = f'''
data:
  name: {experiment_name}
  group1: "{description_group_1}"
  group2: "{description_group_2}"
'''
    yaml_file_name = f"{experiment_name}.yaml"
    with open(yaml_file_name, "w") as f:
        f.write(yaml_file_text)
        print(f"saved {yaml_file_name}")


    file_name = f"{experiment_name}.csv"
    def writeGroup(path, description):
        img_paths = glob(path + "*.jpg") + glob(path + "*.jpeg") + glob(path + "*.png")
        with open(file_name, "a") as f:
            for img_path in img_paths:
                f.write(f"{description},{img_path}\n")

    header = "group_name,path\n"
    with open(file_name, "w") as f:
        f.write(header)
    writeGroup(path_to_group_1, description_group_1)
    writeGroup(path_to_group_2, description_group_2)
    print(f"saved {file_name}")


if __name__=="__main__":
    main()
