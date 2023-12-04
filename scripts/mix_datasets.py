import os
from argparse import ArgumentParser

from aime.utils import DATA_PATH

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_dataset_names", type=str, nargs="+")
    parser.add_argument("-o", "--output_dataset_name", type=str, required=True)
    args = parser.parse_args()

    input_folders = [
        os.path.join(DATA_PATH, dataset_name) for dataset_name in args.input_dataset_names
    ]
    output_folder = os.path.join(DATA_PATH, args.output_dataset_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    index = 0
    for input_folder in input_folders:
        data_names = sorted(os.listdir(input_folder))
        for data_name in data_names:
            input_data_path = os.path.join(input_folder, data_name)
            output_data_path = os.path.join(output_folder, f"{index}.npz")
            index += 1
            os.system(f"cp {input_data_path} {output_data_path}")
