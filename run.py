import os
import argparse
import yaml
from tqdm import tqdm
from src.utils.run_manager import RunManager


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="configs/api_gpt4.yaml")
    parser.add_argument("--output_path", type=str, default="output_groq_step1.jsonl")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    config_file = args.config_file

    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    run_manager = RunManager(config)
    output_path = args.output_path
    if os.path.exists(output_path):
        os.remove(output_path)
    idx = 0
    for item in tqdm(run_manager.items):
        print(idx, item.question)
        run_time, total_item_num_input_tokens, total_item_num_output_tokens = run_manager._run_item(item,output_path=output_path)
        idx += 1
    pass
