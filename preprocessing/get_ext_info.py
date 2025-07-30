

"""

Add more information about stored value
Searching information about value in database

"""

import os
import re
import json
import csv
import sqlite3
import random
import argparse
from tqdm import tqdm
from openai import OpenAI
from tenacity import (
  retry,
  stop_after_attempt,
  wait_random_exponential,
)  # for exponential backoff

PROMPT_SEARCH_SAMPLE = """
In table `{table}` and column `{column}` store sample values: `{values}`.
What is the `{value}`?
Write the information of `{value}`. Write no more than 50 words.
"""

PROMPT_SEARCH = """
In table `{table}` and column `{column}`.
Explain the meaning of the `{value}`.
Write the information of `{value}`. Write no more than 50 words.
"""

@retry(wait=wait_random_exponential(min=10, max=1000), stop=stop_after_attempt(20))
def searching_information(db_id, table, column, value, client, model_llm_api, values):
    # print(PROMPT_SEARCH.format(table=table, column=column, value=value))
    # import pdb; pdb.set_trace()
    response = client.chat.completions.create(
        model=model_llm_api,
        messages=[
            {
                "role": "user",
                "content": PROMPT_SEARCH.format(table=table, column=column, value=value)
                # "content": PROMPT_SEARCH.format(table=table, column=column, value=value, values=values[:5])
            }
        ]
    )
    # import pdb; pdb.set_trace()
    return response.choices[0].message.content

# api_key = os.environ["OPENAI_API_KEY"]
def get_ext_know(file_name,output_path, model_llm_api):
    with open(file_name) as inf:
        distinct_values_data_dict = json.load(inf)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for db_id in distinct_values_data_dict:
        for table in distinct_values_data_dict[db_id]:
            for column in distinct_values_data_dict[db_id][table]:
                column_save_path = column.replace(" ", "_").replace("/", "_")
                save_path = os.path.join(output_path, f'{db_id}_{table}_{column_save_path}.json')
                if os.path.exists(save_path):
                    print(f"File {save_path} already exists, skipping...")
                    continue
                ext_know_dataset = {}
                ext_know_dataset[db_id] = {}
                ext_know_dataset[db_id][table] = {}
                ext_know_dataset[db_id][table][column] = {}
                distinct_value = distinct_values_data_dict[db_id][table][column]
                for value in distinct_value:
                    add_infor = searching_information(db_id, table, column, value, openai, model_llm_api, distinct_value)
                    ext_know_dataset[db_id][table][column][value] = {
                        model_llm_api: add_infor
                    }
                    print(f"db_id: {db_id}, table: {table}, column: {column}, value: {value} : {add_infor}")
                json.dump(ext_know_dataset, open(save_path, "w"), indent=4)

    return

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--databases_dir', default='database_synonym', type=str, help="Path to the directory containing the .sqlite databases")
    parser.add_argument('--output_path', default="output_embedding_update_1", type=str, help="Path to the output directory")
    parser.add_argument('--model', default='all-MiniLM-L6-v2', type=str, help="model name of sentence transformer")
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    databases_spider_dir = '../data_text2sql/spider_data/database'
    databases_bird_dir_dev = '../data_text2sql/bird/dev_20240627/dev_databases'
    databases_bird_dir_train = '../data_text2sql/bird/train/train_databases'

    # api_key = os.environ["OPENAI_API_KEY"]
    # model_llm_api = "gpt-4.1-mini"     #
    # model_llm_api = "gpt-4o"     #
    # model_llm_api = "deepseek-ai/deepseek-r1"
    model_llm_api = "meta/llama-3.3-70b-instruct"

    if "gpt" in model_llm_api:
        api_key = os.environ["OPENAI_API_KEY"]
        openai = OpenAI(api_key=api_key)
    elif "deepseek-ai/deepseek-r1" == model_llm_api or "meta/llama-3.3-70b-instruct" == model_llm_api:
        api_key = os.environ["NVIDIA_API_KEY"]
        openai = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )

    distinct_values_data_dict_path = "AmbiVal_dataset/distinct_values.json" # lấy từ script `get_all_value_from_sqlite.py`

    output_path = "AmbiVal_dataset/ext_know/"
    # output_path = "AmbiVal_dataset/ext_know_llama33_70b/"
    # model_name = args.model
    # model_name_path = model_name.replace("/", "_")
    get_ext_know(distinct_values_data_dict_path, output_path, model_llm_api)



"""
python  embed_value/add_information_saveinfile_ambival.py 
"""