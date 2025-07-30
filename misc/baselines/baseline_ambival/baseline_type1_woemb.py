"""

Baseline for AmbiVal dataset.
Call LLM API 1 turn, using prompt with example values.

"""


import os
import csv
import json
import time
import re
import random
import chromadb
from openai import OpenAI
from collections import defaultdict
from typing import Dict, Union, Any
import sqlite3
from func_timeout import func_set_timeout

from src.utils.item import Item

from tenacity import (
  retry,
  stop_after_attempt,
  wait_random_exponential,
)  # for exponential backoff

def openai_completion(prompt, max_tokens):
    response = openai_client.chat.completions.create(
        model=model_llm_api,
        messages=[
            {
                'role': 'user',
                'content': prompt
            }
        ],
        max_tokens=max_tokens,
        temperature=0
    )
    return response.choices[0].message.content
# @retry(wait=wait_random_exponential(min=10, max=1000), stop=stop_after_attempt(20))
def llm_api_completion(prompt):
    if "gpt" in model_llm_api:
        response = openai_completion(prompt,max_tokens=1000)
    elif "deepseek-ai/deepseek-r1" in model_llm_api  or "qwen/qwen2.5-coder-32b-instruct" == model_llm_api or "meta/llama-3.3-70b-instruct" == model_llm_api:
        response = openai_completion(prompt,max_tokens=10000)
        # time.sleep(10)
    elif "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free" ==  model_llm_api  or model_llm_api == "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free":
        response = openai_completion(prompt,max_tokens=1000)
    elif "deepseek/deepseek-r1" in model_llm_api or "meta-llama/llama-3.3-70b-instruct" in model_llm_api:
        print("Using inference.net")
        response = openai_completion(prompt,max_tokens=10000)
    return response

def parse_response_val(output):
    """
    Parse response from openai to get list of synonym values.

    :param response:
    :return:
    """
    if "</think>" in output:
        output = output.split("</think>")[-1]
    if "```sql" in output:
        output = output.split("```sql")[1].split("```")[0]
    elif "```" in output:
        output = output.split("```")[1].split("```")[0]
    output = re.sub(r"^\s+", "", output)
    return output

###################################################################################################
###################################################################################################
@func_set_timeout(10)
def execute_sql(db_path: str, sql: str, fetch: Union[str, int] = "all") -> Any:
    """
    Executes an SQL query on a database and fetches results.

    Args:
        db_path (str): The path to the database file.
        sql (str): The SQL query to execute.
        fetch (Union[str, int]): How to fetch the results. Options are "all", "one", "random", or an integer.

    Returns:
        Any: The fetched results based on the fetch argument.

    Raises:
        Exception: If an error occurs during SQL execution.
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            if fetch == "all":
                return cursor.fetchall()
            elif fetch == "one":
                return cursor.fetchone()
            elif fetch == "random":
                samples = cursor.fetchmany(10)
                return random.choice(samples) if samples else []
            elif isinstance(fetch, int):
                return cursor.fetchmany(fetch)
            else:
                raise ValueError("Invalid fetch argument. Must be 'all', 'one', 'random', or an integer.")
    except Exception as e:
        # print(f"Error in execute_sql: {e}\nSQL: {sql}")
        raise e
def _compare_sqls_outcomes(db_path: str, predicted_sql: str, ground_truth_sql: str) -> int:
    """
    Compares the outcomes of two SQL queries to check for equivalence.

    Args:
        db_path (str): The path to the database file.
        predicted_sql (str): The predicted SQL query.
        ground_truth_sql (str): The ground truth SQL query.

    Returns:
        int: 1 if the outcomes are equivalent, 0 otherwise.

    Raises:
        Exception: If an error occurs during SQL execution.
    """
    try:
        ground_truth_res = execute_sql(db_path, ground_truth_sql)
        predicted_res = execute_sql(db_path, predicted_sql)
    except:
        return 0, [], []
    return int(set(predicted_res) == set(ground_truth_res)), predicted_res, ground_truth_res


def eval_sql(amb_query, pred_query,db_id,databases_dir):
    if os.path.exists(os.path.join(databases_dir, db_id, db_id + ".sqlite")):
        db_path = os.path.join(databases_dir, db_id, db_id + ".sqlite")
    else:
        db_path = os.path.join(databases_dir, "wikisql.sqlite")

    result, predicted_res, ground_truth_res = _compare_sqls_outcomes(db_path, pred_query, amb_query)
    result_eval = {
            "result" : result,
            "predicted_res": predicted_res,
            "ground_truth_res": ground_truth_res,
    }
    return result_eval

if __name__ == "__main__":
    ###################################################################################
    #### Init
    ###################################################################################
    # data_path =  "/mnt/tampm/EmbeddingTaskAmbiText2SQL/AmbiVal_dataset/dev_final_dataset.json"
    # data_path =  "/mnt/tampm/EmbeddingTaskAmbiText2SQL/AmbiVal_dataset/dev_final_dataset_evidence.json"
    # db_context_dict_path = "/mnt/tampm/EmbeddingTaskAmbiText2SQL/AmbiVal_dataset/db_schema_dataset.json"
    # databases_dir = "/mnt/tampm/EmbeddingTaskAmbiText2SQL/AmbiVal_dataset/database_synonym_newques"  # root path to database dir, including all databases
    ###################    Spider  #####################################################
    data_path = "/mnt/tampm/EmbeddingTaskAmbiText2SQL/SpiderBIRD_dataset/dev_final_spider_evidence.json"
    databases_dir = "/mnt/tampm/data_text2sql/spider_data/database"
    ###################    BIRD  #####################################################
    # data_path = "/mnt/tampm/EmbeddingTaskAmbiText2SQL/SpiderBIRD_dataset/dev_final_bird_evidence.json"
    # databases_dir = "/mnt/tampm/data_text2sql/bird/dev_20240627/dev_databases"
    ###################################################################################
    items = []
    # model_llm_api = "gpt-4.1"
    # model_llm_api = "deepseek-ai/deepseek-r1" # nvidia
    # model_llm_api = "qwen/qwen2.5-coder-32b-instruct"   # nvidia
    model_llm_api = "meta/llama-3.3-70b-instruct"   # nvidia
    # model_llm_api = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
    # model_llm_api = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"
    # model_llm_api = "deepseek/deepseek-r1/fp-8"   # inference.net
    # model_llm_api = "meta-llama/llama-3.3-70b-instruct/fp-16"   # inference.net # 231

    if "gpt" in model_llm_api:
        api_key = os.environ["OPENAI_API_KEY"]
        openai_client = OpenAI(api_key=api_key)
    elif "deepseek-ai/deepseek-r1" == model_llm_api or "qwen/qwen2.5-coder-32b-instruct" == model_llm_api  or "meta/llama-3.3-70b-instruct" == model_llm_api:
        api_key = os.environ["NVIDIA_API_KEY"]
        openai_client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )
    elif "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free" ==  model_llm_api or model_llm_api == "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free":
        api_key = os.environ["TOGETHER_API_KEY"]
        openai_client = OpenAI(
              api_key=api_key,
              base_url="https://api.together.xyz/v1",
            )
    elif "deepseek/deepseek-r1" in model_llm_api or "meta-llama/llama-3.3-70b-instruct" in model_llm_api:
        api_key = os.environ["INFERENCE_API_KEY"]
        openai_client = OpenAI(
            base_url="https://api.inference.net/v1",
            api_key=api_key
        )
    model_llm_api_name = model_llm_api.replace("/", "_")
    # output_path = f"output_baseline_type1_{model_llm_api_name}_evidence.json"      # Cho local api
    output_path = f"spider_dev_baseline_type1_{model_llm_api_name}_evidence.json"      # Cho local api
    # output_path = f"bird_dev_baseline_type1_{model_llm_api_name}_evidence.json"      # Cho local api
    # if os.path.exists(output_path):
    #     os.remove(output_path)

    ###################################################################################
    ####### Load data in run_manager.py
    ###################################################################################
    with open(data_path) as f:
        data_dict_list = json.load(f)
    for item in data_dict_list:
        # print(item)
        item_obj = Item(
            db_id=item["db_id"],
            question=item["question"] + item['evidence'],
            schema_text_wo_content=item['schema_text_wo_content'],
            schema_text_with_content=item['schema_text_with_content'],
            evidence=item.get("evidence", ""),
            ori_query=item.get("ori_query", ""),
            amb_query=item.get("amb_query", "")
        )
        items.append(item_obj)
    # with open(db_context_dict_path) as f:
    #     db_context_dict = json.load(f)
    ###################################################################################
    ####### Run
    ###################################################################################
    # f_out = open(output_path, "a+")
    prompt_template = open("models/api_llm/templates/template_baseline_type1.txt", "r").read()
    idx = 0
    for item in items:
        db_id = item.db_id
        idx += 1
        print(f"Processing {idx} / {len(items)}: {item.question} - {db_id}")
        # SCHEMA_STR = item.schema_text_wo_content
        SCHEMA_STR = item.schema_text_with_content

        request_kwargs = {
            "SCHEMA_STR": SCHEMA_STR,
            "QUESTION_TEXT": item.question,
        }
        prompt = prompt_template.format(**request_kwargs)

        response_api = llm_api_completion(prompt)
        pred_query = parse_response_val(response_api)

        ###################################################################################
        # Evaluate column linking
        ###################################################################################
        result_eval = eval_sql(item.amb_query, pred_query, db_id,databases_dir)
        print(idx, model_llm_api ,result_eval , "pred_query:", pred_query)
        result = {
            "result": result_eval,
            "db_id": item.db_id,
            "question": item.question,
            "ori_query": item.ori_query,
            "amb_query": item.amb_query,
            "pred_query": pred_query,
            "response_api" : response_api
        }
        # f_out.write(json.dumps(result) + "\n")
    # f_out.close()



"""
python -m baselines.baseline_ambival.baseline_type1_woemb
"""