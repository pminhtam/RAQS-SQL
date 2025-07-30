"""
Eval all SQL queries from output files.
Lấy tất cả câu SQL từ file output. Chỉ cần 1 câu đúng là được.
Eval script for evaluating the performance of a model from output files.

"""

import os
import json
import sqlite3
import itertools

from typing import Dict, Union, Any
import random
import func_timeout
from func_timeout import func_set_timeout


@func_set_timeout(20)
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

        ground_truth_res = list(itertools.chain.from_iterable(ground_truth_res))
        ground_truth_res = set(ground_truth_res)
        predicted_res = list(itertools.chain.from_iterable(predicted_res))
        predicted_res = set(predicted_res)
    except func_timeout.exceptions.FunctionTimedOut as e:
        return 0, [], []
    except:
        return 0, [], []

    return int(set(predicted_res) == set(ground_truth_res)), predicted_res, ground_truth_res
    # return int(set(predicted_res).issubset(set(ground_truth_res))), predicted_res, ground_truth_res


if __name__ == "__main__":
    # output_file = "output_api_deepseekr1_ambival_evidence_2.jsonl"  #   
    # output_file = "spider_dev_api_llama33.jsonl"  #   
    # output_file = "spider_dev_api_llama33_together.jsonl"  #   
    # output_file = "spider_dev_api_qwen25.jsonl"  #   
    # output_file = "spider_dev_api_deepseek.jsonl"  #   
    # output_file = "spider_dev_api_mistral_large.jsonl"  #   
    # output_file = "spider_dev_baseline_type1_qwen_qwen2.5-coder-32b-instruct_evidence.json"  #   
    # output_file = "spider_dev_baseline_type1_meta_llama-3.3-70b-instruct_evidence.json"  #   
    # output_file = "bird_dev_api_llama33.jsonl"  #   
    # output_file = "bird_dev_api_qwen25.jsonl"  #   
    # output_file = "bird_dev_api_mistral_large.jsonl"  #   
    # output_file = "bird_dev_api_qwen25_wo_evidence.jsonl"  #   
    # output_file = "bird_dev_api_qwen25_wo_evidence_rewriteEmpty.jsonl"  #   
    # output_file = "bird_dev_api_mistral_large_wo_evidence.jsonl"  #   
    # output_file = "bird_dev_api_mistral_large_wo_evidence_rewriteEmpty.jsonl"  #   
    # output_file = "bird_dev_api_llama33_wo_evidence.jsonl"  #   
    # output_file = "bird_dev_api_llama33_wo_evidence_rewriteEmpty.jsonl"  #   
    # output_file = "bird_dev_api_deepseek.jsonl"  #   
    output_file = "bird_dev_api_deepseek_wo_evidence_rewriteEmpty.jsonl"  # 
    # output_file = "bird_dev_baseline_type1_qwen_qwen2.5-coder-32b-instruct_evidence.json"  #   
    # output_file = "bird_dev_baseline_type1_meta_llama-3.3-70b-instruct_evidence.json"  #   
    if not os.path.exists(output_file):
        print(f"Output file {output_file} does not exist.")
        exit(1)

    with open(output_file, 'r') as f:
        lines = f.readlines()
    # databases_dir = "/mnt/tampm/EmbeddingTaskAmbiText2SQL/AmbiVal_dataset/database_synonym_newques"
    # databases_dir = "/mnt/tampm/data_text2sql/spider_data/database"
    databases_dir = "/mnt/tampm/data_text2sql/bird/dev_20240627/dev_databases"
    total_items = 0
    total_correct_items = 0


    for line in lines:
        item = json.loads(line.strip())
        db_id = item["db_id"]
        if os.path.exists(os.path.join(databases_dir, db_id, db_id + ".sqlite")):
            db_path = os.path.join(databases_dir, db_id, db_id + ".sqlite")
        else:
            db_path = os.path.join(databases_dir, "wikisql.sqlite")
        result, predicted_res, ground_truth_res = 0, [], []
        if "pipeline_log" in item:
            for previous_step in item["pipeline_log"]:
                if "sql_preds" in previous_step:
                    state = previous_step
                    ground_truth_sql = item["amb_query"]
                    for i in range(len(state["sql_preds"])):
                        predicted_sql = state["sql_preds"][i]["sql"]
                        result_i, predicted_res, ground_truth_res = _compare_sqls_outcomes(
                            db_path=db_path,
                            predicted_sql=predicted_sql,
                            ground_truth_sql=ground_truth_sql
                        )
                        result = max(result, result_i)  # Take the maximum result

        total_items += 1
        if result >= 1:
            total_correct_items += 1
        print(f"Rewrite result: {result}, predicted_res: {predicted_res}, ground_truth_res: {ground_truth_res}")
    print(f"Total items: {total_items}, Total correct items: {total_correct_items}")

"""
python evaluate_recall.py
"""
