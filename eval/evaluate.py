"""

Eval script for evaluating the performance of a model from output files.

"""

import os
import json
import sqlite3
import itertools
import argparse

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        required=True, type=str,
                        choices=["ambival", "spider","bird"])
    parser.add_argument('--output_path',
                        required=True, type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    output_file = args.output_path
    if args.dataset == "ambival":
        databases_dir = "misc/dataset/ambival/database_test/"
    elif args.dataset == "spider":
        databases_dir = "misc/dataset/SpiderBIRD_dataset/spider/spider_data/database"
    elif args.dataset == "bird":
        databases_dir = "misc/dataset/SpiderBIRD_dataset/bird/dev_20240627/dev_databases"
    else:
        print(f"Unsupported dataset: {args.dataset}")
        exit(1)
    if not os.path.exists(output_file):
        print(f"Output file {output_file} does not exist.")
        exit(1)

    with open(output_file, 'r') as f:
        lines = f.readlines()

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
                # print(previous_step["step"])
                if "re_write" == previous_step["step"]:
                    state = previous_step
                    result, predicted_res, ground_truth_res = _compare_sqls_outcomes(
                        db_path=db_path,
                        predicted_sql=state["response"]['SQL'],
                        ground_truth_sql=item["amb_query"]
                    )
                    break
                if "select_final" == previous_step["step"]:
                    state = previous_step
                    ground_truth_sql = item["amb_query"]
                    predicted_sql = state["sql_preds"][0]["sql"]
                    result, predicted_res, ground_truth_res = _compare_sqls_outcomes(
                        db_path=db_path,
                        predicted_sql=predicted_sql,
                        ground_truth_sql=ground_truth_sql
                    )
                    break
        else:
            ground_truth_sql = item["ori_query"]
            predicted_sql = item["pred_query"]
            result, predicted_res, ground_truth_res = _compare_sqls_outcomes(
                db_path=db_path,
                predicted_sql=predicted_sql,
                ground_truth_sql=ground_truth_sql
            )
        total_items += 1
        if result == 1:
            total_correct_items += 1
    print(f"Total items: {total_items}, Total correct items: {total_correct_items}")
