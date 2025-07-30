"""

Step 4 : Dùng api để rewrite câu sql query với value tương đồng với value trong câu hỏi.

"""
import os
import time
import threading
from typing import Dict, Union, Any
import random
import sqlite3
from func_timeout import func_set_timeout


from src.models.api_llm.models import get_llm_chain, async_llm_chain_call , call_llm_chain
from src.models.api_llm.prompts import get_prompt
from src.models.api_llm.parsers import get_parser
from src.utils.state import GraphState
from src.utils.tool import Tool

class ReWrite(Tool):
    """
    Tool for extracting keywords from the question and hint.
    """

    def __init__(self, template_name: str = None, engine_config: str = None, parser_name: str = None, databases_dir: str = None):
        super().__init__()

        self.template_name = template_name  # rewrite
        self.engine_config = engine_config
        self.parser_name = parser_name  # markdown
        self.databases_dir = databases_dir  # "/mnt/tampm/EmbeddingTaskAmbiText2SQL/AmbiVal_dataset/database_synonym_newques"

    @func_set_timeout(10)
    def execute_sql(self, db_path: str, sql: str, fetch: Union[str, int] = "all") -> Any:
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

    def _compare_sqls_outcomes(self, db_path: str, predicted_sql: str, ground_truth_sql: str) -> int:
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
            ground_truth_res = self.execute_sql(db_path, ground_truth_sql)
            predicted_res = self.execute_sql(db_path, predicted_sql)
        except:
            return 0, [], []
        return int(set(predicted_res) == set(ground_truth_res)), predicted_res, ground_truth_res

    def _run(self, state: GraphState):
        print("re_write run ReWrite")
        SCHEMA_STR = state["item"].schema_text_with_content
        existing_sql_query = ""
        equivalent_set = {}
        for previous_step in state["pipeline_log"]:
            if "relational_classification" in previous_step["step"]:
                equivalent_set = previous_step.get("equivalent_set", {})
            if previous_step["step"] == "initial_sql":
                existing_sql_query = previous_step["response"]['SQL']
                # break
            elif previous_step["step"] == "gen_ambiqt_sql":
                queries_list = previous_step["queries_list"]['queries']
                existing_sql_query = queries_list[0]
                # break
        match_content_str = ""
        for val in equivalent_set:
            match_content_str += f"'{val}' : {equivalent_set[val]}\n"
        request_kwargs = {
            "SCHEMA_STR": SCHEMA_STR,
            "QUESTION_TEXT": state["item"].question,
            "MATCHED_CONTENTS": match_content_str,
            "EXISTING_SQL_QUERY" : existing_sql_query,
        }
        start_time = time.time()
        response,response_ori, num_input_tokens, num_output_tokens = call_llm_chain(  # Chỉ chạy 1 thread
            prompt=get_prompt(template_name=self.template_name),
            engine=get_llm_chain(**self.engine_config),
            parser=get_parser(self.parser_name),
            request_kwargs=request_kwargs,
            step=self.tool_name,
            log_file_lock=threading.Lock(),
            time_sleep=self.engine_config.get("time_sleep", 0),
        )
        # print(response)
        # import pdb; pdb.set_trace()
        end_time = time.time()
        run_time = end_time - start_time
        result, predicted_res, ground_truth_res = 0, [], []
        # db_id = state["item"].db_id
        # if os.path.exists(os.path.join(self.databases_dir, db_id, db_id + ".sqlite")):
        #     db_path = os.path.join(self.databases_dir, db_id, db_id + ".sqlite")
        # else:
        #     db_path = os.path.join(self.databases_dir, "wikisql.sqlite")
        # result, predicted_res, ground_truth_res = self._compare_sqls_outcomes(
        #     db_path=db_path,
        #     predicted_sql=response['SQL'],
        #     ground_truth_sql=state["item"].amb_query
        # )
        # print(f"Rewrite result: {result}, predicted_res: {predicted_res}, ground_truth_res: {ground_truth_res}")
        if state["pipeline_log"] is None:
            state["pipeline_log"] = []
        state["pipeline_log"].append({
            "step": self.tool_name,
            "request": request_kwargs,
            "response": response,
            "response_ori": response_ori,
            "ambival_eval": {
                "result": result,
                "predicted_res": predicted_res,
                "ground_truth_res": ground_truth_res
            },
            "num_input_tokens": num_input_tokens,
            "num_output_tokens": num_output_tokens,
            "run_time": run_time
        })
        # import pdb; pdb.set_trace()

