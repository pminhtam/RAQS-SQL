"""

Selects the final query from list of queries generated.

"""
import os
import time
import threading
from typing import Dict, Union, Any
import random
import sqlite3
from func_timeout import func_set_timeout
import func_timeout

from src.models.api_llm.models import get_llm_chain, async_llm_chain_call , call_llm_chain
from src.models.api_llm.prompts import get_prompt
from src.models.api_llm.parsers import get_parser
from src.utils.state import GraphState
from src.utils.tool import Tool

class SelectFinal(Tool):
    """
    Tool for extracting keywords from the question and hint.
    """
    existing_sql_query_txt_format = "\n SQL query: ```sql\n{sql_query}``` \n This execution results: {sql_res}\n"
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
        except Exception as e:
            return 0, [], []
        except func_timeout.exceptions.FunctionTimedOut as e:
            print(f"Error executing SQL: Error timeout {e}")
            return 0, [], []
        return int(set(predicted_res) == set(ground_truth_res)), predicted_res, ground_truth_res

    def _run(self, state: GraphState):
        print("select_final run SelectFinal")
        SCHEMA_STR = state["item"].schema_text_with_content
        existing_sql_query = []
        for previous_step in state["pipeline_log"]:
            if previous_step["step"] == "re_write_spiderbird":
                existing_sql_query = previous_step["sql_preds"] # Chọn query từ bước re_write_spiderbird
        start_time = time.time()
        if len(existing_sql_query) == 1:
            sql_preds = existing_sql_query
            request_kwargs, response,response_ori, num_input_tokens, num_output_tokens = {}, {}, "", 0, 0
        else:
            existing_sql_query_str = ""
            for sql_query_item in existing_sql_query:       # ReWrite từng câu SQL
                existing_sql_query_str += self.existing_sql_query_txt_format.format(sql_query=sql_query_item['sql'], sql_res=sql_query_item['sql_response'])
            print(f"existing_sql_query_str : {existing_sql_query_str}")
            request_kwargs = {
                "SCHEMA_STR": SCHEMA_STR,
                "QUESTION_TEXT": state["item"].question,
                "EXISTING_SQL_QUERY" : existing_sql_query_str,
            }
            response,response_ori, num_input_tokens, num_output_tokens = call_llm_chain(  # Chỉ chạy 1 thread
                prompt=get_prompt(template_name=self.template_name),
                engine=get_llm_chain(**self.engine_config),
                parser=get_parser(self.parser_name),
                request_kwargs=request_kwargs,
                step=self.tool_name,
                log_file_lock=threading.Lock(),
                time_sleep=self.engine_config.get("time_sleep", 0),
            )
            sql_preds = [{
                "sql": response.get("SQL", ""),
            }]
            # print(response)
            # import pdb; pdb.set_trace()

        end_time = time.time()
        run_time = end_time - start_time
        print("sql_preds : ", sql_preds)
        result, predicted_res, ground_truth_res = 0, [], []
        db_id = state["item"].db_id
        if os.path.exists(os.path.join(self.databases_dir, db_id, db_id + ".sqlite")):
            db_path = os.path.join(self.databases_dir, db_id, db_id + ".sqlite")
        else:
            db_path = os.path.join(self.databases_dir, "wikisql.sqlite")
        result, predicted_res, ground_truth_res = self._compare_sqls_outcomes(
            db_path=db_path,
            predicted_sql=sql_preds[0]["sql"],
            ground_truth_sql=state["item"].amb_query
        )
        print(f"Rewrite result: {result}")
        if state["pipeline_log"] is None:
            state["pipeline_log"] = []
        state["pipeline_log"].append({
            "step": self.tool_name,
            "request": request_kwargs,
            "response": response,
            "response_ori": response_ori,
            "eval": {
                "result": result,
                "predicted_res": predicted_res[:1],
                "ground_truth_res": ground_truth_res[:1]
            },
            "num_input_tokens": num_input_tokens,
            "num_output_tokens": num_output_tokens,
            "run_time": run_time,
            "sql_preds" : sql_preds
        })
        # import pdb; pdb.set_trace()

