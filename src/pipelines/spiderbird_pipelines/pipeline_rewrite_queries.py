"""

Rewrite error queries using LLMs.

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

class ReWriteSpiderbird(Tool):
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
    def make_str_response(self, sql_pred_response):
        """
        Convert SQL response to string format.
        """
        unique_result = set(sql_pred_response)
        if len(str(sql_pred_response)) > 120:
            # If the result is too long, return description of result
            n_duplicated_results = len(sql_pred_response) - len(unique_result)
            # return f"The result contains {len(sql_pred_response)} rows with {n_duplicated_results} duplicated rows. The first 3 rows are {sql_pred_response[:3]}."
            return f"The result contains {len(sql_pred_response)} rows with {n_duplicated_results} duplicated rows. The output rows are {str(sql_pred_response[:3])[:50]}...."
        else:
            return str(sql_pred_response)
    def _run(self, state: GraphState):
        print("re_write_spiderbird run ReWriteSpiderbird")
        SCHEMA_STR = state["item"].schema_text_with_content
        existing_sql_query = []
        equivalent_set = {}
        for previous_step in state["pipeline_log"]:
            if "relational_classification" in previous_step["step"]:
                equivalent_set = previous_step.get("equivalent_set", {})
            if "sql_preds" in previous_step:
                existing_sql_query.extend(previous_step["sql_preds"])
                # print(f"sql_queries : {sql_queries}")
        match_content_str = ""
        for val in equivalent_set:
            match_content_str += f"'{val}' : {equivalent_set[val]}\n"
        db_id = state["item"].db_id
        if os.path.exists(os.path.join(self.databases_dir, db_id, db_id + ".sqlite")):
            db_path = os.path.join(self.databases_dir, db_id, db_id + ".sqlite")
        else:
            raise Exception(f"Database {db_id} not found in {self.databases_dir}")
        start_time = time.time()
        num_input_tokens = 0
        num_output_tokens = 0
        sql_preds = []
        rewrite_prompt_infor = []
        for sql_query_item in existing_sql_query:       # ReWrite từng câu SQL
            # if "Error executing SQL:" not in sql_query_item['sql_response']:    # Câu không có lỗi thì không cần re-write
            if "Error executing SQL:" not in sql_query_item['sql_response'] and sql_query_item["len_sql_response"] > 0:    # Câu không có lỗi thì không cần re-write
                """
                sql_query_item["len_sql_response"] == 0 : tức là câu SQL trả về rỗng, có thể là do value bị sai
                File output có tên *_rewriteEmpty.jsonl
                """
                sql_preds.append(sql_query_item)
            else:       # Chỉ re-write những câu SQL có lỗi
                if "Error executing SQL:" not in sql_query_item['sql_response'] and sql_query_item["len_sql_response"] == 0:
                    """
                    Response rỗng vẫn cho vào selector
                    """
                    sql_preds.append(sql_query_item)
                existing_sql_query_str = self.existing_sql_query_txt_format.format(sql_query=sql_query_item['sql'], sql_res=sql_query_item['sql_response'])

                request_kwargs = {
                    "SCHEMA_STR": SCHEMA_STR,
                    "QUESTION_TEXT": state["item"].question,
                    "MATCHED_CONTENTS": match_content_str,
                    "EXISTING_SQL_QUERY" : existing_sql_query_str,
                }

                response,response_ori, num_input_tokens_i, num_output_tokens_i = call_llm_chain(  # Chỉ chạy 1 thread
                    prompt=get_prompt(template_name=self.template_name),
                    engine=get_llm_chain(**self.engine_config),
                    parser=get_parser(self.parser_name),
                    request_kwargs=request_kwargs,
                    step=self.tool_name,
                    log_file_lock=threading.Lock(),
                    time_sleep=self.engine_config.get("time_sleep", 0),
                )
                num_input_tokens += num_input_tokens_i
                num_output_tokens += num_output_tokens_i
                sql_pred = response.get("SQL", "")
                try:
                    sql_pred_response = self.execute_sql(db_path, sql_pred, fetch="all")
                    sql_pred_response_str = self.make_str_response(sql_pred_response)
                except Exception as e:
                    sql_pred_response_str = f"Error executing SQL: {e}"
                    print(f"Error executing SQL: {e}\nSQL: {sql_pred}")
                except func_timeout.exceptions.FunctionTimedOut as e:
                    sql_pred_response_str = f"Error executing SQL: Error timeout {e}"
                    print(f"Error executing SQL: Error timeout {e}")
                    # import pdb; pdb.set_trace()
                sql_preds.append({
                    "sql": sql_pred,
                    "sql_response": sql_pred_response_str
                })
                rewrite_prompt_infor.append({
                    "request_kwargs" : request_kwargs,
                    "response": response,
                    "response_ori": response_ori,
                })
        # print(response)
        # import pdb; pdb.set_trace()
        end_time = time.time()
        run_time = end_time - start_time
        if state["pipeline_log"] is None:
            state["pipeline_log"] = []
        state["pipeline_log"].append({
            "step": self.tool_name,
            "num_input_tokens": num_input_tokens,
            "rewrite_prompt_infor" : rewrite_prompt_infor,
            "num_output_tokens": num_output_tokens,
            "run_time": run_time,
            "sql_preds" : sql_preds,
        })
        # import pdb; pdb.set_trace()

