"""

Generate all probable queries for dataset using LLM.
Using hint schema linking from previous steps.

"""

from typing import Dict
import time
import re
import json
import os
import random
from typing import Dict, Union, Any
import sqlite3
import func_timeout
from func_timeout import func_set_timeout
import threading
from src.models.api_llm.models import get_llm_chain, async_llm_chain_call , call_llm_chain
from src.models.api_llm.prompts import get_prompt
from src.models.api_llm.parsers import get_parser
from src.utils.state import GraphState
from src.utils.tool import Tool

class GenManySQL(Tool):
    """
    Tool for generate all probable SQL queries for a given question and database schema.
    """

    def __init__(self, template_name: str = None, engine_config: str = None, parser_name: str = None, databases_dir: str = ""):
        super().__init__()

        self.template_name = template_name  # "gen_ambiqt_sql"
        self.engine_config = engine_config
        self.parser_name = parser_name      # "none"
        self.databases_dir = databases_dir  #

    def parse_response_val(self, output):
        """
        Parse response from openai to get list of synonym values.

        :param response:
        :return:
        """
        if "</think>" in output:
            output = output.split("</think>")[-1]
            # output = json.loads(output)
        if "```json" in output:
            # output = output.split("```")[-2].split("```json")[-1]
            output = output.split("```json")[-1].split("```")[0]
            # json_pattern = re.compile(r'```json.*queries.*```', re.DOTALL | re.IGNORECASE)
            # json_match = json_pattern.findall(output)
            # output = json_match[-1]
            # output = json_match
            # output = output.split("```json")[1].split("```")[0]
        elif "```" in output:
            output = output.split("```")[1].split("```")[0]
        try:
            output = json.loads(output)
        except:
            try:
                corrected_output = re.sub(r',\s*]', ']', output)        # , ở cuối cùng của mảng --> lỗi
                output = json.loads(corrected_output)
            except:
                try:
                    output = json.loads(output.replace(",\n  },", "\n  },"))  # dấu , cuối cùng ở json
                except:
                    try:
                        output = json.loads(output.replace("\"s", "'s"))
                    except:
                        try:
                            output = json.loads(output.replace("r\"s", "r's"))
                        except:
                            try:
                                output = json.loads(output.replace("s\" ", "s' "))
                            except:
                                output = {"queries": ["SELECT * FROM table"]}  # default query if parsing fails
                                # output = output.replace("'", "\"")
                                # output = json.loads(output.replace("'\",", "'\""))
                                # import pdb; pdb.set_trace()
        return output

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
        print("gen_many_sql  run GenManySQL ")
        SCHEMA_STR = state["item"].schema_text_with_content
        hint_schema_linking = ""
        for previous_step in state["pipeline_log"]:
            if "schema_linking" in previous_step["step"]:
                hint_schema_linking = previous_step["schema_linking"]
                break
        request_kwargs = {
            "SCHEMA_STR": SCHEMA_STR,
            "QUESTION_TEXT": state["item"].question,
            "HINT": hint_schema_linking
        }
        start_time = time.time()
        response,response_ori, num_input_tokens, num_output_tokens = call_llm_chain(      # Chỉ chạy 1 thread
            prompt=get_prompt(template_name=self.template_name),
            engine=get_llm_chain(**self.engine_config),
            parser=get_parser(self.parser_name),
            request_kwargs=request_kwargs,
            step=self.tool_name,
            log_file_lock=threading.Lock(),
            time_sleep=self.engine_config.get("time_sleep", 0),
        )
        queries_list = self.parse_response_val(response)
        end_time = time.time()
        run_time = end_time - start_time
        # print(response)
        # import pdb; pdb.set_trace()
        """
        Lấy execute response của câu SQL
        """
        db_id = state["item"].db_id
        if os.path.exists(os.path.join(self.databases_dir, db_id, db_id + ".sqlite")):
            db_path = os.path.join(self.databases_dir, db_id, db_id + ".sqlite")
        else:
            db_path = os.path.join(self.databases_dir, "wikisql.sqlite")

        sql_preds = []
        sql_queries_pred = queries_list["queries"]
        print(f"Number of SQL queries predicted: {len(sql_queries_pred)}")
        len_sql_response = 0
        for sql_query_pred in sql_queries_pred:
            try:
                sql_pred_response = self.execute_sql(db_path, sql_query_pred, fetch="all")
                sql_pred_response_str = self.make_str_response(sql_pred_response)
                len_sql_response = len(sql_pred_response)
                print(f"SQL query: {sql_query_pred}\nResponse: {sql_pred_response_str} \n len_sql_response: {len_sql_response}")
            except Exception as e:
                sql_pred_response_str = f"Error executing SQL: {e}"
                # print(f"Error executing SQL: {e}")
            except func_timeout.exceptions.FunctionTimedOut as e:
                sql_pred_response_str = f"Error executing SQL: Error timeout {e}"
                # print(f"Error executing SQL: Error timeout {e}")
                # import pdb; pdb.set_trace()
            sql_preds.append({
                "sql": sql_query_pred,
                "sql_response": sql_pred_response_str,
                "len_sql_response" : len_sql_response
            })

        if state["pipeline_log"] is None:
            state["pipeline_log"] = []
        state["pipeline_log"].append({
            "step": self.tool_name,
            "request": request_kwargs,
            "response": response,
            "response_ori": response_ori,
            "queries_list": queries_list,
            "num_input_tokens": num_input_tokens,
            "num_output_tokens": num_output_tokens,
            "run_time": run_time,
            "sql_preds": sql_preds,  # Chứa thông tin về câu SQL và kết quả trả về khi execute câu SQL đó
        })
        # import pdb; pdb.set_trace()

