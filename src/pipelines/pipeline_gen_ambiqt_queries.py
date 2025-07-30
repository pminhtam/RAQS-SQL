"""

Generate all probable queries for ambiqt dataset using LLM.

"""

from typing import Dict
import time
import re
import json
import threading
from src.models.api_llm.models import get_llm_chain, async_llm_chain_call , call_llm_chain
from src.models.api_llm.prompts import get_prompt
from src.models.api_llm.parsers import get_parser
from src.utils.state import GraphState
from src.utils.tool import Tool

class GenAmbiqtSQL(Tool):
    """
    Tool for generate all probable SQL queries for a given question and database schema.
    """

    def __init__(self, template_name: str = None, engine_config: str = None, parser_name: str = None):
        super().__init__()

        self.template_name = template_name  # "gen_ambiqt_sql"
        self.engine_config = engine_config
        self.parser_name = parser_name      # "none"

    def parse_response_val(self, output):
        """
        Parse response from openai to get list of synonym values.

        :param response:
        :return:
        """
        if "```json" in output:
            output = output.split("```json")[1].split("```")[0]
        elif "```" in output:
            output = output.split("```")[1].split("```")[0]
        try:
            output = json.loads(output)
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

    def _run(self, state: GraphState):
        print("gen_ambiqt_sql  run gen AmbiQT queries ")
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
        response,response_ori,num_input_tokens, num_output_tokens = call_llm_chain(      # Chỉ chạy 1 thread
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
            "run_time": run_time
        })
        # import pdb; pdb.set_trace()

