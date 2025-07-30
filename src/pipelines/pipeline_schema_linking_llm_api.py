"""

Generate a dummy SQL.


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

class SchemaLinkingLLMAPI(Tool):
    """
    Tool for linking between question words and database schema using LLM API.
    """

    def __init__(self, template_name: str = None, engine_config: str = None, parser_name: str = None):
        super().__init__()

        self.template_name = template_name      # schema_linking
        self.engine_config = engine_config
        self.parser_name = parser_name          # none

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
            output = output.split("```json")[1].split("```")[0]
        elif "```" in output:
            output = "{" + output.rsplit("{")[-1]
            output = output.rsplit("}", 1)[0] + "}"
        try:
            output = json.loads(output)
        except:
            try:
                output = output.replace("'", "\"")
                output = json.loads(output)
            except:
                try:
                    output = json.loads(output.replace("\"s", "'s"))
                except:
                    try:
                        output = json.loads(output.replace("r\"s", "r's"))
                    except:
                        try:
                            output =  json.loads(output.replace("s\" ", "s' "))
                        except:
                            try:
                                lines = output.splitlines();
                                cleaned_lines = [line.split('//')[0] for line in lines if
                                                 not line.strip().startswith('//')];
                                cleaned_text = '\n'.join(cleaned_lines);
                                output = json.loads(cleaned_text)
                            except:
                                try:
                                    lines = output.splitlines();
                                    cleaned_lines = [line.split('#')[0] for line in lines if
                                                     not line.strip().startswith('#')];
                                    cleaned_text = '\n'.join(cleaned_lines);
                                    output = json.loads(cleaned_text)
                                except:
                                    # import pdb; pdb.set_trace();
                                    output = {}

        return output
    def _run(self, state: GraphState):
        print("schema_linking_llmapi  run Schema Linking LLM API")
        SCHEMA_STR = state["item"].schema_text_with_content
        predict_intend_list = []
        # import pdb; pdb.set_trace();
        for previous_step in state["pipeline_log"]:
            if "intend_detection" in previous_step["step"] :
                predict_intend_list = previous_step["predict_intend_list"]
                break
        # print("predict_intend_list: ", predict_intend_list)
        request_kwargs = {
            "SCHEMA_STR": SCHEMA_STR,
            "QUESTION_TEXT": state["item"].question,
            "LIST_WORD_TOKENS": str(predict_intend_list)
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
        schema_linking = self.parse_response_val(response)
        end_time = time.time()
        run_time = end_time - start_time
        # import pdb; pdb.set_trace()
        if state["pipeline_log"] is None:
            state["pipeline_log"] = []
        state["pipeline_log"].append({
            "step": self.tool_name,
            "request": request_kwargs,
            "response": response,
            "response_ori": response_ori,
            "schema_linking": schema_linking,
            "num_input_tokens": num_input_tokens,
            "num_output_tokens": num_output_tokens,
            "run_time": run_time
        })
        # import pdb; pdb.set_trace()

