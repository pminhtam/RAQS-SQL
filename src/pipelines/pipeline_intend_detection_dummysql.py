"""

Generate a dummy SQL.


"""

from typing import Dict
import time
import re
import threading
import json
from collections import defaultdict
import sqlglot
from sqlglot import exp, parse_one
from sqlglot.optimizer.scope import build_scope
from sqlglot.optimizer.scope import find_all_in_scope
from sqlglot.optimizer.qualify import qualify

from src.models.api_llm.models import get_llm_chain, async_llm_chain_call , call_llm_chain
from src.models.api_llm.prompts import get_prompt
from src.models.api_llm.parsers import get_parser
from src.utils.state import GraphState
from src.utils.tool import Tool

class IntendDetectionDummySQL(Tool):
    """
    Tool for get all intend word from question using dummy SQL.
    """

    def __init__(self, template_name: str = None, engine_config: str = None, parser_name: str = None):
        super().__init__()

        self.template_name = template_name  # "intend_detection_dummysql"
        self.engine_config = engine_config
        self.parser_name = parser_name      # "none"

    def merge_adjacent_words_with_stopwords(self,sentence, word_list):
        # Define stopwords : of, the, to, a, and, in, is, that, for, on, with, as, of, with, by, at, all, each, ',
        # stopwords = {'of', 'the', 'to', 'a', 'and', 'in', 'is', 'that', 'for', 'on', 'with', 'as','by', 'at', 'all', 'each', "'"}
        stopwords = {'of', 'the', 'to', 'a', 'in', 'is', 'that', 'for', 'on', 'with', 'as', 'by', 'at', 'all', 'each',
                     "'"}
        # Split the sentence using regex to preserve quoted strings
        sentence_words = re.findall(r"'[^']*'|\w+|[^\w\s]", sentence)

        # Split each phrase in word_list using the same regex
        phrase_word_lists = [re.findall(r"'[^']*'|\w+|[^\w\s]", phrase) for phrase in word_list]

        # Find matches in the sentence
        matches = []
        for phrase_words in phrase_word_lists:
            n = len(phrase_words)
            for i in range(len(sentence_words) - n + 1):
                if sentence_words[i:i + n] == phrase_words:
                    # Store the match with start index, end index, and the phrase
                    matches.append((i, i + n - 1, ' '.join(phrase_words)))

        # Sort matches by start index
        matches.sort(key=lambda x: x[0])

        # Merge adjacent matches, considering stopwords
        if not matches:
            return []

        result = []
        current = matches[0]
        i = 1
        while i < len(matches):
            next_match = matches[i]
            # Check if the phrases are "adjacent" with only stopwords in between
            start_idx = current[1] + 1
            end_idx = next_match[0]
            between_words = sentence_words[start_idx:end_idx]
            # Check if all words between are stopwords
            has_comma = ',' in between_words
            if not has_comma and all(word.lower() in stopwords for word in between_words if word.isalnum()):
                # Merge the phrases, including the stopwords
                merged_phrase = current[2] + ' ' + ' '.join(between_words) + ' ' + next_match[2]
                current = (current[0], next_match[1], merged_phrase)
            else:
                result.append(current[2])
                current = next_match
            i += 1
        result.append(current[2])

        return result

    def parse_response(self,response, question):
        # response = response.replace('```', '').replace("\"", "'").replace("DISTINCT", '').replace("sql\n", '')
        try:
            if "```" in response:
                response = response.split("```")[1].split("```")[0]
                response = response.replace('```', '').replace("sql\n", '')
            ast = sqlglot.parse_one(response)
            qualify(ast)
            root = build_scope(ast)
            # if "country" in response.lower():
            #     import pdb; pdb.set_trace()
            column_list = []
            # column_list.extend([node.name for node in root.find_all(exp.Literal)])
            column_list.extend([node.name for node in root.find_all(exp.Column)])
            column_list.extend([node.name for node in root.find_all(exp.Table)])
            for word in column_list:
                if word.lower() in question.lower():
                    continue
                else:
                    column_list.remove(word)
            # column_list = merge_adjacent_words(question, column_list)
            # column_list.extend(self.merge_adjacent_words_with_stopwords(question, column_list))
            column_list = list(set(column_list))  # Remove duplicates
            column_list = self.merge_adjacent_words_with_stopwords(question, column_list)
        except:
            print(f"Error parsing IntendDetectionDummySQL response: {response}")
            column_list = []
        return column_list
    def _run(self, state: GraphState):
        print("intend_detection_dummy_sql   run Intend Detection DummySQL tool")
        request_kwargs = {
            "QUESTION_TEXT": state["item"].question,
        }
        start_time = time.time()
        # import pdb; pdb.set_trace()
        response,response_ori,num_input_tokens, num_output_tokens = call_llm_chain(      # Chỉ chạy 1 thread
            prompt=get_prompt(template_name=self.template_name),
            engine=get_llm_chain(**self.engine_config),
            parser=get_parser(self.parser_name),
            request_kwargs=request_kwargs,
            step=self.tool_name,
            log_file_lock=threading.Lock(),
            time_sleep=self.engine_config.get("time_sleep", 0),
        )
        predict_intend_list = self.parse_response(response, state["item"].question)
        end_time = time.time()
        run_time = end_time - start_time
        predict_intend_list = list(set(predict_intend_list))
        # import pdb; pdb.set_trace()
        if state["pipeline_log"] is None:
            state["pipeline_log"] = []
        state["pipeline_log"].append({
            "step": self.tool_name,
            "request": request_kwargs,
            "response": response,
            "response_ori": response_ori,
            "predict_intend_list" : predict_intend_list,
            "num_input_tokens": num_input_tokens,
            "num_output_tokens": num_output_tokens,
            "run_time": run_time
        })
        # print(predict_intend_list)
        # import pdb; pdb.set_trace()

