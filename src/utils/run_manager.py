import os
import json
import time
from pathlib import Path
from multiprocessing import Pool
from typing import List, Dict, Any, Tuple
from langgraph.graph import StateGraph

from .item import Item
from .workflow_builder import build_workflow , GraphState
class RunManager:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.items: List[Item] = []
        self.data_path = Path(self.config["data_path"])
        self.initialize_items()
    def cvt_schema_str(self, db_schema: dict) -> str:
        schema_str = ""
        for table in db_schema:
            schema_str += f"Table {table}, columns = [{', '.join(db_schema[table])}]\n"
        return schema_str
    def initialize_items(self):
        with open(self.data_path) as f:
            data_dict_list = json.load(f)
        for item in data_dict_list:
            # print(item)
            item_obj = Item(
                db_id=item["db_id"],
                question=item["question"] + item['evidence'],
                # question=item["question"],
                schema_text_wo_content=item['schema_text_wo_content'],
                schema_text_with_content=item['schema_text_with_content'],
                evidence=item.get("evidence", ""),
                ori_query=item.get("ori_query", ""),
                amb_query=item.get("amb_query", "")
            )
            self.items.append(item_obj)

    def run(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        with Pool(self.config["num_workers"]) as p:
            results = p.map(self._run_item, items)
        return results

    def _run_item(self, item: Item, output_path: str):
        work_flow = build_workflow(self.config)
        initial_state = GraphState(item = item)
        start_time = time.time()
        # try:
        for state in work_flow.stream(initial_state):  # Cái này là bắt đầu chạy nè. Nếu các node là class thì chạy method __call__ của class đó
            # print(state)
            continue
        # except Exception as e:
        #     print("Exeception run_manager.py 125 : ",e)
            # import pdb; pdb.set_trace()
        end_time = time.time()
        run_time = end_time - start_time
        # state_end = state["__end__"]
        # import pdb; pdb.set_trace()
        total_item_num_input_tokens, total_item_num_output_tokens = 0, 0
        # print(f"Run time ori: {run_time} seconds")
        state_end = state[list(state.keys())[-1]]
        for step in state_end["pipeline_log"]:
            # print(f"Step: {step['step']}, run_time: {step.get('run_time', 0)}, num_input_tokens: {step.get('num_input_tokens', 0)}, num_output_tokens: {step.get('num_output_tokens', 0)}")
            if "schema_linking_llmapi" in step["step"]:
                run_time -= step.get("run_time", 0)
            else:
                num_input_tokens = step.get("num_input_tokens", 0)
                num_output_tokens = step.get("num_output_tokens", 0)
                total_item_num_input_tokens += num_input_tokens
                total_item_num_output_tokens += num_output_tokens

        print(f"Run time: {run_time} seconds, total input tokens: {total_item_num_input_tokens}, total output tokens: {total_item_num_output_tokens}  at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        # import pdb; pdb.set_trace()
        # print(state_end["pipeline_log"])
        result = {
            "question" : state_end["item"].question,
            "ori_query" : state_end["item"].ori_query,
            "amb_query" : state_end["item"].amb_query,
            "db_id" : state_end["item"].db_id,
            "pipeline_log" : state_end["pipeline_log"],
        }
        # import pdb; pdb.set_trace()
        with open(output_path, "a+") as f:
            json.dump(result, f)
            # json.dump(str(state_end), f)
            f.write("\n")
        # import pdb; pdb.set_trace()
        return run_time, total_item_num_input_tokens, total_item_num_output_tokens


