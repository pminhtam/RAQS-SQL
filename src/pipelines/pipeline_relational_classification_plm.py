"""

Generate all probable queries for ambiqt dataset using LLM.

"""

from typing import Dict
import time
import re
import json
import threading
import torch
from transformers import AutoTokenizer
from enum import Enum

from src.utils.state import GraphState
from src.utils.tool import Tool
from src.models.plm_model.relational_classification import RelClassifier, pre_process_input_sample, infer_one_batch_relval, RelClassLabel


class RelationalClassificationPLM(Tool):
    """
    Tool for generate all probable SQL queries for a given question and database schema.
    """

    def __init__(self, model_path: str = None, device: str = None):
        super().__init__()

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.model_path = model_path

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, do_lower_case=True)
        self.model = RelClassifier(self.model_path)
        self.model.load_state_dict(torch.load(self.model_path + "/pytorch_model.pt"))
        self.model.eval()
        self.model.to(self.device)

    def _run(self, state: GraphState):
        print("relational_classification_plm run RealationalClassificationPLM tool")
        val_linking_emb = {}
        for previous_step in state["pipeline_log"]:
            if "embedding_search" in previous_step["step"]:
                val_linking_emb = previous_step["val_linking_emb"]
                break
        db_id = state["item"].db_id
        equivalent_set = {}
        # print(f"val_linking_emb: {val_linking_emb}")
        start_time = time.time()
        for value in val_linking_emb:
            table_name = val_linking_emb[value]["table"]
            column_name = val_linking_emb[value]["column"]
            equivalent_set[value] = []
            for v in val_linking_emb[value]["candidate"]:
                item_rel = [db_id, table_name, column_name, value, v]
                try:
                    input_id, input_mask, item_pos = pre_process_input_sample(item_rel, self.tokenizer)
                    cls_res = infer_one_batch_relval(self.model, input_id, input_mask, item_pos, self.device)
                    cls_out = torch.argmax(cls_res, dim=-1)
                    if cls_out.item() == RelClassLabel.SYM or cls_out.item() == RelClassLabel.SUB:
                        equivalent_set[value].append(v)
                except Exception as e:
                    continue
                # print(f"cls_res : {cls_res} , cls_out: {cls_out}, item_rel: {item_rel}")
        end_time = time.time()
        run_time = end_time - start_time
        # print(f"equivalent_set: {equivalent_set}")
        # import pdb; pdb.set_trace()
        state["pipeline_log"].append({
            "step": self.tool_name,
            "equivalent_set": equivalent_set,
            "run_time": run_time
        })
        # import pdb; pdb.set_trace()

