"""

Generate a dummy SQL.


"""

from typing import Dict
import time
import re
import json
import threading
import torch
from transformers import AutoTokenizer

from src.utils.state import GraphState
from src.utils.tool import Tool
from src.models.plm_model.schema_linking import SchemaLinking, pre_process_input_sample, infer_one_batch_column_linking_corr, post_process_input_sample_corr

class SchemaLinkingPLM(Tool):
    """
    Tool for linking between question words and database schema using LLM API.
    """

    def __init__(self, model_path: str = None, device: str = None, schema_dicts_path: str = None):
        super().__init__()
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, do_lower_case=True)
        self.model = SchemaLinking(self.model_path)
        self.model.load_state_dict(torch.load(self.model_path + "/pytorch_model.pt"))
        self.model.eval()
        self.model.to(self.device)
        self.schema_dict = json.load(open(schema_dicts_path, "r"))

    def infer_schemalinking(self, question: str, db_schema: Dict):
        schema_linking = {}
        try:
            input_id, input_mask, segment_id, que_tok_pos, item_pos, que_tok_num, idx_que_have_link, col_num = pre_process_input_sample(
                question, db_schema, self.tokenizer
            )
            cls_output, col_corr = infer_one_batch_column_linking_corr(
                self.model, input_id, input_mask, segment_id, que_tok_pos, item_pos, que_tok_num, idx_que_have_link, col_num,
                self.device
            )
            schema_linking = post_process_input_sample_corr(question, db_schema, cls_output, col_corr, 0.5)
        except:
            """
            Split db_schema into multiple parts if it is too long
            """
            # print("Schema is too long, splitting into two parts")
            db_schema_1 = {}
            db_schema_2 = {}
            len_dict = len(db_schema)
            total_num_col = 0
            for key in db_schema:
                total_num_col += len(db_schema[key])
            half_num_col = total_num_col // 2
            half_point = len_dict // 2
            numcol_1 = 0
            for i, (key, value) in enumerate(db_schema.items()):
                # if i < half_point:
                if numcol_1 < half_num_col:
                    db_schema_1[key] = value
                    numcol_1 += len(value)
                else:
                    db_schema_2[key] = value
            try:
                input_id_1, input_mask_1, segment_id_1, que_tok_pos_1, item_pos_1, que_tok_num_1, idx_que_have_link_1, col_num_1 = pre_process_input_sample(
                    question, db_schema_1, self.tokenizer
                )
                cls_output_1, col_corr_1 = infer_one_batch_column_linking_corr(
                    self.model, input_id_1, input_mask_1, segment_id_1, que_tok_pos_1, item_pos_1, que_tok_num_1, idx_que_have_link_1, col_num_1,
                    self.device
                )
                schema_linking_1 = post_process_input_sample_corr(question, db_schema_1, cls_output_1, col_corr_1,0.5)

                input_id_2, input_mask_2, segment_id_2, que_tok_pos_2, item_pos_2, que_tok_num_2, idx_que_have_link_2, col_num_2 = pre_process_input_sample(
                    question, db_schema_2, self.tokenizer
                )
                cls_output_2, col_corr_2 = infer_one_batch_column_linking_corr(
                    self.model, input_id_2, input_mask_2, segment_id_2, que_tok_pos_2, item_pos_2, que_tok_num_2, idx_que_have_link_2, col_num_2,
                    self.device
                )
                schema_linking_2 = post_process_input_sample_corr(question, db_schema_2, cls_output_2, col_corr_2, 0.5)
                """
                merge two schema_linking 
                """
                schema_linking = {}
                for key in schema_linking_1:
                    if key not in schema_linking:
                        schema_linking[key] = schema_linking_1[key]
                    else:
                        schema_linking[key].extend(schema_linking_1[key])
                for key in schema_linking_2:
                    if key not in schema_linking:
                        schema_linking[key] = schema_linking_2[key]
                    else:
                        schema_linking[key].extend(schema_linking_2[key])
            except Exception as e:
                print(f"Error processing question: {question}")
                print(f"Exception: {e}")
                # import pdb; pdb.set_trace()
                schema_linking = {}
                # continue
        return schema_linking
    def _run(self, state: GraphState):
        print("schema_linking_plm  run Schema Linking PLM")
        question = state["item"].question
        db_id = state["item"].db_id
        db_schema = self.schema_dict[db_id]
        predict_intend_list = []
        # import pdb; pdb.set_trace();
        for previous_step in state["pipeline_log"]:
            if "intend_detection" in previous_step["step"] :
                predict_intend_list = previous_step["predict_intend_list"]
                break
        start_time = time.time()
        """
        Mark intend words inside # characters in the question.
        """
        question_lower = question.lower()
        for intend_word in predict_intend_list:
            intend_word_lower = intend_word.lower()
            try:
                question_lower = re.sub(f'(?<=\\s){intend_word_lower}(?=[ ,.?!])', f" #{intend_word_lower}# ", question, count=1, flags=re.IGNORECASE)
            except:
                question_lower = re.sub(r'(?<=\\s){}(?=[ ,.?!])'.format(re.escape(intend_word_lower)), f" #{intend_word_lower}# ", question, count=1, flags=re.IGNORECASE)
            if f" #{intend_word_lower}# " not in question:
                if len(intend_word_lower.split()) > 2:
                    question = question.replace(intend_word_lower, f" #{intend_word_lower}# ")  #
        # try:
        #     input_id, input_mask, segment_id, que_tok_pos, item_pos, que_tok_num, idx_que_have_link, col_num = pre_process_input_sample(
        #         question_lower, db_schema, self.tokenizer
        #     )
        #     cls_output, col_corr = infer_one_batch_column_linking_corr(
        #         self.model, input_id, input_mask, segment_id, que_tok_pos, item_pos, que_tok_num, idx_que_have_link, col_num,
        #         self.device
        #     )
        #     schema_linking = post_process_input_sample_corr(question_lower, db_schema, cls_output, col_corr, 0.5)
        # except:
        #     schema_linking = {}
        schema_linking = self.infer_schemalinking(question_lower, db_schema)
        # import pdb; pdb.set_trace()

        end_time = time.time()
        run_time = end_time - start_time
        # import pdb; pdb.set_trace()
        if state["pipeline_log"] is None:
            state["pipeline_log"] = []
        state["pipeline_log"].append({
            "step": self.tool_name,
            "schema_linking": schema_linking,
            "run_time": run_time
        })
        # import pdb; pdb.set_trace()

