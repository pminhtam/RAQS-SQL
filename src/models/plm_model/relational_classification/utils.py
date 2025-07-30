"""
Code inference for relational values.

"""


import torch
from enum import Enum

class RelClassLabel(int, Enum):
    OTHER = 0  # Others

    SYM = 1
    SUP = 2
    SUB = 3
def pre_process_input_sample(item_rel, tokenizer):
    db_id_re, table_name_re, column_name_re, value_re, v_re = item_rel
    tokens = ["[CLS]"]
    """
    Thêm các context nè
    """
    max_seq_length = 128
    tokens += tokenizer.tokenize("In Database : " + db_id_re)
    tokens += tokenizer.tokenize(", table name : " + table_name_re)
    tokens += tokenizer.tokenize(", column name : " + column_name_re)
    tokens.append("[SEP]")
    # Tokenize DB entity sau
    item_pos = []
    item_pos.append(len(tokens))
    tokens += tokenizer.tokenize(str(value_re))
    tokens.append("[SEP]")
    item_pos.append(len(tokens))

    # tokens = tokens_ori.copy()
    tokens += tokenizer.tokenize(str(v_re))
    tokens.append("[SEP]")
    item_pos.append(len(tokens))

    input_id = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_id)
    segment_id = [0] * len(tokens)
    padding = [0] * (max_seq_length - len(input_id))

    # padding để đủ 512 token
    input_id += padding
    input_mask += padding
    segment_id += padding
    assert len(input_id) == len(input_mask) == len(segment_id) == max_seq_length
    input_id, input_mask, item_pos = torch.tensor(input_id), torch.tensor(input_mask), torch.tensor(item_pos)
    input_id, input_mask, item_pos = input_id.unsqueeze(0), input_mask.unsqueeze(0), item_pos.unsqueeze(0)
    return input_id, input_mask, item_pos


def infer_one_batch_relval(model, input_id, input_mask,  item_pos, device):
    """
    Infer  model with 1 sample
    Output là classification model

    """
    # model.eval()
    input_id, input_mask, item_pos = input_id.to(device), input_mask.to(device), item_pos.to(device)
    cls_output = model(input_id, input_mask, item_pos)
    cls = cls_output[0]
    return cls

