"""
Code inference for column linking  task

"""


import torch
from enum import Enum
import numpy as np
from itertools import combinations

def pre_process_input_sample(question, db_schema, tokenizer):
    max_seq_length = 512  # Độ dài tối đa
    # max_seq_length = 1024  # Độ dài tối đa
    input_seq = []
    db_schema_list = []
    input_seq.append(question)
    # input_seq.append((0, '[*]'))
    # db_schema_list.append('[*]')
    # ti = 1
    ti = -1
    # input_seq : sẽ tổ chức theo kiểu
    # [question,
    # table_name1, table_name1.col_name11, table_name1.col_name12,...,
    # table_name2, table_name2.col_name21, table_name2.col_name22,...,
    # ...]

    col_num = 0  # Không dùng [*] nữa
    # col_num = 1             # Vì có phần tử '[*]'
    db_schema_keys = list(db_schema.keys())
    for table_name in db_schema_keys:
        col_num += len(db_schema[table_name])
        for col_name in db_schema[table_name]:
            ti += 1
            input_seq.append((ti, table_name.lower() + '.' + col_name.lower()))
            db_schema_list.append(table_name.lower() + '.' + col_name.lower())
    # Bỏ đi phần tử '[none]'
    # ti += 1
    # input_seq.append((ti, '[none]'))  # Cái này để link với value mà không biết column nào
    # db_schema_list.append('[none]')  # Cái này để link với value mà không biết column nào

    # Tokenize câu hỏi trước
    tokens = ["[CLS]"]  # Token đầu tiên là [CLS]
    orig_to_tok_map = {}
    que_toks = input_seq[0].split("#")  # Xét từng part phân cách bằng '#'
    question_token_interest_list = que_toks[1::2]    # chỉ lấy từ ở trong cặp # #
    for i, (token) in enumerate((que_toks)):
        orig_to_tok_map[i] = len(tokens)  # orig_to_tok_map[0] = 1 vì tokens[0] là [CLS]
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tokens.append(sub_token)
    orig_to_tok_map[len(que_toks)] = len(tokens)

    tokens.append("[SEP]")
    segment_id = [0] * len(tokens)
    # Tokenize DB entity sau
    item_pos_map = {}
    for (seg_id, seg) in input_seq[1:]:
        item_pos_map[seg_id] = []
        seg_toks = seg.split()  # Vì dbschema chỉ có 2 từ --> cái này chỉ là 1 token thôi
        for st in seg_toks:
            item_pos_map[seg_id].append(len(tokens))
            st_tokens = tokenizer.tokenize(st)
            tokens += st_tokens
        tokens.append("[SEP]")
    item_pos_map[len(input_seq[1:])] = [len(tokens)]
    segment_id += [1] * (len(tokens) - len(segment_id))

    if len(segment_id) != len(tokens):
        print(f"Error: segment_id length {len(segment_id)} does not match tokens length {len(tokens)}")
        return
    # assert len(segment_id) == len(tokens)
    if len(tokens) > max_seq_length:  # hơn 512 token --> lỗi . Thêm table thì bị lỗi cmnl
        print(f"Error: tokens length {len(tokens)} exceeds max_seq_length {max_seq_length}")
        return  # Bỏ qua câu hỏi quá dài. Cần xem lại dữ liệu. Load tiếp câu khác
        # return None
    input_id = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_id)
    padding = [0] * (max_seq_length - len(input_id))
    # padding để đủ 512 token
    input_id += padding
    input_mask += padding
    segment_id += padding
    assert len(input_id) == len(input_mask) == len(segment_id) == max_seq_length

    item_pos = []
    for i in range(len(item_pos_map)):
        pos = item_pos_map[i]
        item_pos.append(pos)
    # Các thông tin cho phần Input
    que_tok_pos = list(orig_to_tok_map.values())
    que_tok_num = len(orig_to_tok_map) - 1

    input_id = torch.tensor(input_id)
    input_mask = torch.tensor(input_mask)
    segment_id = torch.tensor(segment_id)
    que_tok_pos = torch.tensor(que_tok_pos + [0] * (110 - len(que_tok_pos)))
    item_pos = torch.tensor(item_pos)
    que_tok_num = torch.tensor(que_tok_num)
    col_num = torch.tensor(col_num)
    idx_que_have_link = [i for i in range(1,len(que_toks),2)]  # Danh sách các que_tok có khả năng linking
    idx_que_have_link = torch.tensor(idx_que_have_link)

    input_id, input_mask, segment_id, que_tok_pos, item_pos, que_tok_num, idx_que_have_link, col_num = input_id.unsqueeze(0), input_mask.unsqueeze(0), segment_id.unsqueeze(0), que_tok_pos.unsqueeze(0), item_pos.unsqueeze(0), que_tok_num.unsqueeze(0), idx_que_have_link.unsqueeze(0), col_num.unsqueeze(0)
    return input_id, input_mask, segment_id, que_tok_pos, item_pos, que_tok_num,idx_que_have_link, col_num


def infer_one_batch_column_linking(model, input_id, input_mask, segment_id,que_tok_pos, item_pos,que_tok_num , idx_que_have_link, col_num, device):
    """
    Infer SLSQL val model with 1 sample
    Output là xác suất bị ambiguous value

    """
    model.eval()
    sigmoid_max_fn = torch.nn.Sigmoid()
    input_id, input_mask, segment_id, que_tok_pos, item_pos, que_tok_num, idx_que_have_link, col_num = input_id.to(device), input_mask.to(device), segment_id.to(device), que_tok_pos.to(device), item_pos.to(device), que_tok_num.to(device), idx_que_have_link.to(device), col_num.to(device)
    cls_output = model(input_id, input_mask, segment_id, que_tok_pos, item_pos, que_tok_num, idx_que_have_link, col_num )
    cls = sigmoid_max_fn(cls_output[0])  # Apply sigmoid to the output
    # cls = cls_output
    # cls_sigmoid = torch.sigmoid(cls)[:, 0] > 0.5
    return cls, []
def infer_one_batch_column_linking_corr(model, input_id, input_mask, segment_id,que_tok_pos, item_pos,que_tok_num , idx_que_have_link, col_num, device):
    """
    Infer SLSQL val model with 1 sample
    Output là xác suất bị ambiguous value

    """
    model.eval()
    sigmoid_max_fn = torch.nn.Sigmoid()
    input_id, input_mask, segment_id, que_tok_pos, item_pos, que_tok_num, idx_que_have_link, col_num = input_id.to(device), input_mask.to(device), segment_id.to(device), que_tok_pos.to(device), item_pos.to(device), que_tok_num.to(device), idx_que_have_link.to(device), col_num.to(device)
    cls_output, col_corr = model(input_id, input_mask, segment_id, que_tok_pos, item_pos, que_tok_num, idx_que_have_link, col_num )
    cls = sigmoid_max_fn(cls_output[0])  # Apply sigmoid to the output
    col_corr = sigmoid_max_fn(col_corr[0])  # Apply sigmoid to the output
    # cls = cls_output
    # cls_sigmoid = torch.sigmoid(cls)[:, 0] > 0.5
    return cls, col_corr
def post_process_input_sample(question, db_schema, cls_output,col_corr, threshold=0.5):
    """
    From the cls_output, we can get the linking results between question and db_schema.
    :param question:
    :param db_schema:
    :param cls_output:
    :return:
    """
    db_schema_list = []
    for table in db_schema:
        for col in db_schema[table]:
            db_schema_list.append(table.lower() + '.' + col.lower())
    question_token_interest_list = question.split("#")[1::2]  # Chỉ lấy từ ở trong cặp # #
    schema_linking = {}
    # import pdb; pdb.set_trace()
    for i in range(len(cls_output)):
        schema_linking_item = []
        for j in range(len(cls_output[i])):
            if cls_output[i][j] > threshold:
                schema_linking_item.append(db_schema_list[j])
        schema_linking[question_token_interest_list[i]] = schema_linking_item
    return schema_linking

def is_highly_correlated(subset, corr_matrix, threshold):
    """Check if all pairs in subset have correlation above threshold."""
    for i, j in combinations(subset, 2):
        if abs(corr_matrix[i][j]) <= threshold:
            return False
    return True


def find_max_correlated_set(indices, corr_matrix, threshold):
    """Find the largest subset where all pairs have high correlation."""
    max_set = []
    max_size = 0

    # Check all possible subsets (from largest to smallest)
    # for r in range(len(indices), 0, -1):
    for r in range(len(indices), len(indices)-2, -1):
        for subset in combinations(indices, r):
            if is_highly_correlated(subset, corr_matrix, threshold):
                if len(subset) > max_size:
                    max_set = list(subset)
                    max_size = len(subset)
                # Return early if we found a set of maximum possible size
                if max_size == len(indices):
                    return max_set
    return max_set
def post_process_input_sample_corr(question, db_schema, cls_output,col_corr, threshold=0.5):
    """
    From the cls_output, we can get the linking results between question and db_schema.
    :param question:
    :param db_schema:
    :param cls_output:
    :return:
    """
    db_schema_list = []
    for table in db_schema:
        for col in db_schema[table]:
            db_schema_list.append(table.lower() + '.' + col.lower())
    question_token_interest_list = question.split("#")[1::2]  # Chỉ lấy từ ở trong cặp # #
    schema_linking = {}
    schema_linking_idx = {}
    # import pdb; pdb.set_trace()
    for i in range(len(cls_output)):
        # schema_linking_item = []
        schema_linking_item_idx = []
        for j in range(len(cls_output[i])):
            if cls_output[i][j] > threshold:
                # schema_linking_item.append(db_schema_list[j])
                schema_linking_item_idx.append(j)
        # schema_linking[question_token_interest_list[i]] = schema_linking_item
        schema_linking_idx[question_token_interest_list[i]] = schema_linking_item_idx
    for question_token in schema_linking_idx:
        schema_linking_item = []
        schema_linking_item_idx_high_corr = find_max_correlated_set(schema_linking_idx[question_token], col_corr, threshold=0.5)
        for idx in schema_linking_item_idx_high_corr:
            schema_linking_item.append(db_schema_list[idx])
        schema_linking[question_token] = schema_linking_item
    # import pdb; pdb.set_trace()
    return schema_linking