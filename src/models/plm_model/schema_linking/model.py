"""
Original code from https://github.com/WING-NUS/slsql/blob/main/model/slsql.py

Input :
Question, Schema

Output :
Correlation matrix between token in question and token in schema

"""

"""

v1 = : linking with small token in question and db schema 
v0 with change bert to AutoModel
"""

from copy import deepcopy
from queue import Queue

import torch
import torch.nn.functional as F
from transformers import AutoConfig, RobertaModel, XLMRobertaModel
from transformers import AutoModel
from torch import nn
from torch.nn import Embedding

INF = 100000000
NINF = -INF


class SchemaLinking(nn.Module):
    def __init__(self, model_name_or_path) -> None:
        super(SchemaLinking, self).__init__()

        try:
            self.bert = AutoModel.from_pretrained(model_name_or_path)
        except:
            """
            Load từ model trained , không có weight của bert, chỉ có config 
            """
            config = AutoConfig.from_pretrained(model_name_or_path)
            """
            Model sẽ là random weight
            """
            self.bert = AutoModel.from_config(config)
        self.item_gru = nn.GRU(input_size=self.bert.config.hidden_size, hidden_size=self.bert.config.hidden_size, batch_first=True)
        try:
            self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        except:
            self.dropout = nn.Dropout(0.2)
        self.ct_link_scorer = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size * 2, self.bert.config.hidden_size),
            nn.Tanh(),
            nn.Linear(self.bert.config.hidden_size, 1)
        )
        self.cls_scoreer = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size),
            nn.Tanh(),
            nn.Linear(self.bert.config.hidden_size, 8),
            # nn.Softmax(dim=1)
        )
        """
        Không dùng Softmax ở đây vì sẽ dùng CrossEntropyLoss. Hàm CrossEntropyLoss đã bao gồm softmax rồi.
        https://discuss.pytorch.org/t/about-evaluating-and-understanding-the-output-of-crossentropyloss/73163/2
        """
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_ids, input_mask, segment_ids, que_tok_pos, item_pos,
                que_tok_num,idx_que_have_link, col_nums):

        outputs = self.bert(input_ids=input_ids, attention_mask=input_mask)
        sequence_output = outputs.last_hidden_state
        if self.training:
            sequence_output = self.dropout(sequence_output)
        all_ct_link_scores = []
        all_corr_link_scores = []
        for i in range(len(input_ids)):     # Lặp từng câu
            que_index = que_tok_pos[i][:que_tok_num[i] + 1]     ## Index những token là question , lấy thêm index của token sau nữa để biết last_token index
            que_emb = sequence_output[i, que_index]         ## Embedding của riêng question thôi
            cur_item_pos = item_pos[i]                      ## Ví trí của item (chính là table name/ column name) trong embedding
            col_num = col_nums[i].item()        # số lượng cột
            idx_que_have_link_item = idx_que_have_link[i]     # Danh sách các que_tok đã linking được với DB_schema. Ex [2, 7]
            item_emb = []
            que_have_link_emb = []
            """
            Lấy embedding của các word trong question đã linking được với DB_schema
            """
            for j in range(len(idx_que_have_link_item)):
                if idx_que_have_link_item[j] == -100:
                    break
                # import pdb; pdb.set_trace()
                first_pos_que = que_index[idx_que_have_link_item[j]]
                last_pos_que = que_index[idx_que_have_link_item[j] + 1] # lấy hết embedding ứng với vị trí đó
                hs = sequence_output[i, first_pos_que:last_pos_que]
                _, h = self.item_gru(hs.unsqueeze(0))
                h = h.squeeze()
                que_have_link_emb.append(h)
            """
            Lấy embedding của các column trong DB_schema
            """
            for j in range(col_num):
                first_pos = cur_item_pos[j, 0]
                # if first_pos == -100:
                #     continue
                # try:
                last_pos = cur_item_pos[j+1, 0] - 1 # Không lấy token [SEP] của column name
                if last_pos == -100:
                    # continue
                    last_pos = cur_item_pos[j, 1]+1
                    # import pdb; pdb.set_trace()
                # except:
                    # import pdb; pdb.set_trace()
                # print("first_pos, last_pos: ", first_pos, last_pos)
                hs = sequence_output[i, first_pos:last_pos]
                # try:
                _, h = self.item_gru(hs.unsqueeze(0))
                # except:
                #     import pdb;
                #     pdb.set_trace()
                h = h.squeeze()
                item_emb.append(h)

            item_emb = torch.stack(item_emb)    # size = (col_num + tbl_num + 1, hidden_size)
            que_have_link_emb = torch.stack(que_have_link_emb)    # size = (col_num + tbl_num + 1, hidden_size)
            # # import pdb; pdb.set_trace()
            # que_entity_embedding_attn_score = torch.matmul(que_have_link_emb, item_emb.T)
            # ct_link_scores = que_entity_embedding_attn_score.unsqueeze(-1)
            # ct_link_scores = self.sigmoid(que_entity_embedding_attn_score).unsqueeze(-1)      ## Tính score

            item_que_emb_wo_key = []
            for j, item in enumerate(item_emb):
                cur_emb_wo_key = []
                for que in que_have_link_emb:
                    cur_emb_wo_key.append(torch.cat([item, que]))
                cur_emb_wo_key = torch.stack(cur_emb_wo_key)
                item_que_emb_wo_key.append(cur_emb_wo_key)

            item_que_emb_wo_key = torch.stack(
                item_que_emb_wo_key)  ## Một cái correlation matrix giữa token trong question với token trong schema
            que_item_emb = item_que_emb_wo_key.transpose(0, 1)

            # ct_link_scores = self.sigmoid(self.ct_link_scorer(que_item_emb))      ## Tính score
            ct_link_scores = self.ct_link_scorer(que_item_emb)  ## Tính score
            #########################################
            col_col_emb_wo_key = []
            for j, item_1 in enumerate(item_emb):
                corr_emb_wo_key = []
                for item_2 in item_emb:
                    corr_emb_wo_key.append(torch.cat([item_1, item_2]))
                corr_emb_wo_key = torch.stack(corr_emb_wo_key)
                col_col_emb_wo_key.append(corr_emb_wo_key)

            col_col_emb_wo_key = torch.stack(
                col_col_emb_wo_key)  ## Một cái correlation matrix giữa token trong question với token trong schema
            col_col_item_emb = col_col_emb_wo_key.transpose(0, 1)

            # ct_link_scores = self.sigmoid(self.ct_link_scorer(que_item_emb))      ## Tính score
            corr_link_scores = self.ct_link_scorer(col_col_item_emb)  ## Tính correlation của các column
            # import pdb; pdb.set_trace()
            """
            Dùng BCEWithLogitsLoss thì không cần dùng sigmoid ở đây. Hàm BCEWithLogitsLoss đã bao gồm sigmoid rồi.
            """
            all_ct_link_scores.append(ct_link_scores)
            all_corr_link_scores.append(corr_link_scores)

        return all_ct_link_scores, all_corr_link_scores


