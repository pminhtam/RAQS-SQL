"""

Model for relation classification.
Input 2 entities
Output:
Relational between 2 entities:
- SYM
- SUP
- SUB
- OTHER


Dataloader file `dataloader/realtional_val.py`

"""

import torch
import torch.nn as nn

from transformers import AutoConfig, RobertaModel, XLMRobertaModel
from transformers import AutoModel


class RelClassifier(nn.Module):
    def __init__(
            self,
            model_name_or_path, num_class = 4
    ):
        super(RelClassifier, self).__init__()
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
        # self.bert.resize_token_embeddings(vocab_size)

        self.plm_hidden_size = self.bert.config.hidden_size
        self.item_gru = nn.GRU(input_size=self.plm_hidden_size, hidden_size=self.plm_hidden_size, batch_first=True)
        # column bi-lstm layer
        self.item_info_bilstm = nn.LSTM(
            input_size=self.plm_hidden_size,
            hidden_size=int(self.plm_hidden_size / 2),
            num_layers=2,
            dropout=0,
            bidirectional=True
        )
        self.val_cls_head_linear1 = nn.Linear(2*self.plm_hidden_size, 256)
        self.leakyrelu = nn.LeakyReLU()
        self.val_cls_head_linear2 = nn.Linear(256, num_class)
        self.softmax = nn.Softmax(dim=1)

        # dropout function, p=0.2 means randomly set 20% neurons to 0
        self.dropout = nn.Dropout(p=0.2)
    def forward(self, input_ids, input_mask, item_pos):
        outputs = self.bert(input_ids=input_ids, attention_mask=input_mask)
        sequence_output = outputs.last_hidden_state
        if self.training:
            sequence_output = self.dropout(sequence_output)
        cls_output = []
        for i in range(len(input_ids)):     # Lặp từng câu
            item_emb = []

            val1 = sequence_output[i, item_pos[i][0]:item_pos[i][1]]
            val2 = sequence_output[i, item_pos[i][1]:item_pos[i][2]]
            # import pdb; pdb.set_trace()
            # _, (hidden_state_1, c) = self.item_info_bilstm(val1.unsqueeze(0))
            # _, (hidden_state_2, c) = self.item_info_bilstm(val2.unsqueeze(0))
            # hidden_state_1 = hidden_state_1[-2:,-1 , :].contiguous().view(1, self.plm_hidden_size)
            # hidden_state_2 = hidden_state_2[-2:,-1 , :].contiguous().view(1, self.plm_hidden_size)
            _, hidden_state_1 = self.item_gru(val1.unsqueeze(0))
            hidden_state_1 = hidden_state_1.view(1, self.plm_hidden_size)
            _, hidden_state_2 = self.item_gru(val2.unsqueeze(0))
            hidden_state_2 = hidden_state_2.view(1, self.plm_hidden_size)
            item_emb.append(hidden_state_1)
            item_emb.append(hidden_state_2)
            # import pdb; pdb.set_trace()
            item_emb = torch.cat(item_emb, dim=1)
            # import pdb; pdb.set_trace()
            val_cls_out = self.val_cls_head_linear1(item_emb)
            val_cls_out = self.leakyrelu(val_cls_out)
            val_cls_out = self.val_cls_head_linear2(val_cls_out)
            val_cls_out = self.softmax(val_cls_out)
            cls_output.append(val_cls_out)

        # import pdb; pdb.set_trace()
        cls_output = torch.cat(cls_output, dim=0)
        """
        Dùng BCEWithLogitsLoss thì không cần dùng sigmoid ở đây. Hàm BCEWithLogitsLoss đã bao gồm sigmoid rồi.
        """
        return cls_output


if __name__ == "__main__":
    from pytorch_pretrained_bert import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = RelClassifier("bert-base-uncased", len(tokenizer.vocab))

    output = model()
    print(output.size())
    print(output)
    print("Done")


