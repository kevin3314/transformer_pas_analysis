import torch
import torch.nn as nn
from transformers import BertModel

from base import BaseModel


class RefinementLayer(BaseModel):
    def __init__(self,
                 bert_model: str,
                 vocab_size: int,
                 dropout: float,
                 num_case: int,
                 coreference: bool,
                 ) -> None:
        super().__init__()

        self.bert: BertModel = BertModel.from_pretrained(bert_model)
        self.bert.resize_token_embeddings(vocab_size)
        self.dropout = nn.Dropout(dropout)

        self.num_case = num_case + int(coreference)
        bert_hidden_size = self.bert.config.hidden_size
        self.hidden_size = 128

        self.W_prd = nn.Linear(bert_hidden_size, self.hidden_size)
        self.U_arg = nn.Linear(bert_hidden_size, self.hidden_size * self.num_case)

        self.case_layer = nn.Linear(self.hidden_size + self.num_case * self.hidden_size,
                                    self.num_case * self.hidden_size)
        self.output = nn.Linear(self.hidden_size, 1, bias=False)

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                ng_token_mask: torch.Tensor,   # (b, seq, case, seq)
                base_score: torch.Tensor,      # (b, seq, case, seq)
                ) -> torch.Tensor:             # (b, seq, case, seq)
        batch_size, sequence_len = input_ids.size()
        # (b, seq, hid)
        sequence_output, _ = self.bert(input_ids,  attention_mask=attention_mask)
        h_p = self.W_prd(self.dropout(sequence_output))  # (b, seq, hid)
        h_a = self.U_arg(self.dropout(sequence_output))  # (b, seq, case*hid)

        h_a = h_a.view(batch_size, 1, sequence_len, self.num_case, self.hidden_size)  # (b, 1, seq, case, hid)
        h_a = h_a.expand(-1, sequence_len, -1, -1, -1)  # (b, seq, seq, case, hid)
        h_a = h_a.transpose(2, 3).contiguous()  # (b, seq, case, seq, hid)
        weighted_a = h_a * base_score.unsqueeze(dim=4)  # (b, seq, case, seq, hid)
        # -> (b, seq, seq, case, hid) -> (b, seq, seq, case*hid)
        arg_rep = weighted_a.transpose(2, 3).contiguous().view(batch_size, sequence_len, sequence_len, -1)

        pred_rep = h_p.unsqueeze(dim=2).expand(-1, -1, sequence_len, -1)  # (b, seq, seq, hid)

        pas_rep = torch.cat([pred_rep, arg_rep], dim=3)  # (b, seq, seq, hid + case*hid)
        h_case = self.case_layer(torch.tanh(self.dropout(pas_rep)))  # (b, seq, seq, case*hid)
        # (b, seq, seq, case, hid)
        h_case = h_case.view(batch_size, sequence_len, sequence_len, self.num_case, self.hidden_size)
        output = self.output(torch.tanh(self.dropout(h_case))).squeeze(dim=4)  # (b, seq, seq, case)

        output = output.transpose(2, 3).contiguous()  # (b, seq, case, seq)

        extended_attention_mask = attention_mask.view(batch_size, 1, 1, sequence_len)  # (b, 1, 1, seq)
        mask = extended_attention_mask & ng_token_mask  # (b, seq, case, seq)

        output += (~mask).float() * -1024.0  # (b, seq, case, seq)

        return output  # (b, seq, case, seq)
