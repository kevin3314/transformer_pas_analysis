import torch
import torch.nn as nn
from transformers import BertModel

from base import BaseModel


class RefinementLayer1(BaseModel):
    """前段の予測のスコアを重みパラメータではなく、素直に与えるモデル"""
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
        self.hidden_size = 256

        self.W_prd = nn.Linear(bert_hidden_size, self.hidden_size * self.num_case)
        self.U_arg = nn.Linear(bert_hidden_size, self.hidden_size * self.num_case)

        self.mid_layers = nn.ModuleList([nn.Linear(self.hidden_size + self.num_case, self.hidden_size)
                                         for _ in range(self.num_case)])
        self.output = nn.Linear(self.hidden_size, 1, bias=False)

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                base_score: torch.Tensor,      # (b, seq, case, seq)
                ) -> torch.Tensor:             # (b, seq, case, seq)
        batch_size, sequence_len = input_ids.size()
        # (b, seq, hid)
        sequence_output, _ = self.bert(input_ids,  attention_mask=attention_mask)

        h_p = self.W_prd(self.dropout(sequence_output)).view(batch_size, sequence_len, self.num_case, self.hidden_size)
        # (b, seq, case*hid) -> (b, seq, case, hid)
        h_a = self.U_arg(self.dropout(sequence_output)).view(batch_size, sequence_len, self.num_case, self.hidden_size)

        h_pa = h_p.unsqueeze(dim=2) + h_a.unsqueeze(dim=1)  # (b, seq, seq, case, hid)
        base_score = base_score.transpose(2, 3).contiguous().unsqueeze(dim=3).expand(-1, -1, -1, self.num_case, -1)
        h = torch.cat([h_pa, base_score], dim=4)  # (b, seq, seq case, hid+case)
        h = torch.tanh(self.dropout(h))
        h_mids = [layer(h[:, :, :, i, :]) for i, layer in enumerate(self.mid_layers)]  # [(b, seq, seq, hid)]
        h_mid = torch.stack(h_mids, dim=3)  # (b, seq, seq, case, hid)
        # -> (b, seq, seq, case, 1) -> (b, seq, seq, case)
        output = self.output(torch.tanh(self.dropout(h_mid))).squeeze(dim=4)

        return output.transpose(2, 3).contiguous()  # (b, seq, case, seq)


class RefinementLayer2(BaseModel):
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

        self.W_prd = nn.Linear(bert_hidden_size, self.hidden_size * self.num_case)
        self.U_arg = nn.Linear(bert_hidden_size, self.hidden_size * self.num_case)

        # self.mid_layer = nn.Linear(self.num_case * self.hidden_size * (self.num_case + 2), self.hidden_size)
        self.mid_layer = nn.Linear(self.hidden_size * (self.num_case + 2), self.hidden_size)
        self.output = nn.Linear(self.hidden_size, 1, bias=False)

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                base_score: torch.Tensor,      # (b, seq, case, seq)
                ) -> torch.Tensor:             # (b, seq, case, seq)
        batch_size, sequence_len = input_ids.size()
        # (b, seq, hid)
        sequence_output, _ = self.bert(input_ids,  attention_mask=attention_mask)
        # (b, seq, case*hid) -> (b, seq, case, hid)
        h_p = self.W_prd(self.dropout(sequence_output)).view(batch_size, sequence_len, self.num_case, self.hidden_size)
        # (b, seq, case*hid) -> (b, seq, case, hid)
        h_a = self.U_arg(self.dropout(sequence_output)).view(batch_size, sequence_len, self.num_case, self.hidden_size)

        h_a = h_a.unsqueeze(dim=1).expand(-1, sequence_len, -1, -1, -1)  # (b, seq, seq, case, hid)
        # (b, seq, case, seq, hid) * (b, seq, case, seq, 1) -> (b, seq, case, seq, hid) -> (b, seq, case, hid)
        weighted_sum = torch.sum(h_a.transpose(2, 3).contiguous() * base_score.unsqueeze(dim=4), dim=3)
        # (b, seq, 1, 1, case*hid)
        ref = weighted_sum.view(batch_size, sequence_len, 1, 1, self.num_case * self.hidden_size)
        ref = ref.expand(-1, -1, sequence_len, self.num_case, -1)  # (b, seq, seq, case, case*hid)

        prd_rep = h_p.unsqueeze(dim=2).expand(-1, -1, sequence_len, -1, -1)  # (b, seq, seq, case, hid)

        h = torch.cat([prd_rep, h_a, ref], dim=4)  # (b, seq, seq, case, hid + hid + hid*case)
        h_mid = self.mid_layer(torch.tanh(self.dropout(h)))  # (b, seq, seq, case, hid)
        output = self.output(torch.tanh(self.dropout(h_mid))).squeeze(dim=4)  # (b, seq, seq, case)

        return output.transpose(2, 3).contiguous()  # (b, seq, case, seq)


class RefinementLayer3(BaseModel):
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
        self.hidden_size = 256

        self.W_prd = nn.Linear(bert_hidden_size, self.hidden_size)
        self.U_arg = nn.Linear(bert_hidden_size, self.hidden_size * self.num_case)

        self.mid_layers = nn.ModuleList([nn.Linear(self.hidden_size * (self.num_case + 2), self.hidden_size)
                                         for _ in range(self.num_case)])
        self.output = nn.Linear(self.hidden_size, 1, bias=False)

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                base_score: torch.Tensor,      # (b, seq, case, seq)
                ) -> torch.Tensor:             # (b, seq, case, seq)
        batch_size, sequence_len = input_ids.size()
        # (b, seq, hid)
        sequence_output, _ = self.bert(input_ids,  attention_mask=attention_mask)

        h_p = self.W_prd(self.dropout(sequence_output)).view(batch_size, sequence_len, 1, 1, self.hidden_size)
        # (b, seq, case*hid) -> (b, seq, case, hid)
        h_a = self.U_arg(self.dropout(sequence_output)).view(batch_size, sequence_len, self.num_case, self.hidden_size)

        h_a = h_a.unsqueeze(dim=1).expand(-1, sequence_len, -1, -1, -1)  # (b, seq, seq, case, hid)
        # (b, seq, case, seq, hid) * (b, seq, case, seq, 1) -> (b, seq, case, seq, hid) -> (b, seq, case, hid)
        weighted_sum = torch.sum(h_a.transpose(2, 3).contiguous() * base_score.unsqueeze(dim=4), dim=3)
        # (b, seq, 1, 1, case*hid)
        ref = weighted_sum.view(batch_size, sequence_len, 1, 1, self.num_case * self.hidden_size)
        ref = ref.expand(-1, -1, sequence_len, self.num_case, -1)  # (b, seq, seq, case, case*hid)

        prd_rep = h_p.expand(-1, -1, sequence_len, self.num_case, -1)  # (b, seq, seq, case, hid)

        h = torch.cat([prd_rep, h_a, ref], dim=4)  # (b, seq, seq, case, hid + hid + hid*case)
        h = torch.tanh(self.dropout(h))
        h_mids = [layer(h[:, :, :, i, :]) for i, layer in enumerate(self.mid_layers)]  # [(b, seq, seq, hid)]
        h_mid = torch.stack(h_mids, dim=3)  # (b, seq, seq, case, hid)
        output = self.output(torch.tanh(self.dropout(h_mid))).squeeze(dim=4)  # (b, seq, seq, case)

        return output.transpose(2, 3).contiguous()  # (b, seq, case, seq)
