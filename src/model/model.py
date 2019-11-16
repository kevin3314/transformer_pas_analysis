import torch
import torch.nn as nn
from transformers import BertModel

from base import BaseModel


class BaselineModel(BaseModel):
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

        # head selection [Zhang+ 16]
        self.W_a = nn.Linear(bert_hidden_size, bert_hidden_size * self.num_case)
        self.U_a = nn.Linear(bert_hidden_size, bert_hidden_size * self.num_case)
        self.v_a = nn.Linear(bert_hidden_size, 1, bias=False)

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                ng_token_mask: torch.Tensor,   # (b, seq, case, seq)
                deps: torch.Tensor,            # (b, seq, seq)
                ) -> torch.Tensor:             # (b, seq, case, seq)
        batch_size, sequence_len = input_ids.size()
        # (b, seq, hid)
        sequence_output, _ = self.bert(input_ids,
                                       attention_mask=attention_mask)

        h_i = self.W_a(sequence_output)  # (b, seq, case*hid)
        h_j = self.U_a(sequence_output)  # (b, seq, case*hid)
        h_i = h_i.view(batch_size, sequence_len, self.num_case, -1)  # (b, seq, case, hid)
        h_j = h_j.view(batch_size, sequence_len, self.num_case, -1)  # (b, seq, case, hid)
        # (b, seq, seq, case, hid) -> (b, seq, seq, case, 1) -> (b, seq, seq, case)
        g_logits = self.v_a(torch.tanh(self.dropout(h_i.unsqueeze(1) + h_j.unsqueeze(2)))).squeeze(-1)

        g_logits = g_logits.transpose(2, 3).contiguous()  # (b, seq, case, seq)

        extended_attention_mask = attention_mask.view(batch_size, 1, 1, sequence_len)  # (b, 1, 1, seq)
        mask = extended_attention_mask & ng_token_mask  # (b, seq, case, seq)
        mask = mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility

        g_logits += (1.0 - mask) * -1024.0  # (b, seq, case, seq)

        return g_logits  # (b, seq, case, seq)


# TODO: introduce dropout
class DependencyModel(BaseModel):
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

        # head selection [Zhang+ 16]
        self.W_a = nn.Linear(bert_hidden_size, bert_hidden_size * self.num_case)
        self.U_a = nn.Linear(bert_hidden_size, bert_hidden_size * self.num_case)
        self.v_a = nn.Linear(bert_hidden_size, 1, bias=False)

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                ng_token_mask: torch.Tensor,   # (b, seq, seq)
                deps: torch.Tensor,            # (b, seq, seq)
                ) -> torch.Tensor:             # (b, seq, case, seq)
        # (b, seq, hid)
        sequence_output, _ = self.bert(input_ids,
                                       attention_mask=attention_mask)
        batch_size, sequence_len, hidden_dim = sequence_output.size()

        h_i = self.W_a(sequence_output)  # (b, seq, case*hid)
        h_j = self.U_a(sequence_output)  # (b, seq, case*hid)
        h_i = h_i.view(batch_size, sequence_len, self.num_case, -1)  # (b, seq, case, hid)
        h_j = h_j.view(batch_size, sequence_len, self.num_case, -1)  # (b, seq, case, hid)
        h = torch.tanh(h_i.unsqueeze(1) + h_j.unsqueeze(2))  # (b, seq, seq, case, hid)

        # make deps symmetric matrix
        deps = deps | deps.transpose(1, 2).contiguous()  # (b, seq, seq)
        # (b, seq, seq, case, 1)
        deps = deps.view(batch_size, sequence_len, sequence_len, 1, 1).expand(-1, -1, -1, self.num_case, 1).float()
        g_logits = self.v_a(torch.cat([h, deps], dim=4)).squeeze(-1)  # (b, seq, seq, case)

        g_logits = g_logits.transpose(2, 3).contiguous()  # (b, seq, case, seq)

        extended_attention_mask = attention_mask.view(batch_size, 1, 1, sequence_len)  # (b, 1, 1, seq)
        mask = extended_attention_mask & ng_token_mask  # (b, seq, case, seq)
        mask = mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility

        g_logits += (1.0 - mask) * -1024.0  # (b, seq, case, seq)

        return g_logits  # (b, seq, case, seq)


# TODO: introduce dropout
class LayerAttentionModel(BaseModel):
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

        self.layer_attn1 = nn.Linear(bert_hidden_size, 100)
        self.layer_attn2 = nn.Linear(100, 1)

        self.W_a = nn.Linear(bert_hidden_size, bert_hidden_size * self.num_case)
        self.U_a = nn.Linear(bert_hidden_size, bert_hidden_size * self.num_case)
        self.v_a = nn.Linear(bert_hidden_size + 1, 1, bias=False)

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                ng_token_mask: torch.Tensor,   # (b, seq, seq)
                deps: torch.Tensor,            # (b, seq, seq)
                ) -> torch.Tensor:             # (b, seq, case, seq)
        # [(b, seq, hid)]
        sequence_output, _ = self.bert(input_ids,
                                       attention_mask=attention_mask)
        sequence_output = torch.stack(sequence_output, dim=1)  # (b, l, seq, hid)
        batch_size, num_layer, sequence_len, hidden_dim = sequence_output.size()

        attn_mid = self.layer_attn1(sequence_output)  # (b, l, seq, 100)
        attn = self.layer_attn2(torch.tanh(attn_mid))  # (b, l, seq, 1)
        softmax_attn = torch.softmax(attn, dim=1)  # (b, l, seq, 1)
        weighted_output = (sequence_output * softmax_attn).sum(dim=1)  # (b, seq, hid)

        h_i = self.W_a(weighted_output)  # (b, seq, case*hid)
        h_j = self.U_a(weighted_output)  # (b, seq, case*hid)
        h_i = h_i.view(batch_size, sequence_len, self.num_case, -1)  # (b, seq, case, hid)
        h_j = h_j.view(batch_size, sequence_len, self.num_case, -1)  # (b, seq, case, hid)
        h = torch.tanh(h_i.unsqueeze(1) + h_j.unsqueeze(2))  # (b, seq, seq, case, hid)
        # (b, seq, seq, case, 1)
        deps = deps.view(batch_size, sequence_len, sequence_len, 1, 1).expand(-1, -1, -1, self.num_case, 1).float()
        g_logits = self.v_a(torch.cat([h, deps], dim=4)).squeeze(-1)  # (b, seq, seq, case)

        g_logits = g_logits.transpose(2, 3).contiguous()  # (b, seq, case, seq)

        extended_attention_mask = attention_mask.view(batch_size, 1, 1, sequence_len)  # (b, 1, 1, seq)
        mask = extended_attention_mask & ng_token_mask  # (b, seq, case, seq)
        mask = mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility

        g_logits += (1.0 - mask) * -1024.0  # (b, seq, case, seq)

        return g_logits  # (b, seq, case, seq)


# TODO: introduce dropout
class MultitaskDepModel(BaseModel):
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

        self.W_a = nn.Linear(bert_hidden_size, bert_hidden_size * self.num_case)
        self.U_a = nn.Linear(bert_hidden_size, bert_hidden_size * self.num_case)
        self.v_a = nn.Linear(bert_hidden_size + 1, 1, bias=False)

        self.W_dep = nn.Linear(bert_hidden_size, bert_hidden_size)
        self.U_dep = nn.Linear(bert_hidden_size, bert_hidden_size)
        self.v_dep = nn.Linear(bert_hidden_size, 1, bias=True)

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                ng_token_mask: torch.Tensor,   # (b, seq, seq)
                _: torch.Tensor,               # (b, seq, seq)
                ) -> torch.Tensor:             # (b, seq, case, seq)
        # (b, seq, hid)
        sequence_output, _ = self.bert(input_ids,
                                       attention_mask=attention_mask)
        batch_size, sequence_len, hidden_dim = sequence_output.size()

        # dependency parsing
        dep_i = self.W_dep(sequence_output)  # (b, seq, hid)
        dep_j = self.U_dep(sequence_output)  # (b, seq, hid)
        dep = self.v_dep(torch.tanh(dep_i.unsqueeze(1) + dep_j.unsqueeze(2)))  # (b, seq, seq, hid) -> (b, seq, seq, 1)

        # PAS analysis
        h_i = self.W_a(sequence_output)  # (b, seq, case*hid)
        h_j = self.U_a(sequence_output)  # (b, seq, case*hid)
        h_i = h_i.view(batch_size, sequence_len, self.num_case, -1)  # (b, seq, case, hid)
        h_j = h_j.view(batch_size, sequence_len, self.num_case, -1)  # (b, seq, case, hid)
        h = torch.tanh(h_i.unsqueeze(1) + h_j.unsqueeze(2))  # (b, seq, seq, case, hid)
        extended_dep = torch.tanh(dep).unsqueeze(3).expand(-1, -1, -1, self.num_case, 1)  # (b, seq, seq, case, 1)
        g_logits = self.v_a(torch.cat([h, extended_dep], dim=4)).squeeze(-1)  # (b, seq, seq, case)

        g_logits = g_logits.transpose(2, 3).contiguous()  # (b, seq, case, seq)

        extended_attention_mask = attention_mask.view(batch_size, 1, 1, sequence_len)  # (b, 1, 1, seq)
        mask = extended_attention_mask & ng_token_mask  # (b, seq, case, seq)
        mask = mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility

        g_logits += (1.0 - mask) * -1024.0  # (b, seq, case, seq)

        return torch.cat([g_logits, dep.transpose(2, 3).contiguous()], dim=2)  # (b, seq, case+1, seq)


# TODO: introduce dropout
# TODO: support coreference
class CaseInteractionModel(BaseModel):
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

        # head selection [Zhang+ 16]
        self.W_a = nn.Linear(bert_hidden_size, bert_hidden_size * self.num_case)
        self.U_a = nn.Linear(bert_hidden_size, bert_hidden_size * self.num_case)
        self.v_a = nn.Linear(bert_hidden_size + num_case, 1, bias=False)

        self.ref = nn.Linear(bert_hidden_size, 1)

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                ng_token_mask: torch.Tensor,   # (b, seq, case, seq)
                deps: torch.Tensor,            # (b, seq, seq)
                ) -> torch.Tensor:             # (b, seq, case, seq)
        batch_size, sequence_len = input_ids.size()
        # (b, seq, hid)
        sequence_output, _ = self.bert(input_ids,
                                       attention_mask=attention_mask)

        h_i = self.W_a(sequence_output)  # (b, seq, case*hid)
        h_j = self.U_a(sequence_output)  # (b, seq, case*hid)
        h_i = h_i.view(batch_size, sequence_len, self.num_case, -1)  # (b, seq, case, hid)
        h_j = h_j.view(batch_size, sequence_len, self.num_case, -1)  # (b, seq, case, hid)

        h = torch.tanh(self.dropout(h_i.unsqueeze(1) + h_j.unsqueeze(2)))  # (b, seq, seq, case, hid)
        # (b, seq, seq, case) -> (b, seq, seq, 1, case)
        ref = self.ref(h).squeeze(-1).unsqueeze(3).expand(-1, -1, -1, self.num_case, -1)
        # (b, seq, seq, case, hid+case) -> (b, seq, seq, case)
        g_logits = self.v_a(torch.cat([h, ref], dim=4)).squeeze(-1)

        g_logits = g_logits.transpose(2, 3).contiguous()  # (b, seq, case, seq)

        extended_attention_mask = attention_mask.view(batch_size, 1, 1, sequence_len)  # (b, 1, 1, seq)
        mask = extended_attention_mask & ng_token_mask  # (b, seq, case, seq)
        mask = mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility

        g_logits += (1.0 - mask) * -1024.0  # (b, seq, case, seq)

        return g_logits  # (b, seq, case, seq)
