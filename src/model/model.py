from typing import Tuple

import torch
import torch.nn as nn
from transformers import BertModel

from base import BaseModel
from model.refinement.refinement_layer import RefinementLayer1, RefinementLayer2, RefinementLayer3


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
        sequence_output, _ = self.bert(input_ids,  attention_mask=attention_mask)

        h_i = self.W_a(self.dropout(sequence_output))  # (b, seq, case*hid)
        h_j = self.U_a(self.dropout(sequence_output))  # (b, seq, case*hid)
        h_i = h_i.view(batch_size, sequence_len, self.num_case, -1)  # (b, seq, case, hid)
        h_j = h_j.view(batch_size, sequence_len, self.num_case, -1)  # (b, seq, case, hid)
        # (b, seq, seq, case, hid) -> (b, seq, seq, case, 1) -> (b, seq, seq, case)
        g_logits = self.v_a(torch.tanh(self.dropout(h_i.unsqueeze(1) + h_j.unsqueeze(2)))).squeeze(-1)

        g_logits = g_logits.transpose(2, 3).contiguous()  # (b, seq, case, seq)

        extended_attention_mask = attention_mask.view(batch_size, 1, 1, sequence_len)  # (b, 1, 1, seq)
        mask = extended_attention_mask & ng_token_mask  # (b, seq, case, seq)

        g_logits += (~mask).float() * -1024.0  # (b, seq, case, seq)

        return g_logits  # (b, seq, case, seq)


class RefinementModel(BaseModel):
    def __init__(self,
                 bert_model: str,
                 vocab_size: int,
                 dropout: float,
                 num_case: int,
                 coreference: bool,
                 refinement_type: int,  # 1 or 2 or 3
                 refinement_bert_model: str,
                 ) -> None:
        super().__init__()

        self.baseline_model = BaselineModel(bert_model, vocab_size, dropout, num_case, coreference)
        args = (refinement_bert_model, vocab_size, dropout, num_case, coreference)
        if refinement_type == 1:
            self.refinement_layer = RefinementLayer1(*args)
        elif refinement_type == 2:
            self.refinement_layer = RefinementLayer2(*args)
        elif refinement_type == 3:
            self.refinement_layer = RefinementLayer3(*args)

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                ng_token_mask: torch.Tensor,   # (b, seq, case, seq)
                deps: torch.Tensor,            # (b, seq, seq)
                ) -> Tuple[torch.Tensor, torch.Tensor]:  # (b, seq, case, seq), (b, seq, case, seq)
        # (b, seq, case, seq)
        base_logits = self.baseline_model(input_ids, attention_mask, ng_token_mask, deps)
        refinement_logits = self.refinement_layer(input_ids, attention_mask, base_logits.detach().softmax(dim=3))

        # refinement_logits を直接出力するときは mask を忘れずに
        # extended_attention_mask = attention_mask.view(batch_size, 1, 1, sequence_len)  # (b, 1, 1, seq)
        # mask = extended_attention_mask & ng_token_mask  # (b, seq, case, seq)
        #
        # output += (~mask).float() * -1024.0  # (b, seq, case, seq)

        return base_logits, base_logits + refinement_logits  # (b, seq, case, seq), (b, seq, case, seq)


class RefinementTwiceModel(BaseModel):
    def __init__(self,
                 bert_model: str,
                 vocab_size: int,
                 dropout: float,
                 num_case: int,
                 coreference: bool,
                 refinement_type: int,  # 1 or 2 or 3
                 refinement_bert_model: str,
                 ) -> None:
        super().__init__()

        self.baseline_model = BaselineModel(bert_model, vocab_size, dropout, num_case, coreference)
        args = (refinement_bert_model, vocab_size, dropout, num_case, coreference)
        if refinement_type == 1:
            self.refinement_layer = RefinementLayer1(*args)
        elif refinement_type == 2:
            self.refinement_layer = RefinementLayer2(*args)
        elif refinement_type == 3:
            self.refinement_layer = RefinementLayer3(*args)

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                ng_token_mask: torch.Tensor,   # (b, seq, case, seq)
                deps: torch.Tensor,            # (b, seq, seq)
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # [(b, seq, case, seq)]
        # (b, seq, case, seq)
        base_logits = self.baseline_model(input_ids, attention_mask, ng_token_mask, deps)
        modification = self.refinement_layer(input_ids, attention_mask, base_logits.detach().softmax(dim=3))
        refined_logits = base_logits + modification
        modification = self.refinement_layer(input_ids, attention_mask, refined_logits.detach().softmax(dim=3))
        refined_logits2 = refined_logits + modification

        # refinement_logits を直接出力するときは mask を忘れずに
        # extended_attention_mask = attention_mask.view(batch_size, 1, 1, sequence_len)  # (b, 1, 1, seq)
        # mask = extended_attention_mask & ng_token_mask  # (b, seq, case, seq)
        #
        # output += (~mask).float() * -1024.0  # (b, seq, case, seq)

        return base_logits, refined_logits, refined_logits2  # [(b, seq, case, seq)]


class DependencyModel(BaseModel):
    """係り受けの情報を方向なしで与える"""
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
        self.v_a = nn.Linear(bert_hidden_size + 1, 1, bias=False)

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                ng_token_mask: torch.Tensor,   # (b, seq, seq)
                deps: torch.Tensor,            # (b, seq, seq)
                ) -> torch.Tensor:             # (b, seq, case, seq)
        # (b, seq, hid)
        sequence_output, _ = self.bert(input_ids,  attention_mask=attention_mask)
        batch_size, sequence_len, hidden_dim = sequence_output.size()

        h_i = self.W_a(self.dropout(sequence_output))  # (b, seq, case*hid)
        h_j = self.U_a(self.dropout(sequence_output))  # (b, seq, case*hid)
        h_i = h_i.view(batch_size, sequence_len, self.num_case, -1)  # (b, seq, case, hid)
        h_j = h_j.view(batch_size, sequence_len, self.num_case, -1)  # (b, seq, case, hid)
        h = torch.tanh(self.dropout(h_i.unsqueeze(1) + h_j.unsqueeze(2)))  # (b, seq, seq, case, hid)

        # make deps symmetric matrix
        deps = deps | deps.transpose(1, 2).contiguous()  # (b, seq, seq)
        # (b, seq, seq, case, 1)
        deps = deps.view(batch_size, sequence_len, sequence_len, 1, 1).expand(-1, -1, -1, self.num_case, 1).float()
        g_logits = self.v_a(torch.cat([h, deps], dim=4)).squeeze(-1)  # (b, seq, seq, case)

        g_logits = g_logits.transpose(2, 3).contiguous()  # (b, seq, case, seq)

        extended_attention_mask = attention_mask.view(batch_size, 1, 1, sequence_len)  # (b, 1, 1, seq)
        mask = extended_attention_mask & ng_token_mask  # (b, seq, case, seq)

        g_logits += (~mask).float() * -1024.0  # (b, seq, case, seq)

        return g_logits  # (b, seq, case, seq)


class LayerAttentionModel(BaseModel):
    """最終層の代わりにBERT 各層の重みを計算し足し合わせて使用 (今は利用不可)"""
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
        self.v_a = nn.Linear(bert_hidden_size, 1, bias=False)

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                ng_token_mask: torch.Tensor,   # (b, seq, seq)
                deps: torch.Tensor,            # (b, seq, seq)
                ) -> torch.Tensor:             # (b, seq, case, seq)
        # [(b, seq, hid)]
        sequence_output, _ = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(torch.stack(sequence_output, dim=1))  # (b, l, seq, hid)
        batch_size, num_layer, sequence_len, hidden_dim = sequence_output.size()

        attn_mid = self.layer_attn1(sequence_output)  # (b, l, seq, 100)
        attn = self.layer_attn2(torch.tanh(self.dropout(attn_mid)))  # (b, l, seq, 1)
        softmax_attn = torch.softmax(attn, dim=1)  # (b, l, seq, 1)
        weighted_output = (sequence_output * softmax_attn).sum(dim=1)  # (b, seq, hid)

        h_i = self.W_a(weighted_output)  # (b, seq, case*hid)
        h_j = self.U_a(weighted_output)  # (b, seq, case*hid)
        h_i = h_i.view(batch_size, sequence_len, self.num_case, -1)  # (b, seq, case, hid)
        h_j = h_j.view(batch_size, sequence_len, self.num_case, -1)  # (b, seq, case, hid)
        h = torch.tanh(self.dropout(h_i.unsqueeze(1) + h_j.unsqueeze(2)))  # (b, seq, seq, case, hid)
        g_logits = self.v_a(h).squeeze(-1)  # (b, seq, seq, case)

        g_logits = g_logits.transpose(2, 3).contiguous()  # (b, seq, case, seq)

        extended_attention_mask = attention_mask.view(batch_size, 1, 1, sequence_len)  # (b, 1, 1, seq)
        mask = extended_attention_mask & ng_token_mask  # (b, seq, case, seq)

        g_logits += (~mask).float() * -1024.0  # (b, seq, case, seq)

        return g_logits  # (b, seq, case, seq)


# TODO: introduce dropout
class MultitaskDepModel(BaseModel):
    """述語項構造解析と同時に構文解析も解く"""
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
        sequence_output, _ = self.bert(input_ids, attention_mask=attention_mask)
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

        g_logits += (~mask).float() * -1024.0  # (b, seq, case, seq)

        return torch.cat([g_logits, dep.transpose(2, 3).contiguous()], dim=2)  # (b, seq, case+1, seq)


class CaseInteractionModel(BaseModel):
    """あるサブワード間の(例えば)ヲ格らしさを計算する際にガ格らしさやニ格らしさも加味する"""
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
        self.v_a = nn.Linear(bert_hidden_size + self.num_case, 1, bias=False)

        self.ref = nn.Linear(bert_hidden_size, 1)

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                ng_token_mask: torch.Tensor,   # (b, seq, case, seq)
                deps: torch.Tensor,            # (b, seq, seq)
                ) -> torch.Tensor:             # (b, seq, case, seq)
        batch_size, sequence_len = input_ids.size()
        # (b, seq, hid)
        sequence_output, _ = self.bert(input_ids, attention_mask=attention_mask)

        h_i = self.W_a(self.dropout(sequence_output))  # (b, seq, case*hid)
        h_j = self.U_a(self.dropout(sequence_output))  # (b, seq, case*hid)
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

        g_logits += (~mask).float() * -1024.0  # (b, seq, case, seq)

        return g_logits  # (b, seq, case, seq)
