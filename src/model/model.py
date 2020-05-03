from typing import Tuple

import torch
import torch.nn as nn
# from transformers import BertModel
from transformers import BertConfig

from base import BaseModel
from model.sub.refinement_layer import RefinementLayer1, RefinementLayer2, RefinementLayer3
from model.sub.mask import Mask
from model.sub.bert import BertModel


class BaselineModel(BaseModel):
    """述語側の重みを共通に"""

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

        self.l_prd = nn.Linear(bert_hidden_size, bert_hidden_size)
        self.l_arg = nn.Linear(bert_hidden_size, bert_hidden_size * self.num_case)
        self.outs = nn.ModuleList([nn.Linear(bert_hidden_size, 1, bias=False) for _ in range(self.num_case)])
        self.mask = Mask()

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                segment_ids: torch.Tensor,     # (b, seq)
                ng_token_mask: torch.Tensor,   # (b, seq, case, seq)
                deps: torch.Tensor,            # (b, seq, seq)
                _: torch.Tensor,
                ) -> torch.Tensor:             # (b, seq, case, seq)
        batch_size, sequence_len = input_ids.size()
        # (b, seq, hid)
        sequence_output, _ = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)

        # -> (b, seq, hid) -> (b, seq, case, hid)
        h_p = self.l_prd(self.dropout(sequence_output)).unsqueeze(2).expand(-1, -1, self.num_case, -1)
        # -> (b, seq, case*hid) -> (b, seq, case, hid)
        h_a = self.l_arg(self.dropout(sequence_output)).view(batch_size, sequence_len, self.num_case, -1)
        h = torch.tanh(self.dropout(h_p.unsqueeze(2) + h_a.unsqueeze(1)))  # (b, seq, seq, case, hid)
        outputs = [out(h[:, :, :, i, :]).squeeze(-1) for i, out in enumerate(self.outs)]  # [(b, seq, seq)]
        output = torch.stack(outputs, dim=2)  # (b, seq, case, seq)

        return self.mask(output, attention_mask, ng_token_mask)  # (b, seq, case, seq)


class BaselineModel2(BaseModel):
    """BaselineModelに加え共参照に別の重みを用意"""

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
        self.num_case_wo_coref = num_case
        self.coreference = coreference
        bert_hidden_size = self.bert.config.hidden_size

        self.l_prd = nn.Linear(bert_hidden_size, bert_hidden_size)
        self.l_arg = nn.Linear(bert_hidden_size, bert_hidden_size * self.num_case_wo_coref)
        self.l_coref = nn.Linear(bert_hidden_size, bert_hidden_size)
        self.outs = nn.ModuleList([nn.Linear(bert_hidden_size, 1, bias=False) for _ in range(self.num_case)])
        self.mask = Mask()

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                segment_ids: torch.Tensor,     # (b, seq)
                ng_token_mask: torch.Tensor,   # (b, seq, case, seq)
                deps: torch.Tensor,            # (b, seq, seq)
                ) -> torch.Tensor:             # (b, seq, case, seq)
        batch_size, sequence_len = input_ids.size()
        # (b, seq, hid)
        sequence_output, _ = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)

        h_p = self.l_prd(self.dropout(sequence_output))  # (b, seq, hid)
        h_p = h_p.unsqueeze(2).expand(-1, -1, self.num_case_wo_coref, -1)  # (b, seq, case-, hid)
        h_a = self.l_arg(self.dropout(sequence_output))  # (b, seq, case*hid)
        h_a = h_a.view(batch_size, sequence_len, self.num_case_wo_coref, -1)  # (b, seq, case-, hid)
        h_pa = torch.tanh(self.dropout(h_p.unsqueeze(1) + h_a.unsqueeze(2)))  # (b, seq, seq, case-, hid)

        if self.coreference:
            h_c = self.l_coref(self.dropout(sequence_output))  # (b, seq, hid)
            h_cc = torch.tanh(self.dropout(h_c.unsqueeze(1) + h_c.unsqueeze(2))).unsqueeze(3)  # (b, seq, seq, 1, hid)
            h = torch.cat([h_pa, h_cc], dim=3)  # (b, seq, seq, case, hid)
        else:
            h = h_pa  # (b, seq, seq, case, hid)
        outputs = [out(h[:, :, :, i, :]).squeeze(-1) for i, out in enumerate(self.outs)]  # [(b, seq, seq)]
        output = torch.stack(outputs, dim=2)  # (b, seq, case, seq)

        return self.mask(output, attention_mask, ng_token_mask)  # (b, seq, case, seq)


class BaselineModel3(BaseModel):
    """BaselineModel2に加え共参照に2つの重みを用意"""

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
        self.num_case_wo_coref = num_case
        self.coreference = coreference
        bert_hidden_size = self.bert.config.hidden_size

        self.l_prd = nn.Linear(bert_hidden_size, bert_hidden_size)
        self.l_arg = nn.Linear(bert_hidden_size, bert_hidden_size * self.num_case_wo_coref)
        self.l_coref1 = nn.Linear(bert_hidden_size, bert_hidden_size)
        self.l_coref2 = nn.Linear(bert_hidden_size, bert_hidden_size)
        self.outs = nn.ModuleList([nn.Linear(bert_hidden_size, 1, bias=False) for _ in range(self.num_case)])
        self.mask = Mask()

    def forward(self,
                input_ids: torch.Tensor,  # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                segment_ids: torch.Tensor,  # (b, seq)
                ng_token_mask: torch.Tensor,  # (b, seq, case, seq)
                deps: torch.Tensor,  # (b, seq, seq)
                ) -> torch.Tensor:  # (b, seq, case, seq)
        batch_size, sequence_len = input_ids.size()
        # (b, seq, hid)
        sequence_output, _ = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)

        h_p = self.l_prd(self.dropout(sequence_output))  # (b, seq, hid)
        h_p = h_p.unsqueeze(2).expand(-1, -1, self.num_case_wo_coref, -1)  # (b, seq, case-, hid)

        h_a = self.l_arg(self.dropout(sequence_output))  # (b, seq, case*hid)
        h_a = h_a.view(batch_size, sequence_len, self.num_case_wo_coref, -1)  # (b, seq, case-, hid)
        h_pa = torch.tanh(self.dropout(h_p.unsqueeze(1) + h_a.unsqueeze(2)))  # (b, seq, seq, case-, hid)

        if self.coreference:
            h_c1 = self.l_coref1(self.dropout(sequence_output))  # (b, seq, hid)
            h_c2 = self.l_coref2(self.dropout(sequence_output))  # (b, seq, hid)
            h_cc = torch.tanh(self.dropout(h_c1.unsqueeze(1) + h_c2.unsqueeze(2))).unsqueeze(3)  # (b, seq, seq, 1, hid)
            h = torch.cat([h_pa, h_cc], dim=3)  # (b, seq, seq, case, hid)
        else:
            h = h_pa  # (b, seq, seq, case, hid)
        outputs = [out(h[:, :, :, i, :]).squeeze(-1) for i, out in enumerate(self.outs)]  # [(b, seq, seq)]
        output = torch.stack(outputs, dim=2)  # (b, seq, case, seq)

        return self.mask(output, attention_mask, ng_token_mask)  # (b, seq, case, seq)


class BaselineModelOld(BaseModel):
    """NLPに出したときのモデル"""

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

        self.l_prd = nn.Linear(bert_hidden_size, bert_hidden_size * self.num_case)
        self.l_arg = nn.Linear(bert_hidden_size, bert_hidden_size * self.num_case)
        self.out = nn.Linear(bert_hidden_size, 1, bias=False)
        self.mask = Mask()

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                segment_ids: torch.Tensor,     # (b, seq)
                ng_token_mask: torch.Tensor,   # (b, seq, case, seq)
                deps: torch.Tensor,            # (b, seq, seq)
                ) -> torch.Tensor:             # (b, seq, case, seq)
        batch_size, sequence_len = input_ids.size()
        # (b, seq, hid)
        sequence_output, _ = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)

        h_p = self.l_prd(self.dropout(sequence_output))  # (b, seq, case*hid)
        h_a = self.l_arg(self.dropout(sequence_output))  # (b, seq, case*hid)
        h_p = h_p.view(batch_size, sequence_len, self.num_case, -1)  # (b, seq, case, hid)
        h_a = h_a.view(batch_size, sequence_len, self.num_case, -1)  # (b, seq, case, hid)
        h_pa = torch.tanh(self.dropout(h_p.unsqueeze(1) + h_a.unsqueeze(2)))  # (b, seq, seq, case, hid)
        # -> (b, seq, seq, case, 1) -> (b, seq, seq, case) -> (b, seq, case, seq)
        output = self.out(h_pa).squeeze(-1).transpose(2, 3).contiguous()

        return self.mask(output, attention_mask, ng_token_mask)  # (b, seq, case, seq)


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
                segment_ids: torch.Tensor,     # (b, seq)
                ng_token_mask: torch.Tensor,   # (b, seq, case, seq)
                deps: torch.Tensor,            # (b, seq, seq)
                ) -> Tuple[torch.Tensor, torch.Tensor]:  # (b, seq, case, seq), (b, seq, case, seq)
        # (b, seq, case, seq)
        base_logits = self.baseline_model(input_ids, attention_mask, segment_ids, ng_token_mask, deps)
        modification = self.refinement_layer(input_ids, attention_mask, base_logits.detach().softmax(dim=3))

        # modification を直接出力するときは mask を忘れずに

        return base_logits, base_logits.detach() + modification  # (b, seq, case, seq), (b, seq, case, seq)


class RefinementModel2(BaseModel):
    """最終結果に前段の出力を使わない"""
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
        self.mask = Mask()

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                segment_ids: torch.Tensor,     # (b, seq)
                ng_token_mask: torch.Tensor,   # (b, seq, case, seq)
                deps: torch.Tensor,            # (b, seq, seq)
                ) -> Tuple[torch.Tensor, torch.Tensor]:  # (b, seq, case, seq), (b, seq, case, seq)
        # (b, seq, case, seq)
        base_logits = self.baseline_model(input_ids, attention_mask, segment_ids, ng_token_mask, deps)
        output = self.refinement_layer(input_ids, attention_mask, base_logits.detach().softmax(dim=3))
        output = self.mask(output, attention_mask, ng_token_mask)

        return base_logits, output  # (b, seq, case, seq), (b, seq, case, seq)


class DuplicateModel(BaseModel):
    """RefinementModel の前段の logits を後段に与えないモデル"""

    def __init__(self,
                 bert_model: str,
                 vocab_size: int,
                 dropout: float,
                 num_case: int,
                 coreference: bool,
                 ) -> None:
        super().__init__()

        self.baseline_model1 = BaselineModel(bert_model, vocab_size, dropout, num_case, coreference)
        self.baseline_model2 = BaselineModel(bert_model, vocab_size, dropout, num_case, coreference)

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                segment_ids: torch.Tensor,     # (b, seq)
                ng_token_mask: torch.Tensor,   # (b, seq, case, seq)
                deps: torch.Tensor,            # (b, seq, seq)
                ) -> Tuple[torch.Tensor, torch.Tensor]:  # [(b, seq, case, seq)]
        # (b, seq, case, seq)
        base_logits = self.baseline_model1(input_ids, attention_mask, segment_ids, ng_token_mask, deps)
        modification = self.baseline_model2(input_ids, attention_mask, segment_ids, ng_token_mask, deps)

        return base_logits, base_logits.detach() + modification  # [(b, seq, case, seq)]


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
                segment_ids: torch.Tensor,     # (b, seq)
                ng_token_mask: torch.Tensor,   # (b, seq, seq)
                deps: torch.Tensor,            # (b, seq, seq)
                ) -> torch.Tensor:             # (b, seq, case, seq)
        # (b, seq, hid)
        sequence_output, _ = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)
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

        self.W_prd = nn.Linear(bert_hidden_size, bert_hidden_size * self.num_case)
        self.U_arg = nn.Linear(bert_hidden_size, bert_hidden_size * self.num_case)
        self.out = nn.Linear(bert_hidden_size + 1, 1, bias=False)

        self.W_dep = nn.Linear(bert_hidden_size, bert_hidden_size)
        self.U_dep = nn.Linear(bert_hidden_size, bert_hidden_size)
        self.out_dep = nn.Linear(bert_hidden_size, 1, bias=True)

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                segment_ids: torch.Tensor,     # (b, seq)
                ng_token_mask: torch.Tensor,   # (b, seq, seq)
                _: torch.Tensor,               # (b, seq, seq)
                ) -> Tuple[torch.Tensor, torch.Tensor]:  # (b, seq, case, seq), (b, seq, seq)
        # (b, seq, hid)
        sequence_output, _ = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)
        batch_size, sequence_len, hidden_dim = sequence_output.size()

        # dependency parsing
        dep_i = self.W_dep(sequence_output)  # (b, seq, hid)
        dep_j = self.U_dep(sequence_output)  # (b, seq, hid)
        # (b, seq, seq, hid) -> (b, seq, seq, 1)
        dep = self.out_dep(torch.tanh(self.dropout(dep_i.unsqueeze(1) + dep_j.unsqueeze(2))))

        # PAS analysis
        h_p = self.W_prd(self.dropout(sequence_output))  # (b, seq, case*hid)
        h_a = self.U_arg(self.dropout(sequence_output))  # (b, seq, case*hid)
        h_p = h_p.view(batch_size, sequence_len, self.num_case, -1)  # (b, seq, case, hid)
        h_a = h_a.view(batch_size, sequence_len, self.num_case, -1)  # (b, seq, case, hid)
        h = torch.tanh(self.dropout(h_p.unsqueeze(1) + h_a.unsqueeze(2)))  # (b, seq, seq, case, hid)
        expanded_dep = torch.tanh(dep).unsqueeze(3).expand(-1, -1, -1, self.num_case, 1)  # (b, seq, seq, case, 1)
        output = self.out(torch.cat([h, expanded_dep], dim=4)).squeeze(-1)  # (b, seq, seq, case)

        output = output.transpose(2, 3).contiguous()  # (b, seq, case, seq)

        extended_attention_mask = attention_mask.view(batch_size, 1, 1, sequence_len)  # (b, 1, 1, seq)
        mask = extended_attention_mask & ng_token_mask  # (b, seq, case, seq)

        output += (~mask).float() * -1024.0  # (b, seq, case, seq)

        return output, dep.unsqueeze(3)  # (b, seq, case, seq), (b, seq, seq)


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
                segment_ids: torch.Tensor,     # (b, seq)
                ng_token_mask: torch.Tensor,   # (b, seq, case, seq)
                deps: torch.Tensor,            # (b, seq, seq)
                ) -> torch.Tensor:             # (b, seq, case, seq)
        batch_size, sequence_len = input_ids.size()
        # (b, seq, hid)
        sequence_output, _ = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)

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


class CaseInteractionModel2(BaseModel):
    """CaseInteractionModel で reference ベクトルについても loss を計算する"""

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
        self.hidden_size = 512

        # head selection [Zhang+ 16]
        self.W_prd = nn.Linear(bert_hidden_size, self.hidden_size * self.num_case)
        self.U_arg = nn.Linear(bert_hidden_size, self.hidden_size * self.num_case)

        self.ref = nn.Linear(self.hidden_size, 1)

        self.mid_layers = nn.ModuleList([nn.Linear(self.hidden_size + self.num_case, self.hidden_size)] * self.num_case)
        self.output = nn.Linear(self.hidden_size, 1, bias=False)

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                segment_ids: torch.Tensor,     # (b, seq)
                ng_token_mask: torch.Tensor,   # (b, seq, case, seq)
                deps: torch.Tensor,            # (b, seq, seq)
                ) -> Tuple[torch.Tensor, torch.Tensor]:  # (b, seq, case, seq)
        batch_size, sequence_len = input_ids.size()
        # (b, seq, bhid)
        sequence_output, _ = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)

        h_p = self.W_prd(self.dropout(sequence_output))  # (b, seq, case*hid)
        h_a = self.U_arg(self.dropout(sequence_output))  # (b, seq, case*hid)
        h_p = h_p.view(batch_size, sequence_len, self.num_case, -1)  # (b, seq, case, hid)
        h_a = h_a.view(batch_size, sequence_len, self.num_case, -1)  # (b, seq, case, hid)

        h_pa = torch.tanh(self.dropout(h_p.unsqueeze(1) + h_a.unsqueeze(2)))  # (b, seq, seq, case, hid)
        h_pa = h_pa.transpose(2, 3).contiguous()  # (b, seq, case, seq, hid)
        # -> (b, seq, case, seq, 1) -> (b, seq, case, seq)
        output_base = self.ref(h_pa).squeeze(-1)
        #  -> (b, seq, 1, case, seq) -> (b, seq, case, case, seq) -> (b, seq, case, seq, case)
        ref = output_base.detach().unsqueeze(2).expand(-1, -1, self.num_case, -1, -1).transpose(3, 4).contiguous()
        # (b, seq, case, seq, hid+case)
        h = torch.tanh(self.dropout(torch.cat([h_pa, ref], dim=4)))
        h_mids = [layer(h[:, :, i, :, :]) for i, layer in enumerate(self.mid_layers)]  # [(b, seq, seq, hid)]
        h_mid = torch.stack(h_mids, dim=2)  # (b, seq, case, seq, hid)
        # -> (b, seq, case, seq, 1) -> (b, seq, case, seq)
        output = self.output(torch.tanh(self.dropout(h_mid))).squeeze(dim=4)

        extended_attention_mask = attention_mask.view(batch_size, 1, 1, sequence_len)  # (b, 1, 1, seq)
        mask = extended_attention_mask & ng_token_mask  # (b, seq, case, seq)

        output_base += (~mask).float() * -1024.0  # (b, seq, case, seq)
        output += (~mask).float() * -1024.0  # (b, seq, case, seq)

        return output_base, output  # (b, seq, case, seq)


class CommonsenseModel(BaseModel):
    """常識推論データセットとマルチタスク"""

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

        self.l_prd = nn.Linear(bert_hidden_size, bert_hidden_size)
        self.l_arg = nn.Linear(bert_hidden_size, bert_hidden_size * self.num_case)
        self.outs = nn.ModuleList([nn.Linear(bert_hidden_size, 1, bias=False) for _ in range(self.num_case)])
        self.mask = Mask()

        self.out_cs = nn.Linear(bert_hidden_size, 1)

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                segment_ids: torch.Tensor,     # (b, seq)
                ng_token_mask: torch.Tensor,   # (b, seq, case, seq) or (b, 1, 1, 1)
                deps: torch.Tensor,            # (b, seq, seq) or (b, 1, 1)
                ) -> Tuple[torch.Tensor, torch.Tensor]:  # (b, seq, case, seq), (b)
        batch_size, sequence_len = input_ids.size()
        # (b, seq, hid), (b, hid)
        sequence_output, pooled_output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)

        h_p = self.l_prd(self.dropout(sequence_output))  # (b, seq, hid)
        h_a = self.l_arg(self.dropout(sequence_output))  # (b, seq, case*hid)
        h_p = h_p.unsqueeze(2).expand(-1, -1, self.num_case, -1)  # (b, seq, case, hid)
        h_a = h_a.view(batch_size, sequence_len, self.num_case, -1)  # (b, seq, case, hid)
        h = torch.tanh(self.dropout(h_p.unsqueeze(1) + h_a.unsqueeze(2)))  # (b, seq, seq, case, hid)
        outputs = [out(h[:, :, :, i, :]).squeeze(-1) for i, out in enumerate(self.outs)]  # [(b, seq, seq)]
        output = torch.stack(outputs, dim=2)  # (b, seq, case, seq)

        output_contingency = self.out_cs(pooled_output).squeeze(-1)  # (b)

        return self.mask(output, attention_mask, ng_token_mask), output_contingency  # (b, seq, case, seq), (b)


class ConditionalModel1(BaseModel):
    """出力層に正解の半分を与える"""

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

        self.l_prd = nn.Linear(bert_hidden_size, bert_hidden_size)
        self.l_arg = nn.Linear(bert_hidden_size, bert_hidden_size * self.num_case)
        self.l_arg2 = nn.Linear(bert_hidden_size, bert_hidden_size * self.num_case)
        self.outs = nn.ModuleList([nn.Linear(bert_hidden_size, 1, bias=False) for _ in range(self.num_case)])
        self.mask = Mask()

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                segment_ids: torch.Tensor,     # (b, seq)
                ng_token_mask: torch.Tensor,   # (b, seq, case, seq)
                deps: torch.Tensor,            # (b, seq, seq)
                target: torch.Tensor,          # (b, seq, case, seq)
                ) -> torch.Tensor:             # (b, seq, case, seq)
        batch_size, sequence_len = input_ids.size()
        # (b, seq, hid)
        sequence_output, _ = self.bert(input_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=segment_ids)

        # -> (b, seq, hid) -> (b, seq, case, hid)
        h_p = self.l_prd(self.dropout(sequence_output)).unsqueeze(2).expand(-1, -1, self.num_case, -1)
        # -> (b, seq, case*hid) -> (b, seq, case, hid)
        h_a = self.l_arg(self.dropout(sequence_output)).view(batch_size, sequence_len, self.num_case, -1)
        h_pa = h_p.unsqueeze(2) + h_a.unsqueeze(1)  # (b, seq, seq, case, hid)

        # (b, seq, hid) -> (b, seq, case, hid)
        h_a2 = self.l_arg2(self.dropout(sequence_output)).view(batch_size, sequence_len, self.num_case, -1)
        # mask half of gold data
        prediction = target.bool() & torch.rand_like(target, dtype=torch.float).lt(0.5)  # (b, seq, case, seq)
        # (b, seq, hid) -> (b, seq, 1, 1, hid)
        h_pa += torch.einsum('bjch,bicj->bih', h_a2, prediction.float()).view(batch_size, sequence_len, 1, 1, -1)

        h_pa = torch.tanh(self.dropout(h_pa))  # (b, seq, seq, case, hid)
        outputs = [out(h_pa[:, :, :, i, :]).squeeze(-1) for i, out in enumerate(self.outs)]  # [(b, seq, seq)]
        output = torch.stack(outputs, dim=2)  # (b, seq, case, seq)

        return self.mask(output, attention_mask, ng_token_mask)  # (b, seq, case, seq)


class ConditionalModel2(BaseModel):
    """BERT の embedding に正解の半分を与える"""

    def __init__(self,
                 bert_model: str,
                 vocab_size: int,
                 dropout: float,
                 num_case: int,
                 coreference: bool,
                 ) -> None:
        super().__init__()
        self.num_case = num_case + int(coreference)

        self.bert: BertModel = BertModel.from_pretrained(
            bert_model,
            conditional_bert_embeddings=True,
            num_case=self.num_case,
        )
        self.bert.resize_token_embeddings(vocab_size)
        self.dropout = nn.Dropout(dropout)

        bert_hidden_size = self.bert.config.hidden_size

        self.l_prd = nn.Linear(bert_hidden_size, bert_hidden_size)
        self.l_arg = nn.Linear(bert_hidden_size, bert_hidden_size * self.num_case)
        self.outs = nn.ModuleList([nn.Linear(bert_hidden_size, 1, bias=False) for _ in range(self.num_case)])
        self.mask = Mask()

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                segment_ids: torch.Tensor,     # (b, seq)
                ng_token_mask: torch.Tensor,   # (b, seq, case, seq)
                deps: torch.Tensor,            # (b, seq, seq)
                target: torch.Tensor,          # (b, seq, case, seq)
                ) -> torch.Tensor:             # (b, seq, case, seq)
        batch_size, sequence_len = input_ids.size()
        prediction = target.bool() & torch.rand_like(target, dtype=torch.float).lt(0.5)  # (b, seq, case, seq)
        # (b, seq, hid)
        sequence_output, _ = self.bert(input_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=segment_ids,
                                       pas_prediction=prediction)

        # -> (b, seq, hid) -> (b, seq, case, hid)
        h_p = self.l_prd(self.dropout(sequence_output)).unsqueeze(2).expand(-1, -1, self.num_case, -1)
        # -> (b, seq, case*hid) -> (b, seq, case, hid)
        h_a = self.l_arg(self.dropout(sequence_output)).view(batch_size, sequence_len, self.num_case, -1)
        h = torch.tanh(self.dropout(h_p.unsqueeze(2) + h_a.unsqueeze(1)))  # (b, seq, seq, case, hid)
        outputs = [out(h[:, :, :, i, :]).squeeze(-1) for i, out in enumerate(self.outs)]  # [(b, seq, seq)]
        output = torch.stack(outputs, dim=2)  # (b, seq, case, seq)

        return self.mask(output, attention_mask, ng_token_mask)  # (b, seq, case, seq)


class ConditionalModel3(BaseModel):
    """BERT の attention に正解の半分を与える"""

    def __init__(self,
                 bert_model: str,
                 vocab_size: int,
                 dropout: float,
                 num_case: int,
                 coreference: bool,
                 ) -> None:
        super().__init__()
        self.num_case = num_case + int(coreference)
        config = BertConfig.from_pretrained(bert_model)

        self.rel_embeddings1 = nn.Embedding(self.num_case * 2 + 1, int(config.hidden_size / config.num_attention_heads))
        self.rel_embeddings2 = nn.Embedding(self.num_case * 2 + 1, int(config.hidden_size / config.num_attention_heads))
        self.bert: BertModel = BertModel.from_pretrained(
            bert_model,
            conditional_self_attention=True,
            rel_embeddings1=self.rel_embeddings1,
            rel_embeddings2=self.rel_embeddings2,
        )
        self.bert.resize_token_embeddings(vocab_size)
        self.dropout = nn.Dropout(dropout)

        bert_hidden_size = self.bert.config.hidden_size

        self.l_prd = nn.Linear(bert_hidden_size, bert_hidden_size)
        self.l_arg = nn.Linear(bert_hidden_size, bert_hidden_size * self.num_case)
        self.outs = nn.ModuleList([nn.Linear(bert_hidden_size, 1, bias=False) for _ in range(self.num_case)])
        self.mask = Mask()

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                segment_ids: torch.Tensor,     # (b, seq)
                ng_token_mask: torch.Tensor,   # (b, seq, case, seq)
                deps: torch.Tensor,            # (b, seq, seq)
                target: torch.Tensor,          # (b, seq, case, seq)
                ) -> torch.Tensor:             # (b, seq, case, seq)
        batch_size, sequence_len = input_ids.size()
        prediction = target.bool() & torch.rand_like(target, dtype=torch.float).lt(0.5)  # (b, seq, case, seq)
        # (b, seq, hid)
        sequence_output, _ = self.bert(input_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=segment_ids,
                                       pas_prediction=prediction)

        # -> (b, seq, hid) -> (b, seq, case, hid)
        h_p = self.l_prd(self.dropout(sequence_output)).unsqueeze(2).expand(-1, -1, self.num_case, -1)
        # -> (b, seq, case*hid) -> (b, seq, case, hid)
        h_a = self.l_arg(self.dropout(sequence_output)).view(batch_size, sequence_len, self.num_case, -1)
        h = torch.tanh(self.dropout(h_p.unsqueeze(2) + h_a.unsqueeze(1)))  # (b, seq, seq, case, hid)
        outputs = [out(h[:, :, :, i, :]).squeeze(-1) for i, out in enumerate(self.outs)]  # [(b, seq, seq)]
        output = torch.stack(outputs, dim=2)  # (b, seq, case, seq)

        return self.mask(output, attention_mask, ng_token_mask)  # (b, seq, case, seq)
