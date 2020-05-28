from typing import Tuple

import torch
import torch.nn as nn
# from transformers import BertModel

from base import BaseModel
from .sub.refinement_layer import RefinementLayer1, RefinementLayer2, RefinementLayer3
from .sub.mask import get_mask
from .sub.bert import BertModel
from .sub.conditional_model import OutputConditionalModel, EmbeddingConditionalModel, AttentionConditionalModel
from .loss import (
    cross_entropy_pas_loss,
    cross_entropy_pas_dep_loss,
    multi_cross_entropy_pas_loss,
    cross_entropy_pas_commonsense_loss
)


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
        self.outs = nn.ModuleList(nn.Linear(bert_hidden_size, 1, bias=False) for _ in range(self.num_case))

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                segment_ids: torch.Tensor,     # (b, seq)
                ng_token_mask: torch.Tensor,   # (b, seq, case, seq)
                target: torch.Tensor,          # (b, seq, case, seq)
                *_,
                **__
                ) -> Tuple[torch.Tensor, ...]:  # (), (b, seq, case, seq)
        batch_size, sequence_len = input_ids.size()
        mask = get_mask(attention_mask, ng_token_mask)
        # (b, seq, hid)
        sequence_output, _ = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)

        # -> (b, seq, hid) -> (b, seq, case, hid)
        h_p = self.l_prd(self.dropout(sequence_output)).unsqueeze(2).expand(-1, -1, self.num_case, -1)
        # -> (b, seq, case*hid) -> (b, seq, case, hid)
        h_a = self.l_arg(self.dropout(sequence_output)).view(batch_size, sequence_len, self.num_case, -1)
        h = torch.tanh(self.dropout(h_p.unsqueeze(2) + h_a.unsqueeze(1)))  # (b, seq, seq, case, hid)
        outputs = [out(h[:, :, :, i, :]).squeeze(-1) for i, out in enumerate(self.outs)]  # [(b, seq, seq)]
        output = torch.stack(outputs, dim=2)  # (b, seq, case, seq)
        output += (~mask).float() * -1024.0  # (b, seq, case, seq)

        loss = cross_entropy_pas_loss(output, target)

        return loss, output


class BaselineModelOld(BaseModel):
    """NLP2020に出したときのモデル"""

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

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                segment_ids: torch.Tensor,     # (b, seq)
                ng_token_mask: torch.Tensor,   # (b, seq, case, seq)
                target: torch.Tensor,          # (b, seq, case, seq)
                *_,
                **__
                ) -> Tuple[torch.Tensor, ...]:  # (), (b, seq, case, seq)
        batch_size, sequence_len = input_ids.size()
        mask = get_mask(attention_mask, ng_token_mask)
        # (b, seq, hid)
        sequence_output, _ = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)

        h_p = self.l_prd(self.dropout(sequence_output))  # (b, seq, case*hid)
        h_a = self.l_arg(self.dropout(sequence_output))  # (b, seq, case*hid)
        h_p = h_p.view(batch_size, sequence_len, self.num_case, -1)  # (b, seq, case, hid)
        h_a = h_a.view(batch_size, sequence_len, self.num_case, -1)  # (b, seq, case, hid)
        h_pa = torch.tanh(self.dropout(h_p.unsqueeze(2) + h_a.unsqueeze(1)))  # (b, seq, seq, case, hid)
        # -> (b, seq, seq, case, 1) -> (b, seq, seq, case) -> (b, seq, case, seq)
        output = self.out(h_pa).squeeze(-1).transpose(2, 3).contiguous()
        output += (~mask).float() * -1024.0

        loss = cross_entropy_pas_loss(output, target)

        return loss, output


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
                target: torch.Tensor,          # (b, seq, case, seq)
                *_,
                **__
                ) -> Tuple[torch.Tensor, ...]:  # (), (b, seq, case, seq)
        # (b, seq, case, seq)
        _, base_logits = self.baseline_model(input_ids, attention_mask, segment_ids, ng_token_mask, target)
        modification = self.refinement_layer(input_ids, attention_mask, base_logits.detach().softmax(dim=3))
        refined_logits = base_logits.detach() + modification

        loss = multi_cross_entropy_pas_loss((base_logits, refined_logits), target)

        return loss, base_logits, refined_logits  # (), (b, seq, case, seq), (b, seq, case, seq)


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

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                segment_ids: torch.Tensor,     # (b, seq)
                ng_token_mask: torch.Tensor,   # (b, seq, case, seq)
                target: torch.Tensor,          # (b, seq, case, seq)
                *_,
                **__
                ) -> Tuple[torch.Tensor, ...]:  # (), (b, seq, case, seq)
        mask = get_mask(attention_mask, ng_token_mask)
        # (b, seq, case, seq)
        _, base_logits = self.baseline_model(input_ids, attention_mask, segment_ids, ng_token_mask, target)
        refined_logits = self.refinement_layer(input_ids, attention_mask, base_logits.detach().softmax(dim=3))
        refined_logits += (~mask).float() * -1024.0

        loss = multi_cross_entropy_pas_loss((base_logits, refined_logits), target)

        return loss, base_logits, refined_logits  # (), (b, seq, case, seq), (b, seq, case, seq)


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
                target: torch.Tensor,          # (b, seq, case, seq)
                *_,
                **__
                ) -> Tuple[torch.Tensor, ...]:  # (), (b, seq, case, seq)
        # (b, seq, case, seq)
        _, base_logits = self.baseline_model1(input_ids, attention_mask, segment_ids, ng_token_mask, target)
        _, modification = self.baseline_model2(input_ids, attention_mask, segment_ids, ng_token_mask, target)
        refined_logits = base_logits.detach() + modification  # (b, seq, case, seq)

        loss = multi_cross_entropy_pas_loss((base_logits, refined_logits), target)

        return loss, base_logits, refined_logits  # (), (b, seq, case, seq), (b, seq, case, seq)


class GoldDepModel(BaseModel):
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

        self.l_prd = nn.Linear(bert_hidden_size, bert_hidden_size)
        self.l_arg = nn.Linear(bert_hidden_size, bert_hidden_size * self.num_case)
        self.outs = nn.ModuleList(nn.Linear(bert_hidden_size + 1, 1, bias=False) for _ in range(self.num_case))

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                segment_ids: torch.Tensor,     # (b, seq)
                ng_token_mask: torch.Tensor,   # (b, seq, case, seq)
                target: torch.Tensor,          # (b, seq, case, seq)
                deps: torch.Tensor,            # (b, seq, seq)
                *_,
                **__
                ) -> Tuple[torch.Tensor, ...]:  # (), (b, seq, case, seq)
        batch_size, sequence_len = input_ids.size()
        mask = get_mask(attention_mask, ng_token_mask)
        # (b, seq, hid)
        sequence_output, _ = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)

        # -> (b, seq, hid) -> (b, seq, case, hid)
        h_p = self.l_prd(self.dropout(sequence_output)).unsqueeze(2).expand(-1, -1, self.num_case, -1)
        # -> (b, seq, case*hid) -> (b, seq, case, hid)
        h_a = self.l_arg(self.dropout(sequence_output)).view(batch_size, sequence_len, self.num_case, -1)
        h = torch.tanh(self.dropout(h_p.unsqueeze(2) + h_a.unsqueeze(1)))  # (b, seq, seq, case, hid)

        # make deps symmetric matrix
        deps = deps | deps.transpose(1, 2).contiguous()  # (b, seq, seq)
        # (b, seq, seq, case, 1)
        deps = deps.view(batch_size, sequence_len, sequence_len, 1, 1).expand(-1, -1, -1, self.num_case, 1).float()
        h = torch.cat([h, deps], dim=4)  # (b, seq, seq, case)

        outputs = [out(h[:, :, :, i, :]).squeeze(-1) for i, out in enumerate(self.outs)]  # [(b, seq, seq)]
        output = torch.stack(outputs, dim=2)  # (b, seq, case, seq)
        output += (~mask).float() * -1024.0  # (b, seq, case, seq)

        loss = cross_entropy_pas_loss(output, target)

        return loss, output


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

        self.W_dep = nn.Linear(bert_hidden_size, bert_hidden_size)
        self.U_dep = nn.Linear(bert_hidden_size, bert_hidden_size)
        self.out_dep = nn.Linear(bert_hidden_size, 1, bias=True)

        self.l_prd = nn.Linear(bert_hidden_size, bert_hidden_size)
        self.l_arg = nn.Linear(bert_hidden_size, bert_hidden_size * self.num_case)
        self.outs = nn.ModuleList(nn.Linear(bert_hidden_size + 1, 1, bias=False) for _ in range(self.num_case))

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                segment_ids: torch.Tensor,     # (b, seq)
                ng_token_mask: torch.Tensor,   # (b, seq, seq)
                target: torch.Tensor,          # (b, seq, case, seq)
                deps: torch.Tensor,            # (b, seq, seq)
                *_,
                **__
                ) -> Tuple[torch.Tensor, ...]:  # (), (b, seq, case, seq)
        batch_size, sequence_len = input_ids.size()
        mask = get_mask(attention_mask, ng_token_mask)
        # (b, seq, hid)
        sequence_output, _ = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)

        # dependency parsing
        dep_i = self.W_dep(sequence_output)  # (b, seq, hid)
        dep_j = self.U_dep(sequence_output)  # (b, seq, hid)
        # (b, seq, seq, hid) -> (b, seq, seq, 1) -> (b, seq, seq)
        h_dep = self.out_dep(torch.tanh(self.dropout(dep_i.unsqueeze(2) + dep_j.unsqueeze(1)))).squeeze(-1)

        # PAS analysis
        # -> (b, seq, hid) -> (b, seq, case, hid)
        h_p = self.l_prd(self.dropout(sequence_output)).unsqueeze(2).expand(-1, -1, self.num_case, -1)
        # -> (b, seq, case*hid) -> (b, seq, case, hid)
        h_a = self.l_arg(self.dropout(sequence_output)).view(batch_size, sequence_len, self.num_case, -1)
        h = torch.tanh(self.dropout(h_p.unsqueeze(2) + h_a.unsqueeze(1)))  # (b, seq, seq, case, hid)

        # -> (b, seq, seq, 1) -> (b, seq, seq, case) -> (b, seq, seq, case, 1)
        expanded_dep = torch.tanh(h_dep).unsqueeze(3).expand(-1, -1, -1, self.num_case).unsqueeze(4)
        h = torch.cat([h, expanded_dep], dim=4)

        outputs = [out(h[:, :, :, i, :]).squeeze(-1) for i, out in enumerate(self.outs)]  # [(b, seq, seq)]
        output = torch.stack(outputs, dim=2)  # (b, seq, case, seq)
        output += (~mask).float() * -1024.0  # (b, seq, case, seq)

        loss = cross_entropy_pas_dep_loss((output, h_dep), target, deps)

        return loss, output, h_dep


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
        self.hidden_size = 512

        # head selection [Zhang+ 16]
        self.W_prd = nn.Linear(bert_hidden_size, self.hidden_size * self.num_case)
        self.U_arg = nn.Linear(bert_hidden_size, self.hidden_size * self.num_case)

        self.ref = nn.Linear(self.hidden_size, 1)

        self.mid_layers = nn.ModuleList(nn.Linear(self.hidden_size + self.num_case, self.hidden_size)
                                        for _ in range(self.num_case))
        self.output = nn.Linear(self.hidden_size, 1, bias=False)

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                segment_ids: torch.Tensor,     # (b, seq)
                ng_token_mask: torch.Tensor,   # (b, seq, case, seq)
                target: torch.Tensor,          # (b, seq, case, seq)
                *_,
                **__
                ) -> Tuple[torch.Tensor, ...]:  # (), (b, seq, case, seq)
        batch_size, sequence_len = input_ids.size()
        mask = get_mask(attention_mask, ng_token_mask)
        # (b, seq, bhid)
        sequence_output, _ = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)

        h_p = self.W_prd(self.dropout(sequence_output))  # (b, seq, case*hid)
        h_a = self.U_arg(self.dropout(sequence_output))  # (b, seq, case*hid)
        h_p = h_p.view(batch_size, sequence_len, self.num_case, -1)  # (b, seq, case, hid)
        h_a = h_a.view(batch_size, sequence_len, self.num_case, -1)  # (b, seq, case, hid)

        h_pa = torch.tanh(self.dropout(h_p.unsqueeze(2) + h_a.unsqueeze(1)))  # (b, seq, seq, case, hid)
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

        output_base += (~mask).float() * -1024.0  # (b, seq, case, seq)
        output += (~mask).float() * -1024.0  # (b, seq, case, seq)

        loss = multi_cross_entropy_pas_loss((output_base, output), target)

        return loss, output_base, output  # (), (b, seq, case, seq), (b, seq, case, seq)


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
        self.outs = nn.ModuleList(nn.Linear(bert_hidden_size, 1, bias=False) for _ in range(self.num_case))

        self.out_cs = nn.Linear(bert_hidden_size, 1)

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                segment_ids: torch.Tensor,     # (b, seq)
                ng_token_mask: torch.Tensor,   # (b, seq, case, seq) or (b, 1, 1, 1)
                target: torch.Tensor,          # (b, seq, case, seq)
                _,
                task: torch.Tensor,            # (b)
                *args,
                **kwargs
                ) -> Tuple[torch.Tensor, ...]:  # (), (b, seq, case, seq)
        batch_size, sequence_len = input_ids.size()
        mask = get_mask(attention_mask, ng_token_mask)
        # (b, seq, hid), (b, hid)
        sequence_output, pooled_output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)

        h_p = self.l_prd(self.dropout(sequence_output))  # (b, seq, hid)
        h_a = self.l_arg(self.dropout(sequence_output))  # (b, seq, case*hid)
        h_p = h_p.unsqueeze(2).expand(-1, -1, self.num_case, -1)  # (b, seq, case, hid)
        h_a = h_a.view(batch_size, sequence_len, self.num_case, -1)  # (b, seq, case, hid)
        h = torch.tanh(self.dropout(h_p.unsqueeze(2) + h_a.unsqueeze(1)))  # (b, seq, seq, case, hid)
        outputs = [out(h[:, :, :, i, :]).squeeze(-1) for i, out in enumerate(self.outs)]  # [(b, seq, seq)]
        output = torch.stack(outputs, dim=2)  # (b, seq, case, seq)
        output += (~mask).float() * -1024.0

        output_contingency = self.out_cs(pooled_output).squeeze(-1)  # (b)

        loss = cross_entropy_pas_commonsense_loss((output, output_contingency), target, task)

        return loss, output, output_contingency  # (b, seq, case, seq), (b)


class HalfGoldConditionalModel(BaseModel):

    def __init__(self, **kwargs):
        super().__init__()
        conditional_model = kwargs.pop('conditional_model')
        if conditional_model == 'emb':
            self.conditional_model = EmbeddingConditionalModel(**kwargs)
        elif conditional_model == 'atn':
            self.conditional_model = AttentionConditionalModel(**kwargs)
        elif conditional_model == 'out':
            self.conditional_model = OutputConditionalModel(**kwargs)

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                segment_ids: torch.Tensor,     # (b, seq)
                ng_token_mask: torch.Tensor,   # (b, seq, case, seq)
                target: torch.Tensor,          # (b, seq, case, seq)
                pre_output: torch.Tensor,      # (b, seq, case, seq)
                *_,
                **__
                ) -> Tuple[torch.Tensor, ...]:  # (), (b, seq, case, seq)
        half_gold = target.bool() & torch.rand_like(target, dtype=torch.float).lt(0.5)  # (b, seq, case, seq)
        output = self.conditional_model(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        segment_ids=segment_ids,
                                        ng_token_mask=ng_token_mask,
                                        pre_output=(~half_gold).float() * -1024.0)
        loss = cross_entropy_pas_loss(output, target)

        return loss, output


class IterativeRefinementModel(BaseModel):
    """複数回の推論で予測を refine していく"""

    def __init__(self, **kwargs):
        super().__init__()
        conditional_model = kwargs.pop('conditional_model')
        self.num_iter = kwargs.pop('num_iter')
        if conditional_model == 'emb':
            self.conditional_model = EmbeddingConditionalModel(**kwargs)
        elif conditional_model == 'atn':
            self.conditional_model = AttentionConditionalModel(**kwargs)
        elif conditional_model == 'out':
            self.conditional_model = OutputConditionalModel(**kwargs)

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                segment_ids: torch.Tensor,     # (b, seq)
                ng_token_mask: torch.Tensor,   # (b, seq, case, seq)
                target: torch.Tensor,          # (b, seq, case, seq)
                *_,
                **__
                ) -> Tuple[torch.Tensor, ...]:  # (), (b, seq, case, seq)
        outputs, losses = [], []
        mask = get_mask(attention_mask, ng_token_mask)  # (b, seq, case, seq)
        for _ in range(self.num_iter):
            # (b, seq, case, seq)
            pre_output = outputs[-1].detach() if outputs else (~mask).float() * -1024.0
            output = self.conditional_model(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            segment_ids=segment_ids,
                                            ng_token_mask=ng_token_mask,
                                            pre_output=pre_output)
            loss = cross_entropy_pas_loss(output, target)
            outputs.append(output)
            losses.append(loss)
        loss = torch.stack(losses).mean()

        return (loss, *outputs)


class NoRelInitIterativeRefinementModel(BaseModel):
    """複数回の推論で予測を refine していく"""

    def __init__(self, **kwargs):
        super().__init__()
        conditional_model = kwargs.pop('conditional_model')
        self.num_iter = kwargs.pop('num_iter')
        if conditional_model == 'emb':
            self.conditional_model = EmbeddingConditionalModel(**kwargs)
        elif conditional_model == 'atn':
            self.conditional_model = AttentionConditionalModel(**kwargs)
        elif conditional_model == 'out':
            self.conditional_model = OutputConditionalModel(**kwargs)

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                segment_ids: torch.Tensor,     # (b, seq)
                ng_token_mask: torch.Tensor,   # (b, seq, case, seq)
                target: torch.Tensor,          # (b, seq, case, seq)
                *_,
                **__
                ) -> Tuple[torch.Tensor, ...]:  # (), (b, seq, case, seq)
        outputs, losses = [], []
        for _ in range(self.num_iter):
            # (b, seq, case, seq)
            pre_output = outputs[-1].detach() if outputs else torch.full_like(target, -1024.0, dtype=torch.float)
            output = self.conditional_model(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            segment_ids=segment_ids,
                                            ng_token_mask=ng_token_mask,
                                            pre_output=pre_output)
            loss = cross_entropy_pas_loss(output, target)
            outputs.append(output)
            losses.append(loss)
        loss = torch.stack(losses).mean()

        return (loss, *outputs)


class AnnealingIterativeRefinementModel(BaseModel):
    """学習初期は前回の予測としてでたらめなものが入力され，うまく refinement 機構が学習されないと考えられる．
    そのため，初期は正解を与え，学習が進むにつれ自身の出力を与えるようにする"""

    def __init__(self, **kwargs):
        super().__init__()
        conditional_model = kwargs.pop('conditional_model')
        self.num_iter = kwargs.pop('num_iter')
        if conditional_model == 'emb':
            self.conditional_model = EmbeddingConditionalModel(**kwargs)
        elif conditional_model == 'atn':
            self.conditional_model = AttentionConditionalModel(**kwargs)
        elif conditional_model == 'out':
            self.conditional_model = OutputConditionalModel(**kwargs)

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                segment_ids: torch.Tensor,     # (b, seq)
                ng_token_mask: torch.Tensor,   # (b, seq, case, seq)
                target: torch.Tensor,          # (b, seq, case, seq)
                *args,
                **kwargs,
                ) -> Tuple[torch.Tensor, ...]:  # (), (b, seq, case, seq)
        outputs, losses = [], []
        mask = get_mask(attention_mask, ng_token_mask)  # (b, seq, case, seq)
        for _ in range(self.num_iter):
            # (b, seq, 1, 1)
            if self.training:
                progress = kwargs['progress']  # learning progress (0 ~ 1)
                gold_mask = torch.rand_like(input_ids, dtype=torch.float).lt(progress).view(*input_ids.size(), 1, 1)
            else:
                gold_mask = torch.full_like(input_ids, True, dtype=torch.bool).view(*input_ids.size(), 1, 1)
            # (b, seq, case, seq)
            if outputs:
                annealed_pre_output = (~target * -1024.0) * ~gold_mask + outputs[-1].detach() * gold_mask
            else:
                annealed_pre_output = (~mask).float() * -1024.0
            output = self.conditional_model(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            segment_ids=segment_ids,
                                            ng_token_mask=ng_token_mask,
                                            pre_output=annealed_pre_output)
            loss = cross_entropy_pas_loss(output, target)
            outputs.append(output)
            losses.append(loss)
        loss = torch.stack(losses).mean()

        return (loss, *outputs)
