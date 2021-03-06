from typing import Tuple
from pathlib import Path

import torch
import torch.nn as nn
# from transformers import BertModel
from transformers import BertConfig, MBartConfig
from transformers import BartForConditionalGeneration, MBartForConditionalGeneration

from .sub.refinement_layer import RefinementLayer1, RefinementLayer2, RefinementLayer3
from .sub.mask import get_mask
from .sub.bert import BertModel
from .sub.conditional_model import (
    OutputConditionalModel,
    EmbeddingConditionalModel,
    AttentionConditionalModel,
    CaseAwareAttentionConditionalModel
)
from .loss import (
    cross_entropy_pas_loss,
    weighted_cross_entropy_pas_loss,
    cross_entropy_pas_dep_loss,
    multi_cross_entropy_pas_loss,
    cross_entropy_pas_commonsense_loss
)


class BaselineModel(nn.Module):
    """ベースライン"""

    def __init__(self,
                 pretrained_path: str,
                 vocab_size: int,
                 dropout: float,
                 num_case: int,
                 coreference: bool,
                 **_
                 ) -> None:
        super().__init__()

        self.bert: BertModel = BertModel.from_pretrained(pretrained_path)
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
                **_
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


class BaselineBartModel(nn.Module):
    """ベースライン"""

    def __init__(self,
                 pretrained_path: str,
                 vocab_size: int,
                 dropout: float,
                 num_case: int,
                 coreference: bool,
                 **_
                 ) -> None:
        """Japanese Bart Architecture is same as mBART.
        Only for naming.
        """
        super().__init__()

        self.mbart: MBartForConditionalGeneration = MBartForConditionalGeneration.from_pretrained(pretrained_path)
        self.mbart.resize_token_embeddings(vocab_size)
        self.dropout = nn.Dropout(dropout)

        self.num_case = num_case + int(coreference)
        mbart_hidden_size = self.mbart.config.hidden_size

        self.l_prd = nn.Linear(mbart_hidden_size, mbart_hidden_size * self.num_case)
        self.l_arg = nn.Linear(mbart_hidden_size, mbart_hidden_size * self.num_case)
        self.out = nn.Linear(mbart_hidden_size, 1, bias=False)

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                segment_ids: torch.Tensor,     # (b, seq)
                ng_token_mask: torch.Tensor,   # (b, seq, case, seq)
                target: torch.Tensor,          # (b, seq, case, seq)
                **_
                ) -> Tuple[torch.Tensor, ...]:  # (), (b, seq, case, seq)
        batch_size, sequence_len = input_ids.size()
        mask = get_mask(attention_mask, ng_token_mask)
        # (b, seq, hid)
        result = self.mbart(input_ids, decoder_input_ids=input_ids, attention_mask=attention_mask, decoder_attention_mask=attention_mask, use_cache=False, return_dict=True, output_hidden_states=True)
        sequence_output = result.decoder_hidden_states[-1]  # Decoder's last hidden state?

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


class BaselineMBartModel(nn.Module):
    """ベースライン"""

    def __init__(self,
                 pretrained_path: str,
                 vocab_size: int,
                 dropout: float,
                 num_case: int,
                 coreference: bool,
                 **_
                 ) -> None:
        super().__init__()

        self.mbart: MBartForConditionalGeneration = MBartForConditionalGeneration.from_pretrained(pretrained_path)
        self.mbart.resize_token_embeddings(vocab_size)
        self.dropout = nn.Dropout(dropout)

        self.num_case = num_case + int(coreference)
        mbart_hidden_size = self.mbart.config.hidden_size

        self.l_prd = nn.Linear(mbart_hidden_size, mbart_hidden_size * self.num_case)
        self.l_arg = nn.Linear(mbart_hidden_size, mbart_hidden_size * self.num_case)
        self.out = nn.Linear(mbart_hidden_size, 1, bias=False)

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                segment_ids: torch.Tensor,     # (b, seq)
                ng_token_mask: torch.Tensor,   # (b, seq, case, seq)
                target: torch.Tensor,          # (b, seq, case, seq)
                **_
                ) -> Tuple[torch.Tensor, ...]:  # (), (b, seq, case, seq)
        batch_size, sequence_len = input_ids.size()
        mask = get_mask(attention_mask, ng_token_mask)
        # (b, seq, hid)
        result = self.mbart(input_ids, decoder_input_ids=input_ids, attention_mask=attention_mask, decoder_attention_mask=attention_mask, use_cache=False, return_dict=True, output_hidden_states=True)
        sequence_output = result.decoder_hidden_states[-1]  # Decoder's last hidden state?

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


class BaselineBiBartModel(nn.Module):
    """ベースライン"""

    def __init__(self,
                 pretrained_path: str,
                 vocab_size: int,
                 dropout: float,
                 num_case: int,
                 coreference: bool,
                 **_
                 ) -> None:
        super().__init__()

        # BiBart is same as MBart; The only purpose for this is to distinguish them by name.
        # hf_config_path: str = str(Path(pretrained_path, "config.json"))
        # config = MBartConfig.from_json_file(hf_config_path)
        self.mbart: BartForConditionalGeneration = BartForConditionalGeneration.from_pretrained(pretrained_path)
        self.mbart.resize_token_embeddings(vocab_size)
        self.dropout = nn.Dropout(dropout)

        self.num_case = num_case + int(coreference)
        mbart_hidden_size = self.mbart.config.hidden_size

        self.l_prd = nn.Linear(mbart_hidden_size, mbart_hidden_size * self.num_case)
        self.l_arg = nn.Linear(mbart_hidden_size, mbart_hidden_size * self.num_case)
        self.out = nn.Linear(mbart_hidden_size, 1, bias=False)

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                segment_ids: torch.Tensor,     # (b, seq)
                ng_token_mask: torch.Tensor,   # (b, seq, case, seq)
                target: torch.Tensor,          # (b, seq, case, seq)
                **_
                ) -> Tuple[torch.Tensor, ...]:  # (), (b, seq, case, seq)
        batch_size, sequence_len = input_ids.size()
        mask = get_mask(attention_mask, ng_token_mask)
        # (b, seq, hid)
        result = self.mbart(input_ids, decoder_input_ids=input_ids, attention_mask=attention_mask, decoder_attention_mask=attention_mask, use_cache=False, return_dict=True, output_hidden_states=True)
        sequence_output = result.decoder_hidden_states[-1]  # Decoder's last hidden state?

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


class SpecialEmbInitModel(nn.Module):
    """Special token の埋め込み表現をうまく初期化"""

    def __init__(self,
                 pretrained_path: str,
                 special_emb_path: str,
                 vocab_size: int,
                 dropout: float,
                 num_case: int,
                 coreference: bool,
                 **_
                 ) -> None:
        super().__init__()

        self.bert: BertModel = BertModel.from_pretrained(pretrained_path)
        self.bert.resize_token_embeddings(vocab_size)
        self.dropout = nn.Dropout(dropout)

        special_embs = torch.load(special_emb_path)
        self.bert.set_input_embeddings(nn.Embedding.from_pretrained(special_embs, padding_idx=0))

        # emb = self.bert.get_input_embeddings()
        # emb_weight = emb.weight
        # emb_weight[orig_vocab_size:, :] = special_embs
        # self.bert.get_input_embeddings().weight = nn 
        # emb.requires_grad = True

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
                **_
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


class RefinementModel(nn.Module):
    def __init__(self,
                 pretrained_path: str,
                 vocab_size: int,
                 dropout: float,
                 num_case: int,
                 coreference: bool,
                 refinement_type: int,  # 1 or 2 or 3
                 refinement_bert_model: str,
                 ) -> None:
        super().__init__()

        self.baseline_model = BaselineModel(pretrained_path, vocab_size, dropout, num_case, coreference)
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
                **_
                ) -> Tuple[torch.Tensor, ...]:  # (), (b, seq, case, seq)
        # (b, seq, case, seq)
        _, base_logits = self.baseline_model(input_ids, attention_mask, segment_ids, ng_token_mask, target)
        modification = self.refinement_layer(input_ids, attention_mask, base_logits.detach().softmax(dim=3))
        refined_logits = base_logits.detach() + modification

        loss = multi_cross_entropy_pas_loss((base_logits, refined_logits), target)

        return loss, base_logits, refined_logits  # (), (b, seq, case, seq), (b, seq, case, seq)


class DuplicateModel(nn.Module):
    """RefinementModel の前段の logits を後段に与えないモデル"""

    def __init__(self,
                 pretrained_path: str,
                 vocab_size: int,
                 dropout: float,
                 num_case: int,
                 coreference: bool,
                 ) -> None:
        super().__init__()

        self.baseline_model1 = BaselineModel(pretrained_path, vocab_size, dropout, num_case, coreference)
        self.baseline_model2 = BaselineModel(pretrained_path, vocab_size, dropout, num_case, coreference)

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                segment_ids: torch.Tensor,     # (b, seq)
                ng_token_mask: torch.Tensor,   # (b, seq, case, seq)
                target: torch.Tensor,          # (b, seq, case, seq)
                **_
                ) -> Tuple[torch.Tensor, ...]:  # (), (b, seq, case, seq)
        # (b, seq, case, seq)
        _, base_logits = self.baseline_model1(input_ids, attention_mask, segment_ids, ng_token_mask, target)
        _, modification = self.baseline_model2(input_ids, attention_mask, segment_ids, ng_token_mask, target)
        refined_logits = base_logits.detach() + modification  # (b, seq, case, seq)

        loss = multi_cross_entropy_pas_loss((base_logits, refined_logits), target)

        return loss, base_logits, refined_logits  # (), (b, seq, case, seq), (b, seq, case, seq)


class GoldDepModel(nn.Module):
    """係り受けの情報を方向なしで与える"""

    def __init__(self,
                 pretrained_path: str,
                 vocab_size: int,
                 dropout: float,
                 num_case: int,
                 coreference: bool,
                 ) -> None:
        super().__init__()

        self.bert: BertModel = BertModel.from_pretrained(pretrained_path)
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
                **_
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


class MultitaskDepModel(nn.Module):
    """述語項構造解析と同時に構文解析も解く"""

    def __init__(self,
                 pretrained_path: str,
                 vocab_size: int,
                 dropout: float,
                 num_case: int,
                 coreference: bool,
                 ) -> None:
        super().__init__()

        self.bert: BertModel = BertModel.from_pretrained(pretrained_path)
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
                **_
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


class CaseInteractionModel(nn.Module):
    """あるサブワード間の(例えば)ヲ格らしさを計算する際にガ格らしさやニ格らしさも加味する"""

    def __init__(self,
                 pretrained_path: str,
                 vocab_size: int,
                 dropout: float,
                 num_case: int,
                 coreference: bool,
                 ) -> None:
        super().__init__()

        self.bert: BertModel = BertModel.from_pretrained(pretrained_path)
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


class CommonsenseModel(nn.Module):
    """常識推論データセットとマルチタスク"""

    def __init__(self,
                 pretrained_path: str,
                 vocab_size: int,
                 dropout: float,
                 num_case: int,
                 coreference: bool,
                 ) -> None:
        super().__init__()

        self.bert: BertModel = BertModel.from_pretrained(pretrained_path)
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
                task: torch.Tensor,            # (b)
                **_
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


class HalfGoldConditionalModel(nn.Module):

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
                **_
                ) -> Tuple[torch.Tensor, ...]:  # (), (b, seq, case, seq)
        half_gold = target.bool() & torch.rand_like(target, dtype=torch.float).lt(0.5)  # (b, seq, case, seq)
        output = self.conditional_model(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        segment_ids=segment_ids,
                                        ng_token_mask=ng_token_mask,
                                        pre_output=(~half_gold).float() * -1024.0)
        loss = cross_entropy_pas_loss(output, target)

        return loss, output


class FullGoldConditionalModel(nn.Module):

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
                **_
                ) -> Tuple[torch.Tensor, ...]:  # (), (b, seq, case, seq)
        mask = get_mask(attention_mask, ng_token_mask)  # (b, seq, case, seq)
        full_gold = target.float() + (~mask).float() * -1024.0  # (b, seq, case, seq)
        output = self.conditional_model(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        segment_ids=segment_ids,
                                        ng_token_mask=ng_token_mask,
                                        pre_output=full_gold)
        loss = cross_entropy_pas_loss(output, target)

        return loss, output


class IterativeRefinementModel(nn.Module):
    """複数回の推論で予測を refine していく"""

    def __init__(self, **kwargs):
        super().__init__()
        conditional_model = kwargs.pop('conditional_model')
        self.num_iter = kwargs.pop('num_iter')
        self.initial_model = BaselineModel(**kwargs)
        if conditional_model == 'emb':
            self.conditional_model = EmbeddingConditionalModel(**kwargs)
        elif conditional_model == 'atn':
            self.conditional_model = AttentionConditionalModel(**kwargs)
        elif conditional_model == 'catn':
            self.conditional_model = CaseAwareAttentionConditionalModel(**kwargs)
        elif conditional_model == 'out':
            self.conditional_model = OutputConditionalModel(**kwargs)

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                segment_ids: torch.Tensor,     # (b, seq)
                ng_token_mask: torch.Tensor,   # (b, seq, case, seq)
                target: torch.Tensor,          # (b, seq, case, seq)
                **_
                ) -> Tuple[torch.Tensor, ...]:  # (), (b, seq, case, seq)
        outputs, losses = [], []
        loss, output = self.initial_model(input_ids,
                                          attention_mask=attention_mask,
                                          segment_ids=segment_ids,
                                          ng_token_mask=ng_token_mask,
                                          target=target)
        outputs.append(output)
        losses.append(loss)
        for _ in range(self.num_iter):
            # (b, seq, case, seq)
            pre_output = outputs[-1].detach()
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


class FrozenIterativeRefinementModel(nn.Module):
    """初期予測レイヤに学習済みモデルを使用し、パラメータを更新しない"""

    def __init__(self, **kwargs):
        super().__init__()
        conditional_model = kwargs.pop('conditional_model')
        self.num_iter = kwargs.pop('num_iter')
        self.initial_model = BaselineModel(**kwargs)
        if conditional_model == 'emb':
            self.conditional_model = EmbeddingConditionalModel(**kwargs)
        elif conditional_model == 'atn':
            self.conditional_model = AttentionConditionalModel(**kwargs)
        elif conditional_model == 'catn':
            self.conditional_model = CaseAwareAttentionConditionalModel(**kwargs)
        elif conditional_model == 'out':
            self.conditional_model = OutputConditionalModel(**kwargs)

        self.initial_model = BaselineModel(**kwargs)
        dates = ('1124_154941', '1124_165941', '1124_180843')
        import random
        date = dates[random.randrange(len(dates))]
        ckpt = f'/mnt/elm/ueda/bpa/result.20201117/BaselineModel-wnf-4e-nict-ocz-vpa-npa-bar-cr/{date}/model_best.pth'
        state_dict = torch.load(ckpt, map_location=torch.device('cuda:0'))['state_dict']
        self.initial_model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
        self.initial_model.eval()

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                segment_ids: torch.Tensor,     # (b, seq)
                ng_token_mask: torch.Tensor,   # (b, seq, case, seq)
                target: torch.Tensor,          # (b, seq, case, seq)
                **_
                ) -> Tuple[torch.Tensor, ...]:  # (), (b, seq, case, seq)
        outputs, losses = [], []
        _, output = self.initial_model(input_ids,
                                          attention_mask=attention_mask,
                                          segment_ids=segment_ids,
                                          ng_token_mask=ng_token_mask,
                                          target=target)
        outputs.append(output)
        for _ in range(self.num_iter):
            # (b, seq, case, seq)
            pre_output = outputs[-1].detach()
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


class SingleIterativeRefinementModel(nn.Module):
    """初期予測も同一のモデルで行うIterativeRefinementModel
    旧名: IterativeRefinementModel"""

    def __init__(self, **kwargs):
        super().__init__()
        conditional_model = kwargs.pop('conditional_model')
        self.num_iter = kwargs.pop('num_iter')
        if conditional_model == 'emb':
            self.conditional_model = EmbeddingConditionalModel(**kwargs)
        elif conditional_model == 'atn':
            self.conditional_model = AttentionConditionalModel(**kwargs)
        elif conditional_model == 'catn':
            self.conditional_model = CaseAwareAttentionConditionalModel(**kwargs)
        elif conditional_model == 'out':
            self.conditional_model = OutputConditionalModel(**kwargs)

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                segment_ids: torch.Tensor,     # (b, seq)
                ng_token_mask: torch.Tensor,   # (b, seq, case, seq)
                target: torch.Tensor,          # (b, seq, case, seq)
                **_
                ) -> Tuple[torch.Tensor, ...]:  # (), (b, seq, case, seq)
        outputs, losses = [], []
        mask = get_mask(attention_mask, ng_token_mask)  # (b, seq, case, seq)
        for _ in range(self.num_iter):
            # (b, seq, case, seq)
            pre_output = outputs[-1].detach() if outputs \
                else torch.rand_like(target, dtype=torch.float) + (~mask).float() * -1024.0
            # pre_output = torch.rand_like(target, dtype=torch.float) + (~mask).float() * -1024.0
            # pre_output = target.float() + (~mask).float() * -1024.0
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


class AnnealingIterativeRefinementModel(nn.Module):
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
        elif conditional_model == 'catn':
            self.conditional_model = CaseAwareAttentionConditionalModel(**kwargs)
        elif conditional_model == 'out':
            self.conditional_model = OutputConditionalModel(**kwargs)

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                segment_ids: torch.Tensor,     # (b, seq)
                ng_token_mask: torch.Tensor,   # (b, seq, case, seq)
                target: torch.Tensor,          # (b, seq, case, seq)
                progress: float = 1.0,         # learning progress (0 ~ 1)
                **_
                ) -> Tuple[torch.Tensor, ...]:  # (), (b, seq, case, seq)
        outputs, losses = [], []
        mask = get_mask(attention_mask, ng_token_mask)  # (b, seq, case, seq)
        for _ in range(self.num_iter):
            # (b, seq, 1, 1)
            gold_ratio = 0.5 - progress * 0.5 if self.training else 0
            gold_mask = torch.rand_like(input_ids, dtype=torch.float).lt(gold_ratio).view(*input_ids.size(), 1, 1)

            # (b, seq, case, seq)
            if outputs:
                annealed_pre_output = (~target * -1024.0) * gold_mask + outputs[-1].detach() * ~gold_mask
            else:
                annealed_pre_output = torch.rand_like(target, dtype=torch.float) + (~mask).float() * -1024.0
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


class WeightedIterativeRefinementModel(nn.Module):
    """confidence で loss を重み付け"""

    def __init__(self, **kwargs):
        super().__init__()
        conditional_model = kwargs.pop('conditional_model')
        self.num_iter = kwargs.pop('num_iter')
        if conditional_model == 'emb':
            self.conditional_model = EmbeddingConditionalModel(**kwargs)
        elif conditional_model == 'atn':
            self.conditional_model = AttentionConditionalModel(**kwargs)
        elif conditional_model == 'catn':
            self.conditional_model = CaseAwareAttentionConditionalModel(**kwargs)
        elif conditional_model == 'out':
            self.conditional_model = OutputConditionalModel(**kwargs)

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                segment_ids: torch.Tensor,     # (b, seq)
                ng_token_mask: torch.Tensor,   # (b, seq, case, seq)
                target: torch.Tensor,          # (b, seq, case, seq)
                **_
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
            loss_weight = (1.0 - output.detach().softmax(dim=3))  # (b, seq, case, seq)
            loss = weighted_cross_entropy_pas_loss(output, target, loss_weight)
            outputs.append(output)
            losses.append(loss)
        loss = torch.stack(losses).mean()

        return (loss, *outputs)


class OvertGivenIterativeRefinementModel(nn.Module):
    """overtのみ常に与えられる"""

    def __init__(self, **kwargs):
        super().__init__()
        conditional_model = kwargs.pop('conditional_model')
        self.num_iter = kwargs.pop('num_iter')
        if conditional_model == 'emb':
            self.conditional_model = EmbeddingConditionalModel(**kwargs)
        elif conditional_model == 'atn':
            self.conditional_model = AttentionConditionalModel(**kwargs)
        elif conditional_model == 'catn':
            self.conditional_model = CaseAwareAttentionConditionalModel(**kwargs)
        elif conditional_model == 'out':
            self.conditional_model = OutputConditionalModel(**kwargs)

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                segment_ids: torch.Tensor,     # (b, seq)
                ng_token_mask: torch.Tensor,   # (b, seq, case, seq)
                target: torch.Tensor,          # (b, seq, case, seq)
                overt_mask: torch.Tensor,      # (b, seq, case, seq)
                **_
                ) -> Tuple[torch.Tensor, ...]:  # (), (b, seq, case, seq)
        outputs, losses = [], []
        mask = get_mask(attention_mask, ng_token_mask)  # (b, seq, case, seq)
        for _ in range(self.num_iter):
            # (b, seq, case, seq)
            pre_output = outputs[-1].detach() if outputs else (~mask).float() * -1024.0
            pre_output = pre_output + overt_mask.float() * 1024.0
            output = self.conditional_model(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            segment_ids=segment_ids,
                                            ng_token_mask=ng_token_mask,
                                            pre_output=pre_output)
            loss = weighted_cross_entropy_pas_loss(output, target, (~overt_mask).float())
            outputs.append(output)
            losses.append(loss)
        loss = torch.stack(losses).mean()

        return (loss, *outputs)


class OvertGivenConditionalModel(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        conditional_model = kwargs.pop('conditional_model')
        if conditional_model == 'emb':
            self.conditional_model = EmbeddingConditionalModel(**kwargs)
        elif conditional_model == 'atn':
            self.conditional_model = AttentionConditionalModel(**kwargs)
        elif conditional_model == 'catn':
            self.conditional_model = CaseAwareAttentionConditionalModel(**kwargs)
        elif conditional_model == 'out':
            self.conditional_model = OutputConditionalModel(**kwargs)

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                segment_ids: torch.Tensor,     # (b, seq)
                ng_token_mask: torch.Tensor,   # (b, seq, case, seq)
                target: torch.Tensor,          # (b, seq, case, seq)
                overt_mask: torch.Tensor,      # (b, seq, case, seq)
                **_
                ) -> Tuple[torch.Tensor, ...]:  # (), (b, seq, case, seq)
        output = self.conditional_model(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        segment_ids=segment_ids,
                                        ng_token_mask=ng_token_mask,
                                        pre_output=(~overt_mask).float() * -1024.0)
        loss = weighted_cross_entropy_pas_loss(output, target, (~overt_mask).float())

        return loss, output


class CandidateAwareModel(nn.Module):

    def __init__(self,
                 pretrained_path: str,
                 vocab_size: int,
                 dropout: float,
                 num_case: int,
                 coreference: bool,
                 ) -> None:
        super().__init__()
        atn_target = 'kv'
        self.num_case = num_case + int(coreference)
        config = BertConfig.from_pretrained(pretrained_path)
        self.rel_embeddings1 = nn.Embedding(2, int(config.hidden_size / config.num_attention_heads))
        self.rel_embeddings2 = nn.Embedding(2, int(config.hidden_size / config.num_attention_heads))
        kwargs = {'conditional_self_attention': True,
                  'rel_embeddings1': None,
                  'rel_embeddings2': None}
        if 'k' in atn_target:
            kwargs['rel_embeddings1'] = self.rel_embeddings1
        if 'v' in atn_target:
            kwargs['rel_embeddings2'] = self.rel_embeddings2
        self.bert: BertModel = BertModel.from_pretrained(pretrained_path, **kwargs)
        self.bert.resize_token_embeddings(vocab_size)
        self.dropout = nn.Dropout(dropout)

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
                **_
                ) -> Tuple[torch.Tensor, ...]:  # (), (b, seq, case, seq)
        batch_size, sequence_len = input_ids.size()
        device = input_ids.device
        # (b, seq, case, seq) -> (b, seq, 1, seq)
        mask = get_mask(attention_mask, ng_token_mask).any(dim=2, keepdim=True)
        candidate_mask = (mask | mask.transpose(1, 3)).any(dim=3, keepdim=True)  # (b, seq, 1, 1)
        extended_mask = torch.cat([~candidate_mask, candidate_mask], dim=2)  # (b, seq, 2, 1)
        zeros = torch.zeros((batch_size, sequence_len, 2, sequence_len - 1), dtype=torch.bool, device=device)
        extended_mask = torch.cat([zeros, extended_mask], dim=3)  # (b, seq, 2, seq)
        # (b, seq, hid)
        sequence_output, _ = self.bert(input_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=segment_ids,
                                       rel_weights=extended_mask.float())

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


class CorefCAModel(nn.Module):
    """
    出力層で共参照の結果を与える
    soft な重みで、別に用意した coref' の重みで変換
    COLING2020 に出したモデルのバグを修正したもの
    旧名: CoreferenceAwareModel3
    """

    def __init__(self,
                 pretrained_path: str,
                 vocab_size: int,
                 dropout: float,
                 num_case: int,
                 coreference: bool,
                 ) -> None:
        super().__init__()

        self.bert: BertModel = BertModel.from_pretrained(pretrained_path)
        self.bert.resize_token_embeddings(vocab_size)
        self.dropout = nn.Dropout(dropout)

        assert coreference is True
        self.num_case = num_case + int(coreference)
        bert_hidden_size = self.bert.config.hidden_size

        self.l_context = nn.Linear(bert_hidden_size, bert_hidden_size)

        self.l_prd = nn.Linear(bert_hidden_size, bert_hidden_size * self.num_case)
        self.l_arg = nn.Linear(bert_hidden_size, bert_hidden_size * self.num_case)
        self.out = nn.Linear(bert_hidden_size, 1, bias=False)

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                segment_ids: torch.Tensor,     # (b, seq)
                ng_token_mask: torch.Tensor,   # (b, seq, case, seq)
                target: torch.Tensor,          # (b, seq, case, seq)
                progress: float = 1.0,         # learning progress (0 ~ 1)
                **_
                ) -> Tuple[torch.Tensor, ...]:  # (), (b, seq, case, seq)
        batch_size, seq_len = input_ids.size()
        mask = get_mask(attention_mask, ng_token_mask)  # (b, seq, case, seq)
        # (b, seq, hid)
        sequence_output, _ = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)
        # -> (b, seq, case*hid) -> (b, seq, case, hid)
        h_p = self.l_prd(self.dropout(sequence_output)).view(batch_size, seq_len, self.num_case, -1)
        # -> (b, seq, case*hid) -> (b, seq, case, hid)
        h_a = self.l_arg(self.dropout(sequence_output)).view(batch_size, seq_len, self.num_case, -1)

        gold_ratio = 0.5 - progress * 0.5 if self.training else 0
        gold_mask = torch.rand_like(input_ids, dtype=torch.float).lt(gold_ratio).unsqueeze(-1)  # (b, seq, 1)

        # assuming coreference is the last case
        # (b, seq, seq, hid)
        h_coref = torch.tanh(self.dropout(h_p[:, :, -1, :].unsqueeze(2) + h_a[:, :, -1, :].unsqueeze(1)))
        out_coref = self.out(h_coref).squeeze(-1)  # (b, seq, seq)
        out_coref += (~mask[:, :, -1, :]).float() * -1024.0
        # (b, seq, seq)
        annealed_out_coref = (~target[:, :, -1, :] * -1024.0) * gold_mask + out_coref.detach() * ~gold_mask

        hid_context = self.l_context(self.dropout(sequence_output))  # (b, seq, hid)
        hid_context = hid_context.unsqueeze(2).expand(batch_size, seq_len, self.num_case, -1)  # (b, seq, case, hid)
        context = torch.einsum('bjch,bij->bich', hid_context, annealed_out_coref.softmax(dim=2))  # (b, seq, case, hid)
        h_a += context

        h_pa = torch.tanh(self.dropout(h_p.unsqueeze(2) + h_a.unsqueeze(1)))  # (b, seq, seq, case, hid)

        # -> (b, seq, seq, case, 1) -> (b, seq, seq, case) -> (b, seq, case, seq)
        output = self.out(h_pa).squeeze(-1).transpose(2, 3).contiguous()
        output += (~mask).float() * -1024.0

        loss = cross_entropy_pas_loss(output, target)

        return loss, output


class CoreferenceSeparatedModel(nn.Module):
    """
    2つの BERT を使って，右側では coref を解く．左側では coref の情報を使って他の 3タスクを解く
    旧名: CoreferenceSeparatedModel3
    """

    def __init__(self,
                 pretrained_path: str,
                 vocab_size: int,
                 dropout: float,
                 num_case: int,
                 coreference: bool,
                 ) -> None:
        super().__init__()

        self.bert: BertModel = BertModel.from_pretrained(pretrained_path)
        self.bert.resize_token_embeddings(vocab_size)
        self.dropout = nn.Dropout(dropout)

        self.coreference_model: BaselineModel = BaselineModel(pretrained_path, vocab_size, dropout, 0, True)
        dates = ('0623_214717', '0623_222941', '0623_230434', '0623_233957', '0624_001637')
        import random
        date = dates[random.randrange(len(dates))]
        # date = dates[0]
        ckpt = f'/mnt/elm/ueda/bpa/result.20200706/BaselineModelOld-all-4e-nict-coref-corefonly/{date}/model_best.pth'
        state_dict = torch.load(ckpt, map_location=torch.device('cuda:0'))['state_dict']
        self.coreference_model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
        self.coreference_model.eval()

        assert coreference is True
        self.num_case = num_case
        bert_hidden_size = self.bert.config.hidden_size

        self.l_context = nn.Linear(bert_hidden_size, bert_hidden_size)

        self.l_prd = nn.Linear(bert_hidden_size, bert_hidden_size * self.num_case)
        self.l_arg = nn.Linear(bert_hidden_size, bert_hidden_size * self.num_case)
        self.out = nn.Linear(bert_hidden_size, 1, bias=False)

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                segment_ids: torch.Tensor,     # (b, seq)
                ng_token_mask: torch.Tensor,   # (b, seq, case, seq)
                target: torch.Tensor,          # (b, seq, case, seq)
                **_
                ) -> Tuple[torch.Tensor, ...]:  # (), (b, seq, case, seq)
        batch_size, seq_len = input_ids.size()
        ng_token_mask_others, ng_token_mask_coref = ng_token_mask[:, :, :-1, :], ng_token_mask[:, :, -1, :].unsqueeze(2)
        target_others, target_coref = target[:, :, :-1, :], target[:, :, -1, :].unsqueeze(2)
        mask = get_mask(attention_mask, ng_token_mask_others)  # (b, seq, case, seq)
        # (b, seq, hid)
        sequence_output, _ = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)
        # -> (b, seq, case*hid) -> (b, seq, case, hid)
        h_p = self.l_prd(self.dropout(sequence_output)).view(batch_size, seq_len, self.num_case, -1)
        # -> (b, seq, case*hid) -> (b, seq, case, hid)
        h_a = self.l_arg(self.dropout(sequence_output)).view(batch_size, seq_len, self.num_case, -1)

        # (b, seq, 1, seq)
        _, out_coref = self.coreference_model(input_ids=input_ids,
                                              attention_mask=attention_mask,
                                              segment_ids=segment_ids,
                                              ng_token_mask=ng_token_mask_coref,
                                              target=target_coref)
        out_coref = out_coref.detach().squeeze(2)  # (b, seq, seq)

        hid_context = self.l_context(self.dropout(sequence_output))  # (b, seq, hid)
        hid_context = hid_context.unsqueeze(2).expand(batch_size, seq_len, self.num_case, -1)  # (b, seq, case, hid)
        context = torch.einsum('bjch,bij->bich', hid_context, out_coref.softmax(dim=2))  # (b, seq, case, hid)
        h_a += context

        h_pa = torch.tanh(self.dropout(h_p.unsqueeze(2) + h_a.unsqueeze(1)))  # (b, seq, seq, case, hid)

        # -> (b, seq, seq, case, 1) -> (b, seq, seq, case) -> (b, seq, case, seq)
        output = self.out(h_pa).squeeze(-1).transpose(2, 3).contiguous()
        output += (~mask).float() * -1024.0

        loss = cross_entropy_pas_loss(output, target_others)

        return loss, torch.cat([output, out_coref.unsqueeze(2)], dim=2)
