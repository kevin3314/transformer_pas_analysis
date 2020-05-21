from typing import Tuple

import torch
import torch.nn as nn
# from transformers import BertModel
from transformers import BertConfig

from base import BaseModel
from .mask import get_mask
from .bert import BertModel
from ..loss import cross_entropy_pas_loss


class OutputConditionalModel(BaseModel):
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
        self.outs = nn.ModuleList(nn.Linear(bert_hidden_size, 1, bias=False) for _ in range(self.num_case))

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                segment_ids: torch.Tensor,     # (b, seq)
                ng_token_mask: torch.Tensor,   # (b, seq, case, seq)
                target: torch.Tensor,          # (b, seq, case, seq)
                pre_output: torch.Tensor,      # (b, seq, case, seq)
                ) -> Tuple[torch.Tensor, ...]:  # (), (b, seq, case, seq)
        batch_size, sequence_len = input_ids.size()
        mask = get_mask(attention_mask, ng_token_mask)
        eye = torch.eye(sequence_len, dtype=torch.bool, device=input_ids.device)  # (seq)
        pre_prediction = eye[pre_output.argmax(dim=3)] & mask  # (b, seq, case, seq)
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
        # (b, seq, hid) -> (b, seq, 1, 1, hid)
        h_pa += torch.einsum('bjch,bicj->bih', h_a2, pre_prediction).view(batch_size, sequence_len, 1, 1, -1)

        h_pa = torch.tanh(self.dropout(h_pa))  # (b, seq, seq, case, hid)
        outputs = [out(h_pa[:, :, :, i, :]).squeeze(-1) for i, out in enumerate(self.outs)]  # [(b, seq, seq)]
        output = torch.stack(outputs, dim=2)  # (b, seq, case, seq)
        output += (~mask).float() * -1024.0  # (b, seq, case, seq)

        loss = cross_entropy_pas_loss(output, target)

        return loss, output


class EmbeddingConditionalModel(BaseModel):
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
        self.outs = nn.ModuleList(nn.Linear(bert_hidden_size, 1, bias=False) for _ in range(self.num_case))

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                segment_ids: torch.Tensor,     # (b, seq)
                ng_token_mask: torch.Tensor,   # (b, seq, case, seq)
                target: torch.Tensor,          # (b, seq, case, seq)
                pre_output: torch.Tensor,      # (b, seq, case, seq)
                ) -> Tuple[torch.Tensor, ...]:  # (), (b, seq, case, seq)
        batch_size, sequence_len = input_ids.size()
        mask = get_mask(attention_mask, ng_token_mask)
        eye = torch.eye(sequence_len, dtype=torch.bool, device=input_ids.device)  # (seq)
        pre_prediction = eye[pre_output.argmax(dim=3)] & mask  # (b, seq, case, seq)
        # (b, seq, hid)
        sequence_output, _ = self.bert(input_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=segment_ids,
                                       pre_output=pre_prediction)

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


class AttentionConditionalModel(BaseModel):
    """BERT の attention に正解の半分を与える"""

    def __init__(self,
                 bert_model: str,
                 vocab_size: int,
                 dropout: float,
                 num_case: int,
                 coreference: bool,
                 output_aggr: str,
                 ) -> None:
        super().__init__()
        self.num_case = num_case + int(coreference)
        self.output_aggr = getattr(self, f'_{output_aggr}_output_aggr')
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
        self.outs = nn.ModuleList(nn.Linear(bert_hidden_size, 1, bias=False) for _ in range(self.num_case))

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                segment_ids: torch.Tensor,     # (b, seq)
                ng_token_mask: torch.Tensor,   # (b, seq, case, seq)
                target: torch.Tensor,          # (b, seq, case, seq)
                pre_output: torch.Tensor,      # (b, seq, case, seq)
                ) -> Tuple[torch.Tensor, ...]:  # (), (b, seq, case, seq)
        batch_size, sequence_len = input_ids.size()
        mask = get_mask(attention_mask, ng_token_mask)
        rel_weights = self.output_aggr(pre_output, mask)
        # (b, seq, hid)
        sequence_output, _ = self.bert(input_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=segment_ids,
                                       rel_weights=rel_weights)

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

    @staticmethod
    def _hard_output_aggr(pre_output: torch.Tensor,  # (b, seq, case, seq)
                          mask: torch.Tensor,  # (b, seq, case, seq)
                          ) -> torch.Tensor:  # (b, seq, 1+case*2, seq)
        batch_size, seq_len, num_case, _ = pre_output.size()
        device = pre_output.device

        eye = torch.eye(seq_len, dtype=torch.bool, device=device)  # (seq)
        pre_prediction = eye[pre_output.argmax(dim=3)] & mask  # (b, seq, case, seq)
        neg_pre_prediction = (~pre_prediction).float() * -1024.0  # (b, seq, case, seq)
        bi_prediction = torch.cat([
            torch.full((batch_size, seq_len, 1, seq_len), -256.0, device=device),
            neg_pre_prediction,
            neg_pre_prediction.transpose(1, 3)
            ], dim=2)  # (b, seq, 1+case*2, seq)
        rel_weights = bi_prediction.softmax(dim=2)  # (b, seq, 1+case*2, seq)
        return rel_weights

    @staticmethod
    def _hard2_output_aggr(pre_output: torch.Tensor,  # (b, seq, case, seq)
                           mask: torch.Tensor,  # (b, seq, case, seq)
                           ) -> torch.Tensor:  # (b, seq, 1+case*2, seq)
        batch_size, seq_len, num_case, _ = pre_output.size()
        device = pre_output.device

        eye = torch.eye(seq_len, dtype=torch.bool, device=device)  # (seq)
        pre_prediction = eye[pre_output.argmax(dim=3)] & mask  # (b, seq, case, seq)
        neg_pre_prediction = (~pre_prediction).float() * -1024.0  # (b, seq, case, seq)
        bi_prediction = torch.cat([
            torch.full((batch_size, seq_len, 1, seq_len), -256.0, device=device),
            neg_pre_prediction,
            neg_pre_prediction.transpose(1, 3)
            ], dim=2)  # (b, seq, 1+case*2, seq)
        eye = torch.eye(1 + num_case * 2, dtype=torch.bool, device=device)
        rel_weights = eye[bi_prediction.argmax(dim=2)].transpose(2, 3).float()  # (b, seq, 1+case*2, seq)
        return rel_weights

    @staticmethod
    def _soft_output_aggr(pre_output: torch.Tensor,  # (b, seq, case, seq)
                          *_
                          ) -> torch.Tensor:  # (b, seq, 1+case*2, seq)
        """前段の予測結果をソフトに与える"""
        batch_size, seq_len, num_case, _ = pre_output.size()
        device = pre_output.device

        bi_prediction = torch.cat([
            torch.full((batch_size, seq_len, 1, seq_len), -256.0, device=device),
            pre_output,
            pre_output.transpose(1, 3)
        ], dim=2)  # (b, seq, 1+case*2, seq)
        rel_weights = bi_prediction.softmax(dim=2)  # (b, seq, 1+case*2, seq)
        return rel_weights

    @staticmethod
    def _confidence_output_aggr(pre_output: torch.Tensor,  # (b, seq, case, seq)
                                mask: torch.Tensor,  # (b, seq, case, seq)
                                ) -> torch.Tensor:  # (b, seq, 1+case*2, seq)
        """前段の予測結果を confidence で重み付けして与える"""
        batch_size, seq_len, num_case, _ = pre_output.size()
        device = pre_output.device

        eye = torch.eye(seq_len, dtype=torch.bool, device=device)  # (seq)
        hard_output = eye[pre_output.argmax(dim=3)] & mask  # (b, seq, case, seq)
        confidence = pre_output.softmax(dim=3)  # (b, seq, case, seq)
        weighted_hard_output = hard_output * confidence

        bi_weighted_hard_output = torch.cat([
            weighted_hard_output,
            weighted_hard_output.transpose(1, 3)
        ], dim=2)  # (b, seq, case*2, seq)

        bi_weighted_hard_output_sum = bi_weighted_hard_output.sum(dim=2, keepdim=True)  # (b, seq, 1, seq)
        bi_weighted_hard_output_sum[bi_weighted_hard_output_sum.gt(1)] = 1  # clipping

        rel_weights = torch.cat([
            torch.tensor(1.0) - bi_weighted_hard_output_sum,
            bi_weighted_hard_output,
        ], dim=2)

        return rel_weights
