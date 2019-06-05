import torch
import torch.nn as nn
# import torch.nn.functional as F
from pytorch_pretrained_bert.modeling import BertModel

from base import BaseModel


class BaselineModel(BaseModel):
    def __init__(self,
                 bert_model: str,
                 parsing_algorithm: str,
                 num_case: int,
                 arc_representation_dim: int) -> None:
        super().__init__()

        self.bert = BertModel.from_pretrained(bert_model)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.num_case = num_case
        self.parsing_algorithm = parsing_algorithm

        bert_hidden_size = self.bert.config.hidden_size
        # Deep Biaffine [Dozart+ 17]
        if self.parsing_algorithm == "biaffine":
            self.head_arc_linear = nn.Linear(bert_hidden_size, arc_representation_dim)
            self.child_arc_linear = nn.Linear(bert_hidden_size, arc_representation_dim)

            self.arc_W = nn.Parameter(torch.randn((arc_representation_dim, arc_representation_dim), requires_grad=True))
            self.arc_b = nn.Linear(arc_representation_dim, 1, bias=False)

        # head selection [Zhang+ 16]
        elif self.parsing_algorithm == "zhang":
            self.W_a = nn.Linear(bert_hidden_size, bert_hidden_size * num_case)
            self.U_a = nn.Linear(bert_hidden_size, bert_hidden_size * num_case)
            self.v_a = nn.Linear(bert_hidden_size, 1, bias=False)

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                ng_arg_mask: torch.Tensor,     # (b, seq, seq)
                ) -> torch.Tensor:             # (b, seq, case, seq)
        batch_size, sequence_len = input_ids.size()
        # (b, seq, hid)
        sequence_output, _ = self.bert(input_ids,
                                       attention_mask=attention_mask,
                                       output_all_encoded_layers=False)

        g_logits = torch.Tensor()
        if self.parsing_algorithm == "biaffine":
            h_i = torch.relu(self.child_arc_linear(sequence_output))  # (b, seq, arc)
            h_j = torch.relu(self.head_arc_linear(sequence_output))   # (b, seq, arc)

            inter = torch.matmul(h_j, self.arc_W)  # (b, seq, arc)
            g_logits = torch.matmul(inter, h_i.transpose(1, 2))  # (b, seq, seq)
            bias = self.arc_b(h_j).squeeze(2).unsqueeze(1)  # (b, 1, seq)
            g_logits = g_logits + bias  # (b, seq, seq)

        elif self.parsing_algorithm == "zhang":
            h_i = self.W_a(sequence_output)  # (b, seq, case*hid)
            h_j = self.U_a(sequence_output)  # (b, seq, case*hid)
            h_i = h_i.view(batch_size, sequence_len, self.num_case, -1)  # (b, seq, case, hid)
            h_j = h_j.view(batch_size, sequence_len, self.num_case, -1)  # (b, seq, case, hid)
            g_logits = self.v_a(torch.tanh(h_i.unsqueeze(1) + h_j.unsqueeze(2))).squeeze(-1)  # (b, seq, seq, case)

        g_logits = g_logits.transpose(2, 3).contiguous()  # (b, seq, case, seq)

        extended_attention_mask = attention_mask.view(batch_size, 1, 1, sequence_len)  # (b, 1, 1, seq)
        ng_arg_mask = ng_arg_mask.unsqueeze(2)  # (b, seq, 1, seq)
        mask = extended_attention_mask & ng_arg_mask  # (b, seq, 1, seq)
        mask = mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility

        g_logits += (1.0 - mask) * -1024.0  # (b, seq, case, seq)

        return g_logits  # (b, seq, case, seq)

    def expand_vocab(self, num_expand_vocab):
        """Add special tokens to vocab."""

        bert_word_embeddings = self.bert.embeddings.word_embeddings

        old_word_embeddings_numpy = bert_word_embeddings.weight.detach().numpy()
        vocab_size = bert_word_embeddings.weight.shape[0]
        new_word_embeddings = nn.Embedding(vocab_size + num_expand_vocab, bert_word_embeddings.weight.shape[1])
        new_word_embeddings_numpy = new_word_embeddings.weight.detach().numpy()
        new_word_embeddings_numpy[:vocab_size, :] = old_word_embeddings_numpy
        new_word_embeddings.from_pretrained(torch.Tensor(new_word_embeddings_numpy), freeze=False)
        self.bert.embeddings.word_embeddings = new_word_embeddings


class BaseAsymModel(BaseModel):
    def __init__(self,
                 bert_model: str,
                 parsing_algorithm: str,
                 num_case: int,
                 arc_representation_dim: int) -> None:
        super().__init__()

        self.bert = BertModel.from_pretrained(bert_model)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.num_case = num_case
        bert_hidden_size = self.bert.config.hidden_size

        self.l_pred = nn.Linear(bert_hidden_size, bert_hidden_size)
        self.l_cases = nn.Linear(bert_hidden_size, num_case * bert_hidden_size)
        self.l_mid = nn.Linear(2 * bert_hidden_size, bert_hidden_size)
        self.l_out = nn.Linear(bert_hidden_size, 1, bias=False)

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                ng_arg_mask: torch.Tensor,     # (b, seq, seq)
                ) -> torch.Tensor:             # (b, seq, case, seq)
        # (b, seq, hid)
        sequence_output, _ = self.bert(input_ids,
                                       attention_mask=attention_mask,
                                       output_all_encoded_layers=False)
        batch_size, sequence_len, hidden_dim = sequence_output.size()

        h_i = self.l_pred(sequence_output)  # (b, seq, hid)
        h_j = self.l_cases(sequence_output)  # (b, seq, case*hid)
        # -> (b, seq, 1, 1, hid) -> (b, seq, seq, case, hid)
        h_i = h_i.view(batch_size, sequence_len, 1, 1, hidden_dim).expand(-1, -1, sequence_len, self.num_case, -1)
        # -> (b, 1, seq, case, hid) -> (b, seq, seq, case, hid)
        h_j = h_j.view(batch_size, 1, sequence_len, self.num_case, hidden_dim).expand(-1, sequence_len, -1, -1, -1)
        g_logits = self.l_mid(torch.tanh(torch.cat([h_i, h_j], dim=4)))  # (b, seq, seq, case, hid)
        g_logits = self.l_out(g_logits).squeeze(4)  # -> (b, seq, seq, case, 1) -> (b, seq, seq, case)

        g_logits = g_logits.transpose(2, 3).contiguous()  # (b, seq, case, seq)

        extended_attention_mask = attention_mask.view(batch_size, 1, 1, sequence_len)  # (b, 1, 1, seq)
        ng_arg_mask = ng_arg_mask.unsqueeze(2)  # (b, seq, 1, seq)
        mask = extended_attention_mask & ng_arg_mask  # (b, seq, 1, seq)
        mask = mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility

        g_logits += (1.0 - mask) * -1024.0  # (b, seq, case, seq)

        return g_logits  # (b, seq, case, seq)

    def expand_vocab(self, num_expand_vocab):
        """Add special tokens to vocab."""

        bert_word_embeddings = self.bert.embeddings.word_embeddings

        old_word_embeddings_numpy = bert_word_embeddings.weight.detach().numpy()
        vocab_size = bert_word_embeddings.weight.shape[0]
        new_word_embeddings = nn.Embedding(vocab_size + num_expand_vocab, bert_word_embeddings.weight.shape[1])
        new_word_embeddings_numpy = new_word_embeddings.weight.detach().numpy()
        new_word_embeddings_numpy[:vocab_size, :] = old_word_embeddings_numpy
        new_word_embeddings.from_pretrained(torch.Tensor(new_word_embeddings_numpy), freeze=False)
        self.bert.embeddings.word_embeddings = new_word_embeddings
