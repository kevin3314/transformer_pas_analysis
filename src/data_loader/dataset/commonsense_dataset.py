import logging
from typing import List, Optional, NamedTuple, Generator

import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import BertTokenizer


class CommonsenseExample(NamedTuple):
    former: str
    latter: str
    label: bool


class InputFeatures(NamedTuple):
    tokens: List[str]
    input_ids: List[int]
    input_mask: List[int]
    segment_ids: List[int]
    label: bool


class CommonsenseDataset(Dataset):
    def __init__(self,
                 path: str,
                 max_seq_length: int,
                 bert_model: str,
                 logger=None,
                 ) -> None:
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=False, tokenize_chinese_chars=False)
        self.logger = logger if logger else logging.getLogger(__file__)
        self.features: List[InputFeatures] = []
        for example in tqdm(self._read_csv(path), desc='reading commonsense dataset'):
            feature = self._convert_example_to_feature(example, max_seq_length)
            if feature is None:
                continue
            self.features.append(feature)

    @staticmethod
    def _read_csv(path: str) -> Generator[CommonsenseExample]:
        with open(path) as f:
            for line in f:
                label, string = line.strip().split(',')
                former_string, latter_string = string.split('@')
                yield CommonsenseExample(former_string, latter_string, bool(int(label)))

    def _convert_example_to_feature(self,
                                    example: CommonsenseExample,
                                    max_seq_length: int) -> Optional[InputFeatures]:
        """Loads a data file into a list of `InputBatch`s."""

        former_tokens: List[str] = self.tokenizer.tokenize(example.former)
        latter_tokens: List[str] = self.tokenizer.tokenize(example.latter)
        # ignore too long document
        if 1 + len(former_tokens) + 1 + len(latter_tokens) > max_seq_length:
            return None

        tokens: List[str] = []
        segment_ids: List[int] = []

        # cls
        tokens.append(self.tokenizer.cls_token)
        segment_ids.append(0)
        # sentence A
        tokens += former_tokens
        segment_ids += [0] * len(former_tokens)
        # sep
        tokens.append(self.tokenizer.sep_token)
        segment_ids.append(0)
        # sentence B
        tokens += latter_tokens
        segment_ids += [1] * len(latter_tokens)
        # sep
        tokens.append(self.tokenizer.sep_token)
        segment_ids.append(1)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has True for real tokens and False for padding tokens. Only real tokens are attended to.
        input_mask = [True] * len(input_ids)

        # Zero-pad up to the sequence length.
        pad_len = max_seq_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * pad_len
        input_mask += [False] * pad_len
        segment_ids += [0] * pad_len

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        feature = InputFeatures(
            tokens=tokens,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label=example.label,
        )

        return feature

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx) -> tuple:
        feature = self.features[idx]
        input_ids = np.array(feature.input_ids)      # (seq)
        input_mask = np.array(feature.input_mask)    # (seq)
        segment_ids = np.array(feature.segment_ids)  # (seq)
        label = np.array(feature.label)              # ()
        return input_ids, input_mask, segment_ids, label
