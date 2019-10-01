import logging
from typing import List, Dict, Optional
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
from pytorch_transformers import BertConfig, BertTokenizer

from kwdlc_reader import KWDLCDirectoryReader, KWDLCStringReader
from data_loader.dataset.read_example import read_example, PasExample


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputFeatures:
    """A single set of features of data."""

    def __init__(self,
                 tokens: List[str],
                 orig_to_tok_index: List[int],  # use for output
                 tok_to_orig_index: List[Optional[int]],  # use for output
                 input_ids: List[int],  # use for model
                 input_mask: List[int],  # use for model
                 segment_ids: List[int],  # use for model
                 arguments_set: List[List[int]],  # use for model
                 ng_arg_mask: List[List[int]],  # use for model
                 deps: List[List[int]],
                 ):
        self.tokens = tokens
        self.orig_to_tok_index = orig_to_tok_index
        self.tok_to_orig_index = tok_to_orig_index
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.arguments_set = arguments_set
        self.ng_arg_mask = ng_arg_mask
        self.deps = deps


class PASDataset(Dataset):
    def __init__(self,
                 path: Optional[str],
                 max_seq_length: int,
                 cases: List[str],
                 exophors: List[str],
                 coreference: bool,
                 training: bool,
                 bert_model: str,
                 kc: bool = False,
                 knp_string: Optional[str] = None,
                 ) -> None:
        if path is not None:
            self.reader = KWDLCDirectoryReader(Path(path),
                                               target_cases=cases,
                                               target_corefs=['=', '=構', '=≒'] if coreference else [],
                                               target_exophors=exophors,
                                               extract_nes=False)
        else:
            assert knp_string is not None
            self.reader = KWDLCStringReader(knp_string,
                                            target_cases=cases,
                                            target_corefs=['=', '=構', '=≒'] if coreference else [],
                                            target_exophors=exophors,
                                            extract_nes=False)
        special_tokens = self.reader.target_exophors + ['NULL'] + (['NA'] if coreference else [])
        self.num_special_tokens = len(special_tokens)
        self.special_to_index: Dict[str, int] = {token: i + max_seq_length - self.num_special_tokens for i, token
                                                 in enumerate(special_tokens)}
        self.coreference = coreference
        self.examples = [read_example(doc, coreference, kc) for doc in self.reader.process_all_documents()]
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=False, tokenize_chinese_chars=False)
        bert_config = BertConfig.from_json_file(Path(bert_model) / 'bert_config.json')
        self.features = self._convert_examples_to_features(self.examples,
                                                           max_seq_length=max_seq_length,
                                                           # don't use len(self.tokenizer.vocab)
                                                           vocab_size=bert_config.vocab_size,
                                                           num_case=len(self.reader.target_cases))
        self.training = training

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx) -> tuple:
        feature = self.features[idx]
        input_ids = np.array(feature.input_ids)          # (seq)
        input_mask = np.array(feature.input_mask)        # (seq)
        arguments_ids = np.array(feature.arguments_set)  # (seq, case)
        ng_arg_mask = np.array(feature.ng_arg_mask)      # (seq, seq)
        deps = np.array(feature.deps)                    # (seq, seq)
        return input_ids, input_mask, arguments_ids, ng_arg_mask, deps

    def _convert_examples_to_features(self,
                                      examples: List[PasExample],
                                      max_seq_length: int,
                                      vocab_size: int,
                                      num_case: int):
        """Loads a data file into a list of `InputBatch`s."""

        num_expand_vocab = len(self.special_to_index)
        features = []
        num_case_w_coreference = num_case + 1 if self.coreference else num_case

        # document loop
        for example_index, example in enumerate(examples):

            all_tokens, tok_to_orig_index, orig_to_tok_index = self._get_tokenized_tokens(example.words)
            # ignore too long document
            if len(all_tokens) > max_seq_length - num_expand_vocab:
                continue

            tokens: List[str] = []
            segment_ids: List[int] = []
            arguments_set: List[List[int]] = []
            arg_candidates_set: List[List[int]] = []
            deps: List[List[int]] = []

            # subword loop
            for token, orig_index in zip(all_tokens, tok_to_orig_index):
                tokens.append(token)
                segment_ids.append(0)

                # subsequent subword or [CLS] token or [SEP] token
                if token.startswith("##") or orig_index is None:
                    arguments_set.append([-1] * num_case_w_coreference)
                    arg_candidates_set.append([])
                    deps.append([0] * max_seq_length)
                    continue

                arguments: List[int] = []
                for case, argument in example.arguments_set[orig_index].items():
                    if argument is None or '%C' in argument:
                        # none or overt
                        argument_index = -1
                    elif argument.isdigit():
                        if case != '=' and int(argument) not in example.arg_candidates_set[orig_index]:
                            # ignore an argument which is not candidate (except for coreference resolution)
                            logger.debug(f'argument: {argument} of {token} is not in candidates and ignored')
                            argument_index = -1
                        else:
                            # normal
                            argument_index = orig_to_tok_index[int(argument)]
                    else:
                        # special token
                        argument_index = self.special_to_index[argument]
                    arguments.append(argument_index)
                arguments_set.append(arguments)

                ddep = example.ddeps[orig_index]
                deps.append([(0 if idx is None or ddep != example.dtids[idx] else 1) for idx in tok_to_orig_index])
                deps[-1] += [0] * (max_seq_length - len(tok_to_orig_index))

                if any(idx != -1 for idx in arguments):
                    arg_candidates = [orig_to_tok_index[dmid] for dmid in example.arg_candidates_set[orig_index]] + \
                                     list(self.special_to_index.values())
                else:
                    arg_candidates = []
                arg_candidates_set.append(arg_candidates)

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length (except for special tokens).
            while len(input_ids) < max_seq_length - num_expand_vocab:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                arguments_set.append([-1] * num_case_w_coreference)
                arg_candidates_set.append([])
                deps.append([0] * max_seq_length)

            # add special tokens
            for i in range(num_expand_vocab):
                input_ids.append(vocab_size + i)
                input_mask.append(1)
                segment_ids.append(0)
                arguments_set.append([-1] * num_case_w_coreference)
                arg_candidates_set.append([])
                deps.append([0] * max_seq_length)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(arguments_set) == max_seq_length
            assert len(arg_candidates_set) == max_seq_length
            assert len(deps) == max_seq_length

            if example_index == 0:
                logger.info('*** Example ***')
                logger.info(f'tokens: {" ".join(tokens)}')
                logger.info(f'input_ids: {" ".join(str(x) for x in input_ids)}')

            features.append(
                InputFeatures(
                    tokens=tokens,
                    orig_to_tok_index=orig_to_tok_index,
                    tok_to_orig_index=tok_to_orig_index,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    arguments_set=arguments_set,
                    ng_arg_mask=[[int(x in candidates) for x in range(max_seq_length)]
                                 for candidates in arg_candidates_set],  # 0 -> mask, 1 -> keep
                    deps=deps,
                )
            )

        return features

    def _get_tokenized_tokens(self, words: List[str]):
        all_tokens = []
        tok_to_orig_index: List[Optional[int]] = []
        orig_to_tok_index: List[int] = []

        all_tokens.append('[CLS]')
        tok_to_orig_index.append(None)  # There's no original token corresponding to [CLS] token

        for i, word in enumerate(words):
            orig_to_tok_index.append(len(all_tokens))  # assign head subword
            sub_tokens = self.tokenizer.tokenize(word)
            for sub_token in sub_tokens:
                all_tokens.append(sub_token)
                tok_to_orig_index.append(i)

        all_tokens.append('[SEP]')
        tok_to_orig_index.append(None)  # There's no original token corresponding to [SEP] token

        return all_tokens, tok_to_orig_index, orig_to_tok_index
