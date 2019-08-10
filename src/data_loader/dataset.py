import os
import logging
from typing import List, Dict, Optional
from pathlib import Path
from collections import OrderedDict

import numpy as np
from torch.utils.data import Dataset
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertConfig
from pyknp import Tag

from kwdlc_reader import KWDLCReader, Document


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class PasExample:
    """A single training/test example for pas analysis."""

    def __init__(self,
                 words: List[str],
                 arguments_set: List[Dict[str, Optional[str]]],
                 ng_arg_ids_set: List[List[int]],
                 dtids: List[int],
                 ddeps: List[int],
                 comment: Optional[str]
                 ) -> None:
        self.words = words
        self.arguments_set = arguments_set
        self.ng_arg_ids_set = ng_arg_ids_set
        self.dtids = dtids
        self.ddeps = ddeps
        self.comment = comment

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f'word: {" ".join(self.words)}, arguments: {" ".join(args.__repr__() for args in self.arguments_set)}'


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
                 path: str,
                 max_seq_length: int,
                 cases: List[str],
                 exophors: List[str],
                 coreference: bool,
                 training: bool,
                 bert_model: str) -> None:
        self.pas_examples = []
        self.reader = KWDLCReader(Path(path),
                                  target_cases=cases,
                                  target_corefs=['=', '=構', '=≒'] if coreference else [],
                                  target_exophors=exophors)
        self.special_tokens = self.reader.target_exophors + ['NULL'] + (['NA'] if coreference else [])
        for document in self.reader.process_all_documents():
            self.pas_examples.append(self._read_pas_examples(document, training, coreference))
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=False)
        bert_config = BertConfig.from_json_file(os.path.join(bert_model, 'bert_config.json'))
        self.features = self._convert_examples_to_features(self.pas_examples,
                                                           max_seq_length=max_seq_length,
                                                           vocab_size=bert_config.vocab_size,
                                                           is_training=training,
                                                           num_case=len(self.reader.target_cases),
                                                           coreference=coreference)
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

    @staticmethod
    def _read_pas_examples(document: Document,
                           is_training: bool,
                           coreference: bool
                           ) -> PasExample:
        """Read a file into a list of PasExample."""

        cases = document.target_cases
        comment = f'# A-ID:{document.doc_id}'
        words = []
        dtids = []
        ddeps = []
        arguments_set = []
        ng_arg_ids_set = []
        dmid = 0
        non_head_dmids = []
        for sentence in document:
            for tag in sentence.tag_list():
                for idx, mrph in enumerate(tag.mrph_list()):
                    if '<内容語>' not in mrph.fstring and idx > 0:
                        non_head_dmids.append(document.mrph2dmid[mrph])
            sentence_tail_dmid = document.mrph2dmid[sentence.mrph_list()[-1]]
            dmid2pred: Dict[int, Tag] = {pas.dmid: pas.predicate for pas in document.pas_list()}
            for tag in sentence.tag_list():
                for mrph in tag.mrph_list():
                    words.append(mrph.midasi)
                    dtids.append(document.tag2dtid[tag])
                    ddeps.append(document.tag2dtid[tag.parent] if tag.parent is not None else -1)
                    if '<用言:' in tag.fstring \
                            and '<省略解析なし>' not in tag.fstring \
                            and '<内容語>' in mrph.fstring:
                        arguments: Dict[str, str] = OrderedDict()
                        for case in cases:
                            if dmid in dmid2pred:
                                case2args = document.get_arguments(dmid2pred[dmid], relax=True)
                                if case not in case2args:
                                    arguments[case] = 'NULL'
                                    continue
                                arg = case2args[case][0]  # use first argument now
                                # exophor
                                if arg.dep_type == 'exo':
                                    arguments[case] = arg.midasi
                                # overt
                                elif arg.dep_type == 'overt':
                                    arguments[case] = f'{arg.dmid + 1}%C'
                                # normal
                                else:
                                    arguments[case] = str(arg.dmid + 1)
                            else:
                                arguments[case] = 'NULL'
                        ng_arg_ids = non_head_dmids + list(range(sentence_tail_dmid + 1, len(document.mrph2dmid)))
                        ng_arg_ids = [x + 1 for x in ng_arg_ids]  # 1 origin
                    else:
                        arguments = OrderedDict({case: None} for case in cases)
                        ng_arg_ids = []

                    # TODO: coreference
                    if coreference:
                        pass

                    arguments_set.append(arguments)
                    ng_arg_ids_set.append(ng_arg_ids)
                    dmid += 1

        return PasExample(words, arguments_set, ng_arg_ids_set, dtids, ddeps, comment)

    def _convert_examples_to_features(self,
                                      examples: List[PasExample],
                                      max_seq_length: int,
                                      vocab_size: int,
                                      is_training: bool,
                                      num_case: int,
                                      coreference: bool):
        """Loads a data file into a list of `InputBatch`s."""

        num_expand_vocab = len(self.special_tokens)
        features = []
        num_case_w_coreference = num_case + 1 if coreference else num_case

        # document loop
        for example_index, example in enumerate(examples):

            all_tokens, tok_to_orig_index, orig_to_tok_index = self._get_tokenized_tokens(example.words)
            # ignore too long document
            if len(all_tokens) > max_seq_length - num_expand_vocab:
                continue

            tokens: List[str] = []
            segment_ids: List[int] = []
            arguments_set: List[List[int]] = []
            ng_arg_ids_set: List[List[int]] = []
            deps: List[List[int]] = []

            # subword loop
            for token, orig_index in zip(all_tokens, tok_to_orig_index):
                tokens.append(token)
                segment_ids.append(0)

                # subsequent subword or [CLS] token or [SEP] token
                if token.startswith("##") or orig_index is None:
                    arguments_set.append([-1] * num_case_w_coreference)
                    ng_arg_ids_set.append([])
                    deps.append([0] * max_seq_length)
                    continue

                if is_training is False:
                    arguments_set.append([-1] * num_case_w_coreference)
                else:
                    arguments: List[int] = []
                    for case, argument in example.arguments_set[orig_index].items():
                        if argument is None or "%C" in argument:
                            # none or overt
                            argument_index = -1
                        elif argument.isdigit():
                            if case != '=' and int(argument) in example.ng_arg_ids_set[orig_index]:
                                # ng_arg_id (except for coreference resolution)
                                logger.debug("ng_arg_id: {} {} {}".format(example.comment, token, argument))
                                argument_index = -1
                            else:
                                # normal
                                argument_index = orig_to_tok_index[int(argument) - 1]
                        else:
                            # special token
                            argument_index = max_seq_length - num_expand_vocab + self.special_tokens.index(argument)
                        arguments.append(argument_index)

                    arguments_set.append(arguments)

                ddep = example.ddeps[orig_index]
                deps.append([(0 if idx is None or ddep != example.dtids[idx] else 1) for idx in tok_to_orig_index])
                deps[-1] += [0] * (max_seq_length - len(tok_to_orig_index))

                # 0 origin
                ng_arg_ids_set.append(
                    [0] +
                    [orig_to_tok_index[ng_arg_id - 1] for ng_arg_id in example.ng_arg_ids_set[orig_index]] +
                    [len(all_tokens) - 1]
                )

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length (except for special tokens).
            while len(input_ids) < max_seq_length - num_expand_vocab:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                arguments_set.append([-1] * num_case_w_coreference)
                ng_arg_ids_set.append([])
                deps.append([0] * max_seq_length)

            # add special tokens
            for i in range(num_expand_vocab):
                input_ids.append(vocab_size + i)
                input_mask.append(1)
                segment_ids.append(0)
                arguments_set.append([-1] * num_case_w_coreference)
                ng_arg_ids_set.append([])
                deps.append([0] * max_seq_length)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(arguments_set) == max_seq_length
            assert len(ng_arg_ids_set) == max_seq_length
            assert len(deps) == max_seq_length

            if example_index == 0:
                logger.info('*** Example ***')
                logger.info(f'example_index: {example_index}')
                logger.info(f'tokens: {" ".join(tokens)}')
                logger.info(f'input_ids: {" ".join(str(x) for x in input_ids)}')
                logger.info(f'input_mask: {" ".join(str(x) for x in input_mask)}')
                logger.info(f'segment_ids: {" ".join(str(x) for x in segment_ids)}')
                logger.info(f'arguments: '
                            f'{" ".join(",".join(str(arg) for arg in arguments) for arguments in arguments_set)}')
                logger.info(f'ng_arg_ids_set: '
                            f'{" ".join(",".join(str(x) for x in ng_arg_ids) for ng_arg_ids in ng_arg_ids_set)}')
                logger.info(f'deps: '
                            f'{" ".join("".join(str(x) for x in dep) for dep in deps)}')

            features.append(
                InputFeatures(
                    tokens=tokens,
                    orig_to_tok_index=orig_to_tok_index,
                    tok_to_orig_index=tok_to_orig_index,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    arguments_set=arguments_set,
                    ng_arg_mask=[[0 if x in ng_arg_ids else 1 for x in range(max_seq_length)] for ng_arg_ids in
                                 ng_arg_ids_set],
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
