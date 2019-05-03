import os
import logging
from typing import NamedTuple, List, Tuple, Dict, Optional
# from collections import defaultdict

import numpy as np
from torch.utils.data import Dataset
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertConfig

from data_loader.input_features import InputFeatures


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class PasExample:
    """A single training/test example for pas analysis."""

    def __init__(self,
                 example_id,
                 words,
                 lines,
                 arguments_set=None,
                 ng_arg_ids_set=None,
                 comment=None):
        self.example_id: int = example_id
        self.words: List[str] = words
        self.lines: List[str] = lines
        self.arguments_set: List[List[Optional[str]]] = arguments_set
        self.ng_arg_ids_set: List[List[int]] = ng_arg_ids_set
        self.comment: str = comment

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f'id: {self.example_id}, word: {" ".join(self.words)}, ' \
            f'arguments: {" ".join(args.__repr__() for args in self.arguments_set)}'


class PASDataset(Dataset):
    def __init__(self,
                 path: str,
                 max_seq_length: int,
                 cases: List[str],
                 special_tokens: List[str],
                 coreference: bool,
                 training: bool,
                 bert_model: str) -> None:
        self.pas_examples = self._read_pas_examples(path, training, cases, coreference)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=False)
        bert_config = BertConfig.from_json_file(os.path.join(bert_model, 'bert_config.json'))
        self.features = self._convert_examples_to_features(self.pas_examples,
                                                           max_seq_length=max_seq_length,
                                                           vocab_size=bert_config.vocab_size,
                                                           is_training=training,
                                                           num_case=len(cases),
                                                           num_expand_vocab=len(special_tokens),
                                                           special_tokens=special_tokens,
                                                           coreference=coreference)
        self.training = training

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        input_ids = np.array(feature.input_ids)            # (seq)
        input_mask = np.array(feature.input_mask)          # (seq)
        segment_ids = np.array(feature.segment_ids)        # (seq)
        arguments_set = np.array(feature.arguments_set)    # (seq, case)
        example_index = np.array(idx)                      # ()
        ng_arg_ids_set = np.array(feature.ng_arg_ids_set)  # (seq, seq)
        if self.training:
            return input_ids, input_mask, segment_ids, arguments_set, ng_arg_ids_set
        else:
            return input_ids, input_mask, segment_ids, example_index, ng_arg_ids_set

    @staticmethod
    def _read_pas_examples(input_file: str, is_training: bool, cases: List[str], coreference: bool):
        """Read a file into a list of PasExample."""

        examples: List[PasExample] = []
        example_id: int = 0
        comment: Optional[str] = None
        with open(input_file, "r") as reader:
            # 9       関わる  ガ:10,ヲ:NULL,ニ:7%C,ガ２:NULL _ _ _
            words, arguments_set, ng_arg_ids_set, lines = [], [], [], []
            while True:
                line = reader.readline()
                if not line:
                    break
                line = line.strip()
                if line.startswith("#"):
                    comment = line
                    continue
                if not line:
                    example = PasExample(
                        example_id,
                        words,
                        lines,
                        arguments_set=arguments_set,
                        ng_arg_ids_set=ng_arg_ids_set,
                        comment=comment)
                    examples.append(example)

                    example_id += 1
                    words, arguments_set, ng_arg_ids_set, lines = [], [], [], []
                    comment = None
                    continue

                items = line.split("\t")
                word = items[1]
                argument_string = items[5]
                ng_arg_string = items[8]
                if argument_string == "_":
                    arguments = [-1] * len(cases)
                    ng_arg_ids = []
                else:
                    arguments = []
                    for i, argument in enumerate(argument_string.split(",")):
                        case, argument_index = argument.split(":", maxsplit=1)
                        assert cases[i] == case
                        arguments.append(argument_index)
                    if ng_arg_string != "_":
                        ng_arg_ids = [int(ng_arg_id) for ng_arg_id in ng_arg_string.split("/")]
                    else:
                        ng_arg_ids = []

                coreference_string = None
                if coreference is True:
                    coreference_string = items[6]
                    if coreference_string == "_":
                        arguments.append(-1)
                    elif coreference_string == "NA":
                        arguments.append("NA")
                    else:
                        arguments.append(coreference_string)

                if is_training is False:
                    if argument_string != "_":
                        # ガ:55%C,ヲ:57,ニ:NULL,ガ２:NULL
                        arguments = []
                        for i, argument in enumerate(argument_string.split(",")):
                            case, argument_index = argument.split(":", 1)
                            if "%C" not in argument_index:
                                argument_index = "MASKED"
                            arguments.append("{}:{}".format(case, argument_index))
                        items[5] = ",".join(arguments)

                    if coreference is True:
                        if coreference_string != "_":
                            items[6] = "MASKED"
                    line = "\t".join(items)

                words.append(word)
                arguments_set.append(arguments)
                ng_arg_ids_set.append(ng_arg_ids)
                lines.append(line)

        return examples

    def _convert_examples_to_features(self, examples: List[PasExample], max_seq_length: int, vocab_size: int,
                                      is_training: bool, num_case: int, num_expand_vocab: int,
                                      special_tokens: List[str], coreference: bool):
        """Loads a data file into a list of `InputBatch`s."""

        unique_id = 1000000000

        features = []
        num_case_w_coreference = num_case
        if coreference is True:
            num_case_w_coreference += 1

        for example_index, example in enumerate(examples):
            # The -3 accounts for [CLS], [SEP], ROOT
            # max_tokens_for_doc = max_seq_length - 3

            tokens = []
            segment_ids = []
            arguments_set = []
            ng_arg_ids_set = []
            # token_tag_indices = defaultdict(list)

            all_tokens, tok_to_orig_index, orig_to_tok_index = self._get_tokenized_tokens(example.words)

            # CLS
            tokens.append("[CLS]")
            segment_ids.append(0)
            arguments_set.append([-1] * num_case_w_coreference)
            ng_arg_ids_set.append([])

            for j, token in enumerate(all_tokens):
                tokens.append(token)

                arguments = []
                if is_training is False or token.startswith("##"):
                    arguments_set.append([-1] * num_case_w_coreference)
                else:
                    for argument_index in example.arguments_set[tok_to_orig_index[j]]:
                        # -1 or normal
                        if isinstance(argument_index, int) or argument_index.isdigit() is True:
                            # normal
                            if isinstance(argument_index, int) is False:
                                argument_index = int(argument_index)
                                argument_index = orig_to_tok_index[argument_index - 1] + 1
                            else:
                                argument_index = -1
                        else:
                            if "%C" in argument_index:
                                argument_index = -1
                            # special token
                            else:
                                argument_index = max_seq_length - num_expand_vocab + special_tokens.index(
                                    argument_index)
                        arguments.append(argument_index)

                    arguments_set.append(arguments)

                # ng_arg_ids
                if token.startswith("##") is True:
                    ng_arg_ids_set.append([])
                else:
                    ng_arg_ids_set.append([orig_to_tok_index[ng_arg_id - 1] + 1 for ng_arg_id in
                                           example.ng_arg_ids_set[tok_to_orig_index[j]]])

                segment_ids.append(0)

            # SEP
            tokens.append("[SEP]")
            arguments_set.append([-1] * num_case_w_coreference)
            ng_arg_ids_set.append([])
            segment_ids.append(0)

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length (except for ROOT).
            while len(input_ids) < max_seq_length - num_expand_vocab:
                input_ids.append(0)
                input_mask.append(0)
                arguments_set.append([-1] * num_case_w_coreference)
                ng_arg_ids_set.append([])
                segment_ids.append(0)

            for i in range(num_expand_vocab):
                input_ids.append(vocab_size + i)
                input_mask.append(1)
                arguments_set.append([-1] * num_case_w_coreference)
                ng_arg_ids_set.append([])
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(arguments_set) == max_seq_length
            assert len(ng_arg_ids_set) == max_seq_length

            if example_index < 20:
                logger.info("*** Example ***")
                logger.info("unique_id: %s" % unique_id)
                logger.info("example_index: %s" % example_index)
                logger.info("tokens: %s" % " ".join(
                    [x for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logger.info(
                    "arguments: %s" % " ".join(
                        [",".join([str(arg) for arg in arguments]) for arguments in arguments_set]))
                logger.info(
                    "ng_arg_ids_set: %s" % " ".join(
                        [",".join([str(x) for x in ng_arg_ids]) for ng_arg_ids in ng_arg_ids_set]))
                # for namespace in token_tag_indices:
                #     logger.info(
                #         "%s_tags: %s" % (namespace, " ".join([str(x) for x in token_tag_indices[namespace]])))

            features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    tokens=tokens,
                    orig_to_tok_index=orig_to_tok_index,
                    tok_to_orig_index=tok_to_orig_index,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    arguments_set=arguments_set,
                    ng_arg_ids_set=[[1 if x in ng_arg_ids else 0 for x in range(max_seq_length)] for ng_arg_ids in
                                    ng_arg_ids_set],
                    ))
            unique_id += 1

        return features

    def _get_tokenized_tokens(self, words):
        all_tokens = []
        tok_to_orig_index = []
        orig_to_tok_index = []

        for i, token in enumerate(words):
            orig_to_tok_index.append(len(all_tokens))
            sub_tokens = self.tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                all_tokens.append(sub_token)
                tok_to_orig_index.append(i)

        return all_tokens, tok_to_orig_index, orig_to_tok_index
