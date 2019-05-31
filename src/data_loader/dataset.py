import os
import logging
from typing import NamedTuple, List, Tuple, Dict, Optional

import numpy as np
from torch.utils.data import Dataset
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertConfig


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class PasExample:
    """A single training/test example for pas analysis."""

    def __init__(self,
                 example_id: int,
                 words: List[str],
                 lines: List[str],
                 arguments_set: List[List[Optional[str]]],
                 ng_arg_ids_set: List[List[int]],
                 comment: Optional[str]
                 ) -> None:
        self.example_id = example_id
        self.words = words
        self.lines = lines
        self.arguments_set = arguments_set
        self.ng_arg_ids_set = ng_arg_ids_set
        self.comment = comment

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f'id: {self.example_id}, word: {" ".join(self.words)}, ' \
            f'arguments: {" ".join(args.__repr__() for args in self.arguments_set)}'


class InputFeatures:
    """A single set of features of data."""

    def __init__(self,
                 tokens: List[str],
                 orig_to_tok_index: List[int],  # use for output
                 tok_to_orig_index: List[Optional[int]],  # use for output
                 input_ids: List[int],  # use for model
                 input_mask: List[int],  # use for model
                 segment_ids: List[int],  # use for model
                 arguments_set: List[List[int]] = None,  # use for model
                 ng_arg_mask: List[List[int]] = None,  # use for model
                 ):
        self.tokens = tokens
        self.orig_to_tok_index = orig_to_tok_index
        self.tok_to_orig_index = tok_to_orig_index
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.arguments_set = arguments_set
        self.ng_arg_mask = ng_arg_mask


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

    def __getitem__(self, idx) -> tuple:
        feature = self.features[idx]
        input_ids = np.array(feature.input_ids)          # (seq)
        input_mask = np.array(feature.input_mask)        # (seq)
        arguments_ids = np.array(feature.arguments_set)  # (seq, case)
        ng_arg_mask = np.array(feature.ng_arg_mask)      # (seq, seq)
        return input_ids, input_mask, arguments_ids, ng_arg_mask

    @staticmethod
    def _read_pas_examples(input_file: str, is_training: bool, cases: List[str], coreference: bool) -> List[PasExample]:
        """Read a file into a list of PasExample."""

        examples: List[PasExample] = []
        with open(input_file, "r") as reader:
            # 9       関わる  ガ:10,ヲ:NULL,ニ:7%C,ガ２:NULL _ _ _
            example_id: int = 0
            words, arguments_set, ng_arg_ids_set, lines = [], [], [], []
            comment: Optional[str] = None
            for line in reader:
                line = line.strip()
                if line.startswith("#"):
                    comment = line
                    continue
                if not line:
                    example = PasExample(example_id, words, lines, arguments_set, ng_arg_ids_set, comment)
                    examples.append(example)

                    example_id += 1
                    words, arguments_set, ng_arg_ids_set, lines = [], [], [], []
                    comment = None
                    continue

                items = line.split("\t")
                word = items[1]
                argument_string = items[5]
                coreference_string = items[6] if coreference else None
                ng_arg_string = items[8]
                if argument_string == "_":
                    arguments = [None] * len(cases)
                    ng_arg_ids = []
                else:
                    arguments: List[Optional[str]] = []
                    for i, argument in enumerate(argument_string.split(",")):
                        case, arg = argument.split(":", maxsplit=1)
                        assert cases[i] == case
                        arguments.append(arg)
                    if ng_arg_string != "_":
                        ng_arg_ids = [int(ng_arg_id) for ng_arg_id in ng_arg_string.split("/")]
                    else:
                        ng_arg_ids = []

                if coreference_string is not None:
                    if coreference_string == "_":
                        arguments.append(None)
                    elif coreference_string == "NA":
                        arguments.append("NA")
                    else:
                        arguments.append(coreference_string)

                # mask PAS and coreference labels
                if is_training is False:
                    if argument_string != "_":
                        # ガ:55%C,ヲ:57,ニ:NULL,ガ２:NULL
                        arguments = []
                        for i, argument in enumerate(argument_string.split(",")):
                            case, arg = argument.split(":", maxsplit=1)
                            # don't mask overt case
                            if "%C" not in arg:
                                arg = "MASKED"
                            arguments.append(f"{case}:{arg}")
                        items[5] = ",".join(arguments)

                    if coreference_string is not None:
                        if coreference_string != "_":
                            items[6] = "MASKED"
                    line = "\t".join(items)

                words.append(word)
                arguments_set.append(arguments)
                ng_arg_ids_set.append(ng_arg_ids)  # 1 origin
                lines.append(line)

        return examples

    def _convert_examples_to_features(self, examples: List[PasExample], max_seq_length: int, vocab_size: int,
                                      is_training: bool, num_case: int, num_expand_vocab: int,
                                      special_tokens: List[str], coreference: bool):
        """Loads a data file into a list of `InputBatch`s."""

        features = []
        num_case_w_coreference = num_case + 1 if coreference else num_case

        for example_index, example in enumerate(examples):

            all_tokens, tok_to_orig_index, orig_to_tok_index = self._get_tokenized_tokens(example.words)

            tokens: List[str] = []
            segment_ids: List[int] = []
            arguments_set: List[List[int]] = []
            ng_arg_ids_set: List[List[int]] = []

            for token, orig_index in zip(all_tokens, tok_to_orig_index):
                tokens.append(token)
                segment_ids.append(0)

                # subsequent subword or [CLS] token or [SEP] token
                if token.startswith("##") or orig_index is None:
                    arguments_set.append([-1] * num_case_w_coreference)
                    ng_arg_ids_set.append([])
                    continue

                if is_training is False:
                    arguments_set.append([-1] * num_case_w_coreference)
                else:
                    arguments: List[int] = []
                    for k, argument in enumerate(example.arguments_set[orig_index]):
                        if argument is None or "%C" in argument:
                            # none or overt
                            argument_index = -1
                        elif argument.isdigit():
                            if (coreference is False or (coreference is True and k != num_case_w_coreference - 1)) and \
                                   int(argument) in example.ng_arg_ids_set[orig_index]:
                                # ng_arg_id (except for coreference resolution)
                                logger.debug("ng_arg_id: {} {} {}".format(example.comment, token, argument))
                                argument_index = -1
                            else:
                                # normal
                                argument_index = orig_to_tok_index[int(argument) - 1]
                        else:
                            # special token
                            argument_index = max_seq_length - num_expand_vocab + special_tokens.index(argument)
                        arguments.append(argument_index)

                    arguments_set.append(arguments)

                # 0 origin
                ng_arg_ids_set.append([orig_to_tok_index[ng_arg_id - 1] for ng_arg_id in
                                       example.ng_arg_ids_set[orig_index]])

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

            # add special tokens
            for i in range(num_expand_vocab):
                input_ids.append(vocab_size + i)
                input_mask.append(1)
                segment_ids.append(0)
                arguments_set.append([-1] * num_case_w_coreference)
                ng_arg_ids_set.append([])

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(arguments_set) == max_seq_length
            assert len(ng_arg_ids_set) == max_seq_length

            if example_index < 10:
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
