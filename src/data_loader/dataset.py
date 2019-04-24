import os
from typing import NamedTuple, List, Tuple, Dict, Optional
from collections import defaultdict
from glob import glob

import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset


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
    def __init__(self, path: str, is_training: bool, num_case: int, cases: List[str], coreference=False):
        self.pas_examples = self._read_pas_examples(path, is_training, num_case, cases, coreference)

    def __len__(self) -> int:
        return len(self.pas_examples)

    def __getitem__(self, idx):
        return self.pas_examples[idx]

    @staticmethod
    def _read_pas_examples(input_file: str, is_training: bool, num_case: int, cases: List[str], coreference: bool):
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
                    arguments = [None for _ in range(num_case)]
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
                        arguments.append(None)
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