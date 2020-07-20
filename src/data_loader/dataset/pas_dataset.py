import logging
from typing import List, Dict, Optional, NamedTuple, Tuple
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizer
from kyoto_reader import KyotoReader, Document, ALL_EXOPHORS

from data_loader.dataset.read_example import read_example, PasExample
from utils.constants import TASK_ID


class InputFeatures(NamedTuple):
    tokens: List[str]  # for debug
    orig_to_tok_index: List[int]  # for output
    tok_to_orig_index: List[Optional[int]]  # for output
    input_ids: List[int]  # for training
    input_mask: List[bool]  # for training
    segment_ids: List[int]  # for training
    arguments_set: List[List[List[int]]]  # for training
    overt_mask: List[List[List[int]]]  # for training
    ng_token_mask: List[List[List[bool]]]  # for training
    deps: List[List[int]]  # for training


class PASDataset(Dataset):
    def __init__(self,
                 path: Optional[str],
                 cases: List[str],
                 exophors: List[str],
                 coreference: bool,
                 bridging: bool,
                 dataset_config: dict,
                 training: bool,
                 kc: bool,
                 train_targets: List[str],
                 pas_targets: List[str],
                 knp_string: Optional[str] = None,
                 logger=None,
                 kc_joined_path: Optional[str] = None,
                 ) -> None:
        if path is not None:
            source = Path(path)
        else:
            assert knp_string is not None
            source = knp_string
        self.reader = KyotoReader(source,
                                  target_cases=dataset_config['target_cases'],
                                  target_corefs=dataset_config['target_corefs'],
                                  extract_nes=False)
        self.target_cases: List[str] = [c for c in cases if c in self.reader.target_cases and c != 'ノ']
        self.target_exophors: List[str] = [e for e in exophors if e in ALL_EXOPHORS]
        self.coreference: bool = coreference
        self.bridging: bool = bridging
        self.kc: bool = kc
        self.train_overt: bool = 'overt' in train_targets
        self.train_case: bool = 'case' in train_targets
        self.train_zero: bool = 'zero' in train_targets
        self.pas_targets: List[str] = pas_targets
        self.logger = logger if logger else logging.getLogger(__file__)
        special_tokens = exophors + ['NULL'] + (['NA'] if coreference else [])
        self.special_to_index: Dict[str, int] = {token: dataset_config['max_seq_length'] - i - 1 for i, token
                                                 in enumerate(reversed(special_tokens))}
        self.tokenizer = BertTokenizer.from_pretrained(dataset_config['bert_path'], do_lower_case=False,
                                                       tokenize_chinese_chars=False)
        self.expanded_vocab_size: int = self.tokenizer.vocab_size + len(special_tokens)
        documents = list(self.reader.process_all_documents())
        self.documents: Optional[List[Document]] = documents if not training else None
        self.examples: List[PasExample] = []
        self.features: List[InputFeatures] = []

        if self.kc and not training:
            assert kc_joined_path is not None
            reader = KyotoReader(Path(kc_joined_path),
                                 target_cases=dataset_config['target_cases'],
                                 target_corefs=dataset_config['target_corefs'],
                                 extract_nes=False)
            self.joined_documents = list(reader.process_all_documents())

        for document in tqdm(documents, desc='processing documents'):
            example = read_example(document,
                                   cases=self.target_cases,
                                   exophors=self.target_exophors,
                                   coreference=coreference,
                                   bridging=bridging,
                                   kc=kc,
                                   pas_targets=pas_targets,
                                   dataset_config=dataset_config)
            feature = self._convert_example_to_feature(example, dataset_config['max_seq_length'])
            if feature is None:
                continue
            self.examples.append(example)
            self.features.append(feature)

    def _convert_example_to_feature(self,
                                    example: PasExample,
                                    max_seq_length: int) -> Optional[InputFeatures]:
        """Loads a data file into a list of `InputBatch`s."""

        vocab_size = self.tokenizer.vocab_size
        num_special_tokens = len(self.special_to_index)
        num_relations = len(self.target_cases) + int(self.bridging) + int(self.coreference)

        all_tokens, tok_to_orig_index, orig_to_tok_index = self._get_tokenized_tokens(example.words)
        # ignore too long document
        if len(all_tokens) > max_seq_length - num_special_tokens:
            return None

        tokens: List[str] = []
        segment_ids: List[int] = []
        arguments_set: List[List[List[int]]] = []
        candidates_set: List[List[List[int]]] = []
        overts_set: List[List[List[int]]] = []
        deps: List[List[int]] = []

        # subword loop
        for token, orig_index in zip(all_tokens, tok_to_orig_index):
            tokens.append(token)
            segment_ids.append(0)

            # subsequent subword or [CLS] token or [SEP] token
            if token.startswith("##") or orig_index is None:
                arguments_set.append([[] for _ in range(num_relations)])
                overts_set.append([[] for _ in range(num_relations)])
                candidates_set.append([[] for _ in range(num_relations)])
                deps.append([0] * max_seq_length)
                continue

            arguments: List[List[int]] = [[] for _ in range(num_relations)]
            overts: List[List[int]] = [[] for _ in range(num_relations)]
            for i, (case, arg_strings) in enumerate(example.arguments_set[orig_index].items()):
                if not arg_strings:
                    continue
                for arg_string in arg_strings:
                    if case == '=':
                        # coreference (arg_string: 著者, 23, NA, ...)
                        if arg_string in self.special_to_index:
                            arguments[i].append(self.special_to_index[arg_string])
                        else:
                            arguments[i].append(orig_to_tok_index[int(arg_string)])
                    else:
                        # pas (arg_string: 著者, 8%C, 15%O, NULL, ...)
                        if arg_string in self.special_to_index:
                            if self.train_zero is False:
                                continue
                            arguments[i].append(self.special_to_index[arg_string])
                        else:
                            arg_index, flag = int(arg_string[:-2]), arg_string[-1]
                            if flag == 'C':
                                overts[i].append(orig_to_tok_index[arg_index])
                            if (flag == 'C' and self.train_overt is False) or \
                               (flag == 'N' and self.train_case is False) or \
                               (flag == 'O' and self.train_zero is False):
                                continue
                            arguments[i].append(orig_to_tok_index[arg_index])

            arguments_set.append(arguments)
            overts_set.append(overts)

            ddep = example.ddeps[orig_index]
            deps.append([(0 if idx is None or ddep != example.dtids[idx] else 1) for idx in tok_to_orig_index])
            deps[-1] += [0] * (max_seq_length - len(tok_to_orig_index))

            # arguments_set が空のもの (助詞など) には candidates を設定しない
            candidates: List[List[int]] = []
            for case, arg_strings in example.arguments_set[orig_index].items():
                if arg_strings:
                    if case != '=':
                        cands = [orig_to_tok_index[dmid] for dmid in example.arg_candidates_set[orig_index]]
                        specials = self.target_exophors + ['NULL']
                    else:
                        cands = [orig_to_tok_index[dmid] for dmid in example.ment_candidates_set[orig_index]]
                        specials = self.target_exophors + ['NA']
                    cands += [self.special_to_index[special] for special in specials]
                else:
                    cands = []
                candidates.append(cands)
            candidates_set.append(candidates)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = [True] * len(input_ids)

        # Zero-pad up to the sequence length (except for special tokens).
        while len(input_ids) < max_seq_length - num_special_tokens:
            input_ids.append(0)
            input_mask.append(False)
            segment_ids.append(0)
            arguments_set.append([[] for _ in range(num_relations)])
            overts_set.append([[] for _ in range(num_relations)])
            candidates_set.append([[] for _ in range(num_relations)])
            deps.append([0] * max_seq_length)

        # add special tokens
        for i in range(num_special_tokens):
            input_ids.append(vocab_size + i)
            input_mask.append(True)
            segment_ids.append(0)
            arguments_set.append([[] for _ in range(num_relations)])
            overts_set.append([[] for _ in range(num_relations)])
            candidates_set.append([[] for _ in range(num_relations)])
            deps.append([0] * max_seq_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(arguments_set) == max_seq_length
        assert len(overts_set) == max_seq_length
        assert len(candidates_set) == max_seq_length
        assert len(deps) == max_seq_length

        feature = InputFeatures(
            tokens=tokens,
            orig_to_tok_index=orig_to_tok_index,
            tok_to_orig_index=tok_to_orig_index,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            arguments_set=[[[int(x in args) for x in range(max_seq_length)] for args in arguments]
                           for arguments in arguments_set],
            overt_mask=[[[(x in overt) for x in range(max_seq_length)] for overt in overts]
                           for overts in overts_set],
            ng_token_mask=[[[(x in cands) for x in range(max_seq_length)] for cands in candidates]
                           for candidates in candidates_set],  # False -> mask, True -> keep
            deps=deps,
        )

        return feature

    def _get_tokenized_tokens(self, words: List[str]) -> Tuple[List[str], List[Optional[int]], List[int]]:
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

    def stat(self) -> dict:
        n_examples = n_preds = n_mentions = 0
        n_args = defaultdict(int)
        pas_overt = pas_exo = pas_null = pas_normal = 0
        coref_exo = coref_na = coref_normal = 0
        zero = defaultdict(int)

        for example in self.examples:
            for arguments in example.arguments_set:
                for case, argument in arguments.items():
                    if not argument:
                        continue
                    arg: str = argument[0]
                    if case == '=':
                        if arg in self.target_exophors:
                            coref_exo += 1
                        elif arg == 'NA':
                            coref_na += 1
                        else:
                            coref_normal += 1
                    else:
                        if '%C' in arg:
                            pas_overt += 1
                        elif arg in self.target_exophors:
                            pas_exo += 1
                        elif arg == 'NULL':
                            pas_null += 1
                            continue
                        else:
                            pas_normal += 1
                        if '%O' in arg or arg in self.target_exophors:
                            zero[case] += 1
                        n_args[case] += 1
                if self.coreference:
                    if arguments['='] is not None:
                        n_mentions += 1
                    if any(arg is not None for arg in list(arguments.values())[:-1]):
                        n_preds += 1
                else:
                    if any(arg is not None for arg in arguments.values()):
                        n_preds += 1
            n_examples += 1

        n_all_tokens = n_input_tokens = n_unk_tokens = 0
        unk_id = self.tokenizer.convert_tokens_to_ids('[UNK]')
        pad_id = 0
        for feature in self.features:
            for token_id in feature.input_ids:
                n_all_tokens += 1
                if token_id == pad_id:
                    continue
                n_input_tokens += 1
                if token_id == unk_id:
                    n_unk_tokens += 1

        return {'examples': n_examples,
                'predicates': n_preds,
                'mentions': n_mentions,
                'ga_cases': n_args['ガ'],
                'wo_cases': n_args['ヲ'],
                'ni_cases': n_args['ニ'],
                'ga2_cases': n_args['ガ２'],
                'pas_overt': pas_overt,
                'pas_exophor': pas_exo,
                'pas_normal': pas_normal,
                'pas_null': pas_null,
                'coref_exophor': coref_exo,
                'coref_normal': coref_normal,
                'coref_na': coref_na,
                'n_all_tokens': n_all_tokens,
                'n_input_tokens': n_input_tokens,
                'n_unk_tokens': n_unk_tokens,
                'zero': zero
                }

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx) -> tuple:
        feature = self.features[idx]
        input_ids = np.array(feature.input_ids)          # (seq)
        input_mask = np.array(feature.input_mask)        # (seq)
        segment_ids = np.array(feature.segment_ids)      # (seq)
        arguments_ids = np.array(feature.arguments_set)  # (seq, case, seq)
        overt_mask = np.array(feature.overt_mask)        # (seq, case, seq)
        ng_token_mask = np.array(feature.ng_token_mask)  # (seq, case, seq)
        deps = np.array(feature.deps)                    # (seq, seq)
        task = np.array(TASK_ID['pa'])                   # ()
        return input_ids, input_mask, segment_ids, ng_token_mask, arguments_ids, deps, task, overt_mask
