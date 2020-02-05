import os
import logging
from typing import List, Dict, Optional
from pathlib import Path
import _pickle as cPickle
from collections import defaultdict
import hashlib

from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizer

from kwdlc_reader import KWDLCReader
from data_loader.dataset.read_example import read_example, PasExample


class InputFeatures:
    """A single set of features of data."""

    def __init__(self,
                 tokens: List[str],
                 orig_to_tok_index: List[int],  # use for output
                 tok_to_orig_index: List[Optional[int]],  # use for output
                 input_ids: List[int],  # use for model
                 input_mask: List[int],  # use for model
                 segment_ids: List[int],  # use for model
                 arguments_set: List[List[List[int]]],  # use for model
                 ng_arg_mask: List[List[int]],  # use for model
                 ng_ment_mask: List[List[int]],  # use for model
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
        self.ng_ment_mask = ng_ment_mask
        self.deps = deps


class PASDataset(Dataset):
    def __init__(self,
                 path: Optional[str],
                 max_seq_length: int,
                 cases: List[str],
                 corefs: List[str],
                 exophors: List[str],
                 coreference: bool,
                 training: bool,
                 bert_model: str,
                 kc: bool,
                 train_target: List[str],
                 eventive_noun: bool = False,
                 knp_string: Optional[str] = None,
                 logger=None,
                 ) -> None:
        if path is not None:
            source = Path(path)
        else:
            assert knp_string is not None
            source = knp_string
        self.reader = KWDLCReader(source,
                                  target_cases=cases,
                                  target_corefs=corefs,
                                  extract_nes=False)
        self.target_cases = self.reader.target_cases
        self.target_exophors = exophors
        self.coreference = coreference
        self.kc = kc
        self.train_overt = 'overt' in train_target
        self.train_case = 'case' in train_target
        self.train_zero = 'zero' in train_target
        self.logger = logger if logger else logging.getLogger(__file__)
        special_tokens = exophors + ['NULL'] + (['NA'] if coreference else [])
        self.special_to_index: Dict[str, int] = {token: max_seq_length - i - 1 for i, token
                                                 in enumerate(reversed(special_tokens))}
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=False, tokenize_chinese_chars=False)
        self.expanded_vocab_size = self.tokenizer.vocab_size + len(special_tokens)
        documents = list(self.reader.process_all_documents())
        self.documents = documents if not training else None
        self.examples = []
        self.features = []

        for document in tqdm(documents, desc='processing documents'):
            bpa_cache_dir: Path = Path(os.environ.get('BPA_CACHE_DIR', f'/data/{os.environ["USER"]}/bpa_cache'))
            bpa_cache_dir.mkdir(exist_ok=True, parents=True)
            example_hash = self._hash(document, bert_model, eventive_noun)
            cache_path = bpa_cache_dir / example_hash / (document.doc_id + self.reader.pickle_ext)
            if cache_path.exists():
                with cache_path.open('rb') as f:
                    example = cPickle.load(f)
            else:
                example = read_example(document,
                                       target_exophors=exophors,
                                       coreference=coreference,
                                       kc=kc,
                                       eventive_noun=eventive_noun)
                cache_path.parent.mkdir(exist_ok=True)
                with cache_path.open('wb') as f:
                    cPickle.dump(example, f)
            feature = self._convert_example_to_feature(example, max_seq_length)
            if feature is None:
                continue
            self.examples.append(example)
            self.features.append(feature)

    def _hash(self, document, *args) -> str:
        attrs_dataset = ('target_cases', 'target_exophors', 'coreference', 'train_overt', 'train_case', 'train_zero')
        attrs_document = ('target_cases', 'target_corefs', 'relax_cases', 'relax_corefs', 'extract_nes', 'use_pas_tag')
        vars_dataset = {k: v for k, v in vars(self).items() if k in attrs_dataset}
        vars_document = {k: v for k, v in vars(document).items() if k in attrs_document}
        string = repr(sorted(vars_dataset)) + repr(sorted(vars_document)) + ''.join(repr(a) for a in args)
        return hashlib.md5(string.encode()).hexdigest()

    def _convert_example_to_feature(self,
                                    example: PasExample,
                                    max_seq_length: int) -> Optional[InputFeatures]:
        """Loads a data file into a list of `InputBatch`s."""

        vocab_size = self.tokenizer.vocab_size
        num_expand_vocab = len(self.special_to_index)
        num_case_w_coreference = len(self.target_cases) + int(self.coreference)

        all_tokens, tok_to_orig_index, orig_to_tok_index = self._get_tokenized_tokens(example.words)
        # ignore too long document
        if len(all_tokens) > max_seq_length - num_expand_vocab:
            return None

        tokens: List[str] = []
        segment_ids: List[int] = []
        arguments_set: List[List[List[int]]] = []
        arg_candidates_set: List[List[int]] = []
        ment_candidates_set: List[List[int]] = []
        deps: List[List[int]] = []

        # subword loop
        for token, orig_index in zip(all_tokens, tok_to_orig_index):
            tokens.append(token)
            segment_ids.append(0)

            # subsequent subword or [CLS] token or [SEP] token
            if token.startswith("##") or orig_index is None:
                arguments_set.append([[] for _ in range(num_case_w_coreference)])
                arg_candidates_set.append([])
                ment_candidates_set.append([])
                deps.append([0] * max_seq_length)
                continue

            arguments: List[List[int]] = [[] for _ in range(num_case_w_coreference)]
            for i, (case, arg_strings) in enumerate(example.arguments_set[orig_index].items()):
                if not arg_strings:
                    continue
                for arg_string in arg_strings:
                    if case == '=':
                        # coreference (arg_string: 著者, 23, NA, ...)
                        if arg_string in self.special_to_index:
                            # special token
                            arguments[i].append(self.special_to_index[arg_string])
                        else:
                            # normal
                            if int(arg_string) not in example.ment_candidates_set[orig_index]:
                                self.logger.debug(f'mention: {arg_string} of {token} is not in candidates and ignored')
                                continue
                            arguments[i].append(orig_to_tok_index[int(arg_string)])
                    else:
                        # arg_string: 著者, 8%C, 15%O, NULL, ...
                        if arg_string in self.special_to_index:
                            # special token
                            if self.train_zero is False:
                                continue
                            arguments[i].append(self.special_to_index[arg_string])
                        else:
                            # normal
                            if (arg_string.endswith('%C') and self.train_overt is False) or \
                                    (arg_string.endswith('%N') and self.train_case is False) or \
                                    (arg_string.endswith('%O') and self.train_zero is False):
                                continue
                            arg_string = arg_string[:-2]  # strip "%X"
                            if int(arg_string) not in example.arg_candidates_set[orig_index]:
                                self.logger.debug(f'argument: {arg_string} of {token} is not in candidates and ignored')
                                continue
                            arguments[i].append(orig_to_tok_index[int(arg_string)])

            arguments_set.append(arguments)

            ddep = example.ddeps[orig_index]
            deps.append([(0 if idx is None or ddep != example.dtids[idx] else 1) for idx in tok_to_orig_index])
            deps[-1] += [0] * (max_seq_length - len(tok_to_orig_index))

            arg_candidates = [orig_to_tok_index[dmid] for dmid in example.arg_candidates_set[orig_index]] + \
                             [self.special_to_index[special] for special in (self.target_exophors + ['NULL'])]
            arg_candidates_set.append(arg_candidates)
            if self.coreference:
                ment_candidates = [orig_to_tok_index[dmid] for dmid in example.ment_candidates_set[orig_index]] + \
                                  [self.special_to_index[special] for special in (self.target_exophors + ['NA'])]
            else:
                ment_candidates = []
            ment_candidates_set.append(ment_candidates)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = [True] * len(input_ids)

        # Zero-pad up to the sequence length (except for special tokens).
        while len(input_ids) < max_seq_length - num_expand_vocab:
            input_ids.append(0)
            input_mask.append(False)
            segment_ids.append(0)
            arguments_set.append([[] for _ in range(num_case_w_coreference)])
            arg_candidates_set.append([])
            ment_candidates_set.append([])
            deps.append([0] * max_seq_length)

        # add special tokens
        for i in range(num_expand_vocab):
            input_ids.append(vocab_size + i)
            input_mask.append(True)
            segment_ids.append(0)
            arguments_set.append([[] for _ in range(num_case_w_coreference)])
            arg_candidates_set.append([])
            ment_candidates_set.append([])
            deps.append([0] * max_seq_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(arguments_set) == max_seq_length
        assert len(arg_candidates_set) == max_seq_length
        assert len(ment_candidates_set) == max_seq_length
        assert len(deps) == max_seq_length

        feature = InputFeatures(
            tokens=tokens,
            orig_to_tok_index=orig_to_tok_index,
            tok_to_orig_index=tok_to_orig_index,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            arguments_set=[[[(x in args) for x in range(max_seq_length)] for args in arguments]
                           for arguments in arguments_set],
            ng_arg_mask=[[(x in candidates) for x in range(max_seq_length)]
                         for candidates in arg_candidates_set],  # False -> mask, True -> keep
            ng_ment_mask=[[(x in candidates) for x in range(max_seq_length)]
                          for candidates in ment_candidates_set],  # False -> mask, True -> keep
            deps=deps,
        )

        return feature

    def _get_tokenized_tokens(self, words: List[str]) -> tuple:
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
        arguments_ids = np.array(feature.arguments_set)  # (seq, case, seq)
        ng_arg_mask = np.array(feature.ng_arg_mask)      # (seq, seq)
        ng_ment_mask = np.array(feature.ng_ment_mask)    # (seq, seq)
        stacks = [ng_arg_mask] * len(self.target_cases) + ([ng_ment_mask] if self.coreference else [])
        ng_token_mask = np.stack(stacks, axis=1)         # (seq, case, seq)
        deps = np.array(feature.deps)                    # (seq, seq)
        return input_ids, input_mask, arguments_ids, ng_token_mask, deps
