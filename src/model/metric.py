import re
import sys
from logging import Logger
from typing import List, Optional, Dict, NamedTuple
from pathlib import Path

# import torch
from pyknp import Tag

from data_loader.dataset import InputFeatures, PASDataset
from kwdlc_reader import KWDLCReader, Document

#
# def _parse_result(result_str: str, metric_name: str, cases: List[str]):
#     result_lines: List[str] = result_str.split('\n')
#     start_idx: int = result_lines.index(metric_name) + 1
#     f1_dic: Dict[str: float] = {}
#     for idx, line in list(enumerate(result_lines))[start_idx:]:
#         if line in cases:
#             f1_line = result_lines[idx + 3]
#             assert f1_line.startswith('F:')
#             f1_dic[line] = int(f1_line[2:])
#             cases.remove(line)
#         if not cases:
#             break
#     return f1_dic


def case_analysis_f1_ga(result: dict):
    return result['ガ']['case_analysis']['F']


def case_analysis_f1_wo(result: dict):
    return result['ヲ']['case_analysis']['F']


def case_analysis_f1_ni(result: dict):
    return result['ニ']['case_analysis']['F']


def case_analysis_f1_ga2(result: dict):
    return result['ガ２']['case_analysis']['F']


def case_analysis_f1(result: dict):
    return result['all_case']['case_analysis']['F']


def zero_anaphora_f1_ga(result: dict):
    return result['ガ']['anaphora_all']['F']


def zero_anaphora_f1_wo(result: dict):
    return result['ヲ']['anaphora_all']['F']


def zero_anaphora_f1_ni(result: dict):
    return result['ニ']['anaphora_all']['F']


def zero_anaphora_f1_ga2(result: dict):
    return result['ガ２']['anaphora_all']['F']


def zero_anaphora_f1(result: dict):
    return result['all_case']['anaphora_all']['F']


def zero_anaphora_f1_inter(result: dict):
    return result['all_case']['anaphora_inter_sentential']['F']


def zero_anaphora_f1_intra(result: dict):
    return result['all_case']['anaphora_intra_sentential']['F']


def zero_anaphora_f1_writer_reader(result: dict):
    return result['all_case']['anaphora_writer_reader']['F']


# def my_metric(output, target):
#     with torch.no_grad():
#         pred = torch.argmax(output, dim=1)
#         assert pred.shape[0] == len(target)
#         correct = 0
#         correct += torch.sum(pred == target).item()
#     return correct / len(target)
#
#
# def my_metric2(output, target, k=3):
#     with torch.no_grad():
#         pred = torch.topk(output, k, dim=1)[1]
#         assert pred.shape[0] == len(target)
#         correct = 0
#         for i in range(k):
#             correct += torch.sum(pred[:, i] == target).item()
#     return correct / len(target)


class PredictionKNPWriter:
    rel_pat = re.compile(r'<rel type="([^\s]+?)"(?: mode="([^>]+?)")? target="(.*?)"(?: sid="(.*?)" id="(.+?)")?/>')
    tag_pat = re.compile(r'^\+ (-?\d)+\w ?')

    def __init__(self,
                 dataset: PASDataset,
                 dataset_config: dict,
                 logger: Logger,
                 ) -> None:
        self.gold_arguments_sets: List[List[Dict[str, Optional[str]]]] = \
            [example.arguments_set for example in dataset.pas_examples]
        self.all_features: List[InputFeatures] = dataset.features
        self.reader: KWDLCReader = dataset.reader
        self.special_tokens: List[str] = dataset.special_tokens
        max_seq_length: int = dataset_config['max_seq_length']
        self.tok_to_special: Dict[int, str] = {i + max_seq_length - len(self.special_tokens): token for i, token
                                               in enumerate(self.special_tokens)}
        self.coreference: bool = dataset_config['coreference']
        input_dir: str = dataset_config['path']
        self.input_files: List[Path] = sorted(Path(input_dir).glob('*.knp'))
        self.logger = logger

    def write(self,
              arguments_sets: List[List[List[int]]],
              output_dir: Optional[Path]
              ) -> None:
        """Write final predictions to the file."""

        if output_dir is not None:
            self.logger.info(f'Writing predictions to: {output_dir}')
            output_dir.mkdir(exist_ok=True)

        for input_file, features, arguments_set, gold_arguments_set in \
                zip(self.input_files, self.all_features, arguments_sets, self.gold_arguments_sets):
            document = self.reader.process_document(input_file.stem)
            if document is None:
                self.logger.warning(f'document: {document.doc_id} is skipped.')
                continue
            output_basename = document.doc_id + '.knp'
            with output_dir.joinpath(output_basename).open('w') if output_dir is not None else sys.stdout as writer:
                output_knp_lines = self._output_document(input_file,
                                                         features,
                                                         arguments_set,
                                                         gold_arguments_set,
                                                         document)
                writer.write('\n'.join(output_knp_lines))

    def _output_document(self,
                         input_file: Path,
                         features: InputFeatures,
                         arguments_set: List[List[int]],
                         gold_arguments_set: List[Dict[str, Optional[str]]],
                         document: Document,
                         ) -> List[str]:
        with input_file.open() as fin:
            dtid = 0
            output_knp_lines = []
            for line in fin:
                if not line.startswith('+ '):
                    output_knp_lines.append(line.strip())
                    continue
                rel_removed: str = self.rel_pat.sub('', line.strip())  # remove gold data
                assert '<rel ' not in rel_removed
                match = self.tag_pat.match(rel_removed)
                if match is not None:
                    rel_idx = match.end()
                    rel_string = self._rel_string(dtid, arguments_set, gold_arguments_set, features, document)
                    rel_inserted_line = rel_removed[:rel_idx] + rel_string + rel_removed[rel_idx:]
                    output_knp_lines.append(rel_inserted_line)
                else:
                    self.logger.warning(f'invalid format line: {line.strip()}')
                    output_knp_lines.append(rel_removed)

                dtid += 1

        return output_knp_lines

    def _rel_string(self,
                    dtid: int,
                    arguments_set: List[List[int]],  # (max_seq_len, cases)
                    gold_arguments_set: List[Dict[str, Optional[str]]],  # (mrph_len, cases)
                    features: InputFeatures,
                    document: Document,
                    ) -> str:
        rels: List[RelTag] = []
        dmid2tag = {document.mrph2dmid[mrph]: tag for tag in document.tag_list() for mrph in tag.mrph_list()}
        tag2sid = {tag: sentence.sid for sentence in document for tag in sentence.tag_list()}
        tag = document.dtid2tag[dtid]
        assert len(gold_arguments_set) == len(dmid2tag)
        cases: List[str] = document.target_cases + (['='] if self.coreference else [])
        for mrph in tag.mrph_list():
            dmid = document.mrph2dmid[mrph]
            token_index = features.orig_to_tok_index[dmid]
            arguments: List[int] = arguments_set[token_index]
            # {'ガ': '14', 'ヲ': '23%C', 'ニ': 'NULL', 'ガ２': 'NULL', '=': None}
            gold_arguments: Dict[str, Optional[str]] = gold_arguments_set[dmid]
            assert len(cases) == len(arguments)
            assert len(cases) == len(gold_arguments)
            for (case, gold_argument), argument in zip(gold_arguments.items(), arguments):
                # Noneは解析対象としない
                if gold_argument is not None:
                    # overt
                    if gold_argument.endswith('%C'):
                        prediction_dmid = int(gold_argument[:-2])  # overt の場合のみ正解データををそのまま出力
                    else:
                        # special
                        if argument in self.tok_to_special:
                            special_anaphor = self.tok_to_special[argument]
                            if special_anaphor in document.target_exophors:
                                rels.append(RelTag(case, special_anaphor, None, None))
                            continue
                        # [SEP] or [CLS]
                        elif features.tok_to_orig_index[argument] is None:
                            self.logger.warning("Choose [SEP] as an argument. Tentatively, change it to NULL.")
                            continue
                        # normal
                        else:
                            prediction_dmid = features.tok_to_orig_index[argument]
                    prediction_tag: Tag = dmid2tag[prediction_dmid]
                    rels.append(RelTag(case, prediction_tag.midasi, tag2sid[prediction_tag], prediction_tag.tag_id))

        return ''.join([rel.to_string() for rel in rels])


class RelTag(NamedTuple):
    type_: str
    target: str
    sid: Optional[str]
    tid: Optional[int]

    def to_string(self):
        string = f'<rel type="{self.type_}" target="{self.target}"'
        if self.sid is not None:
            string += f' sid="{self.sid}" id="{self.tid}"'
        string += '/>'
        return string
