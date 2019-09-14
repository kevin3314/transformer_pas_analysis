import io
import re
from logging import Logger
from typing import List, Optional, Dict, NamedTuple, Union
from pathlib import Path

from pyknp import Tag

from data_loader.dataset import InputFeatures, PASDataset
from kwdlc_reader import KWDLCReader, KWDLCStringReader, Document


def case_analysis_f1_ga(result: dict):
    return result['ガ']['case_analysis'].f1


def case_analysis_f1_wo(result: dict):
    return result['ヲ']['case_analysis'].f1


def case_analysis_f1_ni(result: dict):
    return result['ニ']['case_analysis'].f1


def case_analysis_f1_ga2(result: dict):
    return result['ガ２']['case_analysis'].f1


def case_analysis_f1(result: dict):
    return result['all_case']['case_analysis'].f1


def zero_anaphora_f1_ga(result: dict):
    return result['ガ']['zero_all'].f1


def zero_anaphora_f1_wo(result: dict):
    return result['ヲ']['zero_all'].f1


def zero_anaphora_f1_ni(result: dict):
    return result['ニ']['zero_all'].f1


def zero_anaphora_f1_ga2(result: dict):
    return result['ガ２']['zero_all'].f1


def zero_anaphora_f1(result: dict):
    return result['all_case']['zero_all'].f1


def zero_anaphora_f1_inter(result: dict):
    return result['all_case']['zero_inter_sentential'].f1


def zero_anaphora_f1_intra(result: dict):
    return result['all_case']['zero_intra_sentential'].f1


def zero_anaphora_f1_exophora(result: dict):
    return result['all_case']['zero_exophora'].f1


class PredictionKNPWriter:
    rel_pat = re.compile(r'<rel type="([^\s]+?)"(?: mode="([^>]+?)")? target="(.*?)"(?: sid="(.*?)" id="(.+?)")?/>')
    tag_pat = re.compile(r'^\+ (-?\d)+\w ?')

    def __init__(self,
                 dataset: PASDataset,
                 logger: Logger,
                 ) -> None:
        self.gold_arguments_sets: List[List[Dict[str, Optional[str]]]] = \
            [example.arguments_set for example in dataset.pas_examples]
        self.all_features: List[InputFeatures] = dataset.features
        self.reader: KWDLCReader = dataset.reader
        self.index_to_special: Dict[int, str] = {idx: token for token, idx in dataset.special_to_index.items()}
        self.coreference: bool = dataset.coreference
        self.input_files: List[Path] = list(dataset.reader.did2path.values())
        self.logger = logger

    def write(self,
              arguments_sets: List[List[List[int]]],
              destination: Union[Path, io.TextIOBase, None],
              ) -> List[Document]:
        """Write final predictions to the file."""

        if isinstance(destination, Path):
            self.logger.info(f'Writing predictions to: {destination}')
            destination.mkdir(exist_ok=True)
        elif not (destination is None or isinstance(destination, io.TextIOBase)):
            self.logger.warning('invalid output destination')

        documents_pred: List[Document] = []
        for input_file, features, arguments_set, gold_arguments_set in \
                zip(self.input_files, self.all_features, arguments_sets, self.gold_arguments_sets):
            if input_file is not None:
                document = self.reader.process_document(input_file.stem)
                with input_file.open() as fin:
                    knp_string = ''.join(fin.readlines())
            else:
                assert isinstance(self.reader, KWDLCStringReader)
                document = self.reader.process_document()
                knp_string = self.reader.knp_string
            if document is None:
                self.logger.warning(f'document: {document.doc_id} is skipped.')
                continue
            output_knp_lines = self._output_document(knp_string,
                                                     features,
                                                     arguments_set,
                                                     gold_arguments_set,
                                                     document)
            output_string = '\n'.join(output_knp_lines) + '\n'
            if isinstance(destination, Path):
                output_basename = document.doc_id + '.knp'
                with destination.joinpath(output_basename).open('w') as writer:
                    writer.write(output_string)
            elif isinstance(destination, io.TextIOBase):
                destination.write(output_string)
            else:
                pass
            documents_pred.append(
                Document(output_string,
                         document.doc_id,
                         document.target_cases,
                         document.target_corefs,
                         document.target_exophors,
                         document.extract_nes)
            )

        return documents_pred

    def _output_document(self,
                         knp_string: str,
                         features: InputFeatures,
                         arguments_set: List[List[int]],
                         gold_arguments_set: List[Dict[str, Optional[str]]],
                         document: Document,
                         ) -> List[str]:
        dtid = 0
        output_knp_lines = []
        for line in knp_string.strip().split('\n'):
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
                        if argument in self.index_to_special:
                            special_anaphor = self.index_to_special[argument]
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
                    target = ''.join(mrph.midasi for mrph in prediction_tag.mrph_list() if '<内容語>' in mrph.fstring)
                    if not target:
                        target = prediction_tag.midasi
                    rels.append(RelTag(case, target, tag2sid[prediction_tag], prediction_tag.tag_id))

        return ''.join(rel.to_string() for rel in rels)


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
