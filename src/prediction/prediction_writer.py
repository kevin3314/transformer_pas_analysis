import io
import re
from collections import defaultdict
from logging import Logger
from pathlib import Path
from typing import List, Optional, Dict, NamedTuple, Union, TextIO
import configparser
import shutil

from pyknp import KNP
from kyoto_reader import Document, Pas, BaseArgument, Argument, SpecialArgument, BasePhrase

from data_loader.dataset import PASDataset
from data_loader.dataset.read_example import PasExample


class PredictionKNPWriter:
    """ システム出力を KNP 形式で書き出すクラス

    Args:
        dataset (PASDataset): 解析対象のデータセット
        logger (Logger): ロガー
        use_knp_overt (bool): overt について本システムの解析結果ではなく、KNP の格解析結果を使用する (default: True)
    """
    REL_PAT = re.compile(r'<rel type="([^\s]+?)"(?: mode="([^>]+?)")? target="(.*?)"(?: sid="(.*?)" id="(.+?)")?/>')
    TAG_PAT = re.compile(r'^\+ -?\d+\w ?')

    def __init__(self,
                 dataset: PASDataset,
                 logger: Logger,
                 use_knp_overt: bool = True,
                 ) -> None:
        self.examples: List[PasExample] = dataset.examples
        self.cases: List[str] = dataset.target_cases
        self.bridging: bool = dataset.bridging
        self.coreference: bool = dataset.coreference
        self.relations: List[str] = dataset.target_cases + (['ノ'] * self.bridging) + (['='] * self.coreference)
        self.exophors: List[str] = dataset.target_exophors
        self.index_to_special: Dict[int, str] = {idx: token for token, idx in dataset.special_to_index.items()}
        self.documents: List[Document] = dataset.documents
        self.logger = logger
        self.use_knp_overt = use_knp_overt
        self.kc: bool = dataset.kc
        self.reader = dataset.reader
        cfg = configparser.ConfigParser()
        cfg.read(Path(__file__).parent.parent / 'analyzer' / 'config.ini')
        if 'default' not in cfg:
            logger.warning('Analyzer config not found. Instead, use default values.')
            cfg['default'] = {}
        section = cfg['default']
        knp_command = section.get('knp_command', shutil.which('knp'))
        jumanpp_command = section.get('juman_command', shutil.which('jumanpp'))
        self.knp = KNP(command=knp_command, option='-tab -case2', jumancommand=jumanpp_command)

    def write(self,
              arguments_sets: List[List[List[int]]],
              destination: Union[Path, TextIO, None],
              skip_untagged: bool = True,
              ) -> List[Document]:
        """Write final predictions to the file."""

        if isinstance(destination, Path):
            self.logger.info(f'Writing predictions to: {destination}')
            destination.mkdir(exist_ok=True)
        elif not (destination is None or isinstance(destination, io.TextIOBase)):
            self.logger.warning('invalid output destination')

        did2examples = {ex.doc_id: ex for ex in self.examples}
        did2arguments_sets = {ex.doc_id: arguments_set for ex, arguments_set in zip(self.examples, arguments_sets)}

        did2knps: Dict[str, List[str]] = defaultdict(list)
        for document in self.documents:
            did = document.doc_id
            input_knp_lines = document.knp_string.strip().split('\n')
            if did in did2examples:
                output_knp_lines = self._rewrite_rel(input_knp_lines,
                                                     did2examples[did],
                                                     did2arguments_sets[did],
                                                     document)
            else:
                if skip_untagged:
                    continue
                output_knp_lines = []
                for line in input_knp_lines:
                    if line.startswith('+ '):
                        line = self.REL_PAT.sub('', line)  # remove gold data
                        assert '<rel ' not in line
                    output_knp_lines.append(line)

            knp_strings: List[str] = []
            buff = ''
            for knp_line in output_knp_lines:
                buff += knp_line + '\n'
                if knp_line.strip() == 'EOS':
                    knp_strings.append(buff)
                    buff = ''
            if self.kc:
                orig_did, idx = did.split('-')
                if idx == '00':
                    did2knps[orig_did] += knp_strings
                else:
                    did2knps[orig_did].append(knp_strings[-1])
            else:
                did2knps[did] = knp_strings

        documents_pred: List[Document] = []  # kc については元通り結合された文書のリスト
        for did, knp_strings in did2knps.items():
            document_pred = Document(''.join(knp_strings),
                                     did,
                                     self.reader.target_cases,
                                     self.reader.target_corefs,
                                     self.reader.relax_cases,
                                     extract_nes=False,
                                     use_pas_tag=False)
            documents_pred.append(document_pred)
            if destination is None:
                continue
            output_knp_lines = self._add_pas_analysis(document_pred.knp_string.split('\n'), document_pred)
            output_string = '\n'.join(output_knp_lines) + '\n'
            if isinstance(destination, Path):
                output_basename = did + '.knp'
                with destination.joinpath(output_basename).open('w') as writer:
                    writer.write(output_string)
            elif isinstance(destination, io.TextIOBase):
                destination.write(output_string)

        return documents_pred

    def _rewrite_rel(self,
                     knp_lines: List[str],
                     example: PasExample,
                     arguments_set: List[List[int]],  # (max_seq_len, cases)
                     document: Document,
                     ) -> List[str]:
        overts = self._extract_overt(knp_lines, document)

        output_knp_lines = []
        dtid = 0
        sent_idx = 0
        for line in knp_lines:
            if not line.startswith('+ '):
                output_knp_lines.append(line)
                if line == 'EOS':
                    sent_idx += 1
                continue

            rel_removed: str = self.REL_PAT.sub('', line)  # remove gold data
            assert '<rel ' not in rel_removed
            match = self.TAG_PAT.match(rel_removed)
            if match is not None:
                rel_string = self._rel_string(document.bp_list()[dtid],
                                              example,
                                              arguments_set,
                                              document,
                                              overts[dtid])
                rel_idx = match.end()
                output_knp_lines.append(rel_removed[:rel_idx] + rel_string + rel_removed[rel_idx:])
            else:
                self.logger.warning(f'invalid format line: {line}')
                output_knp_lines.append(rel_removed)

            dtid += 1

        return output_knp_lines

    def _extract_overt(self,
                       knp_lines: List[str],
                       document: Document,
                       ) -> Dict[int, Dict[str, int]]:
        overts: Dict[int, Dict[str, int]] = defaultdict(dict)
        blists = []
        buff = ''
        for line in knp_lines:
            buff += line + '\n'
            if line == 'EOS':
                try:
                    blists.append(self.knp.reparse_knp_result(buff.strip()))
                except Exception:
                    self.logger.warning(f"overt extraction from '{document.doc_id}' was partly skipped due to an error "
                                        f"occurred in KNP")
                    blists.append(None)
                buff = ''
        assert len(document) == len(blists)
        for sentence, blist in zip(document, blists):
            if blist is None:
                continue
            tag_list = blist.tag_list()
            assert len(sentence.bps) == len(tag_list)
            for bp, tag in zip(sentence.bps, tag_list):
                assert bp.tid == tag.tag_id
                if tag.pas is None:
                    continue
                for case, args in tag.pas.arguments.items():
                    if case not in self.cases:  # ノ格は表層格から overt 判定できないので overts に追加しない
                        continue
                    for arg in filter(lambda a: a.flag == 'C', args):
                        overts[bp.dtid][case] = sentence.bps[arg.tid].dmid
        return overts

    def _rel_string(self,
                    bp: BasePhrase,
                    example: PasExample,
                    arguments_set: List[List[int]],  # (max_seq_len, cases)
                    document: Document,
                    overt_dict: Dict[str, int],
                    ) -> str:
        rels: List[RelTag] = []
        dmid2bp = {document.mrph2dmid[mrph]: bp for bp in document.bp_list() for mrph in bp.mrph_list()}
        assert len(example.arguments_set) == len(dmid2bp)
        for mrph in bp.mrph_list():
            dmid: int = document.mrph2dmid[mrph]
            token_index: int = example.orig_to_tok_index[dmid]
            arguments: List[int] = arguments_set[token_index]
            # 助詞などの非解析対象形態素については gold_args が空になっている
            # inference時、解析対象形態素は ['NULL'] となる
            is_targets: Dict[str, bool] = {rel: bool(args) for rel, args in example.arguments_set[dmid].items()}
            assert len(self.relations) == len(arguments)
            for relation, argument in zip(self.relations, arguments):
                if not is_targets[relation]:
                    continue
                if self.use_knp_overt and relation in overt_dict:
                    # overt
                    prediction_dmid = overt_dict[relation]
                elif argument in self.index_to_special:
                    # special
                    special_arg = self.index_to_special[argument]
                    if special_arg in self.exophors:  # exclude [NULL] and [NA]
                        rels.append(RelTag(relation, special_arg, None, None))
                    continue
                else:
                    # normal
                    prediction_dmid = example.tok_to_orig_index[argument]
                    if prediction_dmid is None:
                        # [SEP] or [CLS]
                        self.logger.warning("Choose [SEP] as an argument. Tentatively, change it to NULL.")
                        continue
                prediction_bp: BasePhrase = dmid2bp[prediction_dmid]
                rels.append(RelTag(relation, prediction_bp.core, prediction_bp.sid, prediction_bp.tid))

        return ''.join(rel.to_string() for rel in rels)

    def _add_pas_analysis(self,
                          knp_lines: List[str],
                          document: Document,
                          ) -> List[str]:
        dtid2pas = {pas.dtid: pas for pas in document.pas_list()}
        dtid = 0
        output_knp_lines = []
        for line in knp_lines:
            if not line.startswith('+ '):
                output_knp_lines.append(line)
                continue
            if dtid in dtid2pas:
                pas_string = self._pas_string(dtid2pas[dtid], 'dummy:dummy', document)
                output_knp_lines.append(line + pas_string)
            else:
                output_knp_lines.append(line)

            dtid += 1

        return output_knp_lines

    def _pas_string(self,
                    pas: Pas,
                    cfid: str,
                    document: Document,
                    ) -> str:
        sid2index: Dict[str, int] = {sid: i for i, sid in enumerate(document.sid2sentence.keys())}
        dtype2caseflag = {'overt': 'C', 'dep': 'N', 'intra': 'O', 'inter': 'O', 'exo': 'E'}
        case_elements = []
        for case in self.cases + (['ノ'] * self.bridging):
            items = ['-'] * 6
            items[0] = case
            args = pas.arguments[case]
            if args:
                arg: BaseArgument = args[0]
                items[1] = dtype2caseflag[arg.dep_type]  # フラグ (C/N/O/D/E/U)
                items[2] = str(arg)  # 見出し
                if isinstance(arg, Argument):
                    items[3] = str(sid2index[pas.sid] - sid2index[arg.sid])  # N文前
                    items[4] = str(arg.tid)  # tag id
                    items[5] = str(document.get_entities(arg)[0].eid)  # Entity ID
                else:
                    assert isinstance(arg, SpecialArgument)
                    items[3] = str(-1)
                    items[4] = str(-1)
                    items[5] = str(arg.eid)  # Entity ID
            else:
                items[1] = 'U'
            case_elements.append('/'.join(items))
        return f"<述語項構造:{cfid}:{';'.join(case_elements)}>"


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
