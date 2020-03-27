import re
import os
import socket
from typing import Dict, Tuple, Union, Optional, List
import configparser
from pathlib import Path
from datetime import datetime

import torch
from mojimoji import han_to_zen
from pyknp import Juman, KNP
from transformers import BertConfig
from textformatting import ssplit
from tqdm import tqdm

import model.model as module_arch
import data_loader.dataset as module_dataset
import data_loader.data_loaders as module_loader
from data_loader.dataset import PASDataset
from utils import read_json, prepare_device
from utils.parse_config import ConfigParser


class Analyzer:
    """Perform PAS analysis given a sentence."""

    def __init__(self, model_path: str, device: str, logger, bertknp: bool = False):
        cfg = configparser.ConfigParser()
        here = Path(__file__).parent
        cfg.read(here / 'config.ini')
        self.juman = cfg.get('default', 'juman_command')
        self.knp = cfg.get('default', 'knp_command')
        self.knp_host = cfg.get('default', 'knp_host')
        self.knp_port = cfg.getint('default', 'knp_port')
        self.juman_option = cfg.get('default', 'juman_option')
        self.knp_dpnd_option = cfg.get('default', 'knp_dpnd_option')
        self.knp_case_option = cfg.get('default', 'knp_case_option')
        self.pos_map, self.pos_map_inv = self._read_pos_list(here / 'pos.list')
        self.logger = logger
        self.bertknp = bertknp

        config_path = Path(model_path).parent / 'config.json'
        config = read_json(config_path)
        self.config = ConfigParser(config, resume=model_path, run_id='')  # save_dir 作らせたくない
        os.environ['BPA_DISABLE_CACHE'] = '1'

        dataset_config = config['test_kwdlc_dataset']['args']
        bert_config = BertConfig.from_pretrained(dataset_config['bert_model'])
        coreference = dataset_config['coreference']
        exophors = dataset_config['exophors']
        expanded_vocab_size = bert_config.vocab_size + len(exophors) + 1 + int(coreference)

        os.environ['CUDA_VISIBLE_DEVICES'] = device
        self.device, device_ids = prepare_device(1, self.logger)

        # build model architecture
        model = self.config.init_obj('arch', module_arch, vocab_size=expanded_vocab_size)
        self.logger.info(model)

        # prepare model
        self.logger.info(f'Loading checkpoint: {model_path} ...')
        state_dict = torch.load(model_path, map_location=self.device)['state_dict']
        model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
        self.model.eval()

    def analyze(self, source: Union[Path, str], knp_dir: Optional[str] = None) -> Tuple[list, PASDataset]:
        if isinstance(source, Path):
            self.logger.info(f'read knp files from {source}')
            save_dir = source
        else:
            save_dir = Path(knp_dir) if knp_dir is not None else Path('log') / datetime.now().strftime(r'%m%d_%H%M%S')
            save_dir.mkdir(exist_ok=True)
            sents = [self.sanitize_string(sent) for sent in ssplit(source)]
            self.logger.info('input: ' + ''.join(sents))
            knp_out = ''
            for i, sent in enumerate(sents):
                knp_out_ = self._apply_knp(sent)
                knp_out_ = knp_out_.replace('S-ID:1', f'S-ID:{i + 1}')
                knp_out += knp_out_
            with save_dir.joinpath(f'doc.knp').open(mode='wt') as f:
                f.write(knp_out)

        return self._analysis(save_dir)

    def analyze_from_knp(self, knp_out: str, knp_dir: Optional[str] = None) -> Tuple[list, PASDataset]:
        save_dir = Path(knp_dir) if knp_dir is not None else Path('log') / datetime.now().strftime(r'%m%d_%H%M%S')
        save_dir.mkdir(exist_ok=True)
        with save_dir.joinpath('doc.knp').open(mode='wt') as f:
            f.write(knp_out)
        return self._analysis(save_dir)

    def _analysis(self, path: Path) -> Tuple[list, PASDataset]:
        dataset_config: dict = self.config['test_kwdlc_dataset']['args']
        dataset_config['path'] = str(path)
        dataset = self.config.init_obj(f'test_kwdlc_dataset', module_dataset)
        data_loader = self.config.init_obj(f'test_data_loader', module_loader, dataset)

        arguments_sets: List[List[List[int]]] = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc='PAS analysis')):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, target, ng_token_mask, deps, task = batch

                output = self.model(input_ids, input_mask, segment_ids, ng_token_mask, deps)  # (b, seq, case, seq)
                if self.config['arch']['type'] == 'MultitaskDepModel':
                    scores = output[0]  # (b, seq, case, seq)
                elif re.match(r'(CaseInteractionModel2|Refinement|Duplicate)', self.config['arch']['type']):
                    scores = output[-1]  # (b, seq, case, seq)
                elif self.config['arch']['type'] == 'CommonsenseModel':
                    scores = output[0]  # (b, seq, case, seq)
                else:
                    scores = output  # (b, seq, case, seq)
                arguments_set = torch.argmax(scores, dim=3)  # (b, seq, case)
                arguments_sets += arguments_set.tolist()

        return arguments_sets, dataset

    def _apply_jumanpp(self, inp: str) -> Tuple[str, str]:
        jumanpp = Juman(command=self.juman, option=self.juman_option)
        jumanpp_result = jumanpp.analysis(inp)
        jumanpp_out = jumanpp_result.spec() + 'EOS\n'
        jumanpp_conll_out = self._jumanpp2conll_one_sentence(jumanpp_out) + 'EOS\n'
        return jumanpp_out, jumanpp_conll_out

    def _apply_knp(self, sent: str) -> str:
        self.logger.info(f'parse sentence: {sent}')
        knp = KNP(command=self.knp, jumancommand=self.juman, option=self.knp_dpnd_option)
        knp_result = knp.parse(sent)

        if self.bertknp is True:
            _, jumanpp_conll_out = self._apply_jumanpp(sent)
            clientsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.logger.info(f'connect to {self.knp_host}:{self.knp_port}')
            clientsock.connect((self.knp_host, self.knp_port))
            clientsock.sendall(jumanpp_conll_out.encode('utf-8'))

            buf = []
            while True:
                data = clientsock.recv(8192)
                data_utf8 = data.decode('utf-8')
                buf.append(data_utf8)
                if data_utf8.endswith('EOS\n'):
                    break
            clientsock.close()
            conllu_out = ''.join(buf)
            self.logger.info(f'received {len(conllu_out)} chars from BERTKNP')

            # modify KNP result by conllu result of BERTKNP
            head_ids, dpnd_types = self._read_conllu_from_buf(conllu_out)
            self._modify_knp(knp_result, head_ids, dpnd_types)

        # add predicate-argument structures by KNP
        knp = KNP(command=self.knp, jumancommand=self.juman, option=self.knp_case_option)
        knp_result_new = knp.parse_juman_result(knp_result.spec())
        return knp_result_new.spec()

    def _jumanpp2conll_one_sentence(self, jumanpp_out: str):

        output_lines = []
        prev_id = 0
        for line in jumanpp_out.splitlines():
            result = []
            if line.startswith('EOS'):
                break
            items = line.strip().split('\t')
            if prev_id == items[1]:
                continue  # skip the same id
            else:
                result.append(str(items[1]))
                prev_id = items[1]
            result.append(items[5])  # midasi
            result.append(items[8])  # genkei
            conll_pos = self.get_pos(items[9], items[11])  # hinsi, bunrui
            result.append(conll_pos)
            result.append(conll_pos)
            result.append('_')
            if len(items) > 19:
                result.append(items[18])  # head
                result.append(items[19])  # dpnd_type
            else:
                result.append('0')  # head
                result.append('D')  # dpnd_type (dummy)
            result.append('_')
            result.append('_')
            output_lines.append('\t'.join(result) + '\n')
        return ''.join(output_lines)

    def get_pos(self, pos: str, subpos: str) -> str:
        if subpos == '*':
            key = pos
        elif pos == '未定義語':
            key = '未定義語-その他'
        else:
            key = f'{pos}-{subpos}'

        if key in self.pos_map:
            return self.pos_map[key]
        else:
            assert f'Found unknown POS: {pos}-{subpos}'

    @staticmethod
    def _read_pos_list(path: str) -> Tuple[Dict[str, str], Dict[str, str]]:
        pos_map, pos_map_inv = {}, {}
        with open(path, encoding='utf-8') as f:
            for line in f:
                pos, pos_code = line.strip().split('\t')
                pos_map[pos] = pos_code
                pos_map_inv[pos_code] = pos
        return pos_map, pos_map_inv

    @staticmethod
    def _read_conllu_from_buf(conllu_out: str) -> Tuple[Dict[int, int], Dict[int, str]]:
        head_ids, dpnd_types = {}, {}
        for line in conllu_out.splitlines():
            if line == '\n' or line.startswith('EOS'):
                break
            items = line.strip().split('\t')
            _id = int(items[0]) - 1
            dpnd_id = int(items[6]) - 1
            head_ids[_id] = dpnd_id
            dpnd_types[_id] = items[7]
        return head_ids, dpnd_types

    @staticmethod
    def _modify_knp(knp_result, head_ids, dpnd_types):

        def modify(tags_, head_ids_, dpnd_types_, mode_: str) -> None:
            mrph_id2tag = {}
            for tag in tags_:
                for mrph in tag.mrph_list():
                    mrph_id2tag[mrph.mrph_id] = tag

            for tag in tags_:
                # この基本句内の形態素IDリスト
                in_tag_mrph_ids = {}
                last_mrph_id_in_tag = -1
                for mrph in tag.mrph_list():
                    in_tag_mrph_ids[mrph.mrph_id] = 1
                    if last_mrph_id_in_tag < mrph.mrph_id:
                        last_mrph_id_in_tag = mrph.mrph_id

                for mrph_id in list(in_tag_mrph_ids.keys()):
                    # 形態素係り先ID
                    mrph_head_id = head_ids_[mrph_id]
                    # 形態素係り先がROOTの場合は何もしない
                    if mrph_head_id == -1:
                        break
                    # 形態素係り先が基本句外に係る場合: 既存の係り先と異なるかチェック
                    if mrph_head_id > last_mrph_id_in_tag:
                        new_parent_tag = mrph_id2tag[mrph_head_id]
                        if mode_ == 'tag':
                            new_parent_id = new_parent_tag.tag_id
                            old_parent_id = tag.parent.tag_id
                        else:
                            new_parent_id = new_parent_tag.bnst_id
                            old_parent_id = tag.parent.bnst_id
                        # 係りタイプの更新
                        if dpnd_types_[mrph_id] != tag.dpndtype:
                            tag.dpndtype = dpnd_types_[mrph_id]
                        # 係り先の更新
                        if new_parent_id != old_parent_id:
                            # 形態素係り先IDを基本句IDに変換しparentを設定
                            tag.parent_id = new_parent_id
                            tag.parent = new_parent_tag
                            # children要更新?
                            break

        tags = knp_result.tag_list()
        bnsts = knp_result.bnst_list()

        # modify tag dependencies
        modify(tags, head_ids, dpnd_types, 'tag')

        # modify bnst dependencies
        modify(bnsts, head_ids, dpnd_types, 'bunsetsu')

    @staticmethod
    def sanitize_string(string: str):
        string = ''.join(string.split())  # remove space character
        string = han_to_zen(string)
        return string
