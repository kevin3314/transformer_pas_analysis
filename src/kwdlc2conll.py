import argparse
import sys
import os

from typing import List, Tuple, Dict

from utils.constants import TOKENS, CASES, DEP_TYPES
from data_loader.reader import Document
from data_loader.reader import KyotoCorpus
from data_loader.reader import KyotoPASGoldExtractor
from data_loader.reader import KyotoPAS

# from preprocess.util import X, D, Y, PE, SM, SD, SP, SY
# from preprocess.util import flatten, build_idmaps
# from preprocess.basic_information_extractors import extract_input, extract_dependency, extract_target
# from preprocess.feature_extractors import extract_path_embedding
# from preprocess.feature_extractors import extract_string_match
# from preprocess.feature_extractors import extract_sentence_distance
# from preprocess.feature_extractors import extract_selectional_preference
# from preprocess.feature_extractors import extract_synonym_dictionary
# from preprocess.selectional_preference.data_loader import SelectionalPreferenceData
# from preprocess.synonym_dictionary.data import SynonymDictionaryData


# def build_io(document: Document):
#     pas_extractor = KyotoPASGoldExtractor()
#     pas_extractor.preprocess_document(document)
#
    # sid2sid, dmid2sid, dmid2tid, dmid2ctid, stid2ctid, ctid2rep = build_idmaps(document)
    #
    # x_doc = extract_input(document)
    # d_doc = extract_dependency(document, pas_extractor, stid2ctid, sid2sid)
    # y_doc = extract_target(document, pas_extractor, dmid2sid, dmid2tid, dmid2ctid)
    # pe_doc = extract_path_embedding(document, pas_extractor, dmid2sid, dmid2tid, dmid2ctid, sid2sid)
    # sm_doc = extract_string_match(x_doc)
    # sd_doc = extract_sentence_distance(x_doc)
    # sp_doc = extract_selectional_preference(sp_data, x_doc, d_doc, ctid2rep)
    # sy_doc = extract_synonym_dictionary(sy_data.data, x_doc)
    # return x_doc, d_doc, y_doc, pe_doc, sm_doc, sd_doc, sp_doc, sy_doc


def load_data(path: str):
    corpus = KyotoCorpus(path)
    for document in corpus.load_files():
        pas_extractor = KyotoPASGoldExtractor()
        pas_extractor.preprocess_document(document)
        print(pas_extractor.extract_from_doc(document))

        # with open(f'data/out/{document.doc_id}.conll', 'w') as writer:
        with sys.stdout as writer:
            writer.write(f'# A-ID:{document.doc_id}\n')
            dmid = 0
            for sentence in document:
                items = ['_'] * 8
                items[4] = sentence.sid
                pas_list = pas_extractor.extract_from_sent(sentence, document)
                dmid2pas: Dict[int, KyotoPAS] = {pas.dmid: pas for pas in pas_list}
                for tag in sentence.tag_list():
                    items[2] = str(tag.parent_id) + tag.dpndtype
                    for idx, mrph in enumerate(tag.mrph_list()):  # 冗長
                        items[0] = str(dmid + 1)
                        items[1] = mrph.midasi
                        if dmid in dmid2pas:
                            # type_dic = dmid2pas[dmid].type_dic
                            _, _, eval_struct = dmid2pas[dmid].dump_latent_ids(
                                document,
                                intersent_args="dump",
                                case_type="core"
                            )
                            # for type_, anaphors in eval_struct.items():
                            #     anaphor = anaphors[0]
                            arguments: List[str] = []
                            cases = ['ガ', 'ヲ', 'ニ', 'ガ２']
                            for case in cases:
                                if case in eval_struct:
                                    anaphor: dict = eval_struct[case][0]
                                    if anaphor['is_special']:
                                        arguments.append(anaphor['anaphor'])
                                    else:
                                        tdmid = anaphor['dmid']
                                        dep_type = anaphor['type']
                                        arguments.append(str(tdmid + 1) + '%C' if dep_type == 'overt' else '')
                                else:
                                    arguments.append('NULL')
                            items[5] = ','.join(f'{case}:{arg}' for case, arg in zip(cases, arguments))
                            items[6] = 'NA'

                        writer.write('\t'.join(items) + '\n')
                        items = ['_'] * 8
                        dmid += 1
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('TRAIN', help='path to directory where train data exists')
    parser.add_argument('VALID', help='path to directory where validation data exists')
    parser.add_argument('TEST', help='path to directory where test data exists')
    parser.add_argument('OUT', help='path to directory where dataset to be located')

    args = parser.parse_args()

    # make directories to save dataset
    os.makedirs(args.OUT, exist_ok=True)

    # load data
    # load_data(args.TRAIN)
    # load_data(args.VALID)
    load_data(args.TEST)

    # if 'all' in target or 'vocab' in target:
    #     save_vocab(os.path.join(args.OUT, 'words.txt'), words)
    #     save_vocab(os.path.join(args.OUT, 'poss.txt'), poss)
    #     save_vocab(os.path.join(args.OUT, 'sposs.txt'), sposs)
    #     save_vocab(os.path.join(args.OUT, 'conjs.txt'), conjs)
    #
    # if 'all' in target or 'input' in target:
    #     output_input(args.OUT, 'train', train_xs, words, poss, sposs, conjs)
    #     output_input(args.OUT, 'valid', valid_xs, words, poss, sposs, conjs)
    #     output_input(args.OUT, 'test', test_xs, words, poss, sposs, conjs)
    #
    # if 'all' in target or 'dep' in target:
    #     output_dependency(args.OUT, 'train', train_ds)
    #     output_dependency(args.OUT, 'valid', valid_ds)
    #     output_dependency(args.OUT, 'test', test_ds)
    #
    #
    #     output_synonym_dictionary(args.OUT, 'test', test_sys)
    #
    # output_config(args.OUT, 'train', train_xs, train_pes, train_sms)
    # output_config(args.OUT, 'valid', valid_xs, valid_pes, valid_sms)
    # output_config(args.OUT, 'test', test_xs, test_pes, test_sms)


if __name__ == '__main__':
    main()
