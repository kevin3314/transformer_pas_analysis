import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional

from pyknp import Tag

from kwdlc_reader import KWDLCReader, Argument


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class Score:
    def __init__(self, cases: List[str]):
        self.cases = cases
        self.measures: Dict[str, Measure] = {case: Measure() for case in cases}

    def __getitem__(self, case: Optional[str] = None) -> Optional['Measure']:
        if case is not None:
            if case not in self.cases:
                logger.warning(f'unknown case: {case}')
                return None
            return self.measures[case]
        else:
            return sum(self.measures.values())

    def print_result(self):
        for case, measure in self.measures.items():
            print(f'--{case}格--')
            print(f'precision: {measure.precision:.3} ({measure.denom_pred})')
            print(f'recall   : {measure.recall:.3} ({measure.denom_gold})')
            print(f'F        : {measure.f1:.3}')


class Measure:
    def __init__(self, denom_pred: int = 0, denom_gold: int = 0, tp: int = 0):
        self.denom_pred = denom_pred
        self.denom_gold = denom_gold
        self.tp = tp

    def __add__(self, other: 'Measure'):
        return Measure(self.denom_pred + other.denom_pred, self.denom_gold + other.denom_gold, self.tp + other.tp)

    @property
    def precision(self) -> float:
        if self.denom_pred == 0:
            logger.warning('zero division at precision')
            return 0
        return self.tp / self.denom_pred

    @property
    def recall(self) -> float:
        if self.denom_gold == 0:
            logger.warning('zero division at recall')
            return 0
        return self.tp / self.denom_gold

    @property
    def f1(self) -> float:
        if self.denom_pred + self.denom_gold == 0:
            logger.warning('zero division at f1')
            return 0
        return 2 * self.tp / (self.denom_pred + self.denom_gold)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction-dir', default=None, type=str,
                        help='path to directory where system output KWDLC files exist (default: None)')
    parser.add_argument('--gold-dir', default=None, type=str,
                        help='path to directory where gold KWDLC files exist (default: None)')
    parser.add_argument('--case-string', type=str, default='ガ,ヲ,ニ,ガ２',
                        help='Case strings. Separate by ","')
    parser.add_argument('--exophors', type=str, default='著者,読者,不特定:人',
                        help='Special tokens. Separate by ",".')
    args = parser.parse_args()

    reader_pred = KWDLCReader(
        Path(args.prediction_dir),
        target_cases=args.case_string.split(','),
        target_corefs=['=', '=構', '=≒'],
        target_exophors=args.exophors.split(','),
        extract_nes=False
    )
    reader_gold = KWDLCReader(
        Path(args.gold_dir),
        target_cases=args.case_string.split(','),
        target_corefs=['=', '=構', '=≒'],
        target_exophors=args.exophors.split(','),
        extract_nes=False
    )
    assert sorted(reader_pred.did2path.keys()) == sorted(reader_gold.did2path.keys())
    cases = reader_gold.target_cases

    score = Score(cases)

    for doc_id in reader_gold.did2path.keys():
        document_pred = reader_pred.process_document(doc_id)
        document_gold = reader_gold.process_document(doc_id)
        predicates_pred = document_pred.get_predicates()
        predicates_gold = document_gold.get_predicates()
        dtid2pred_pred = {document_pred.tag2dtid[tag]: tag for tag in predicates_pred}
        dtid2pred_gold = {document_gold.tag2dtid[tag]: tag for tag in predicates_gold}

        # calculate precision
        for dtid, predicate_pred in dtid2pred_pred.items():
            if dtid in dtid2pred_gold:
                predicate_gold = dtid2pred_gold[dtid]
                arguments_pred = document_pred.get_arguments(predicate_pred)
                arguments_gold = document_gold.get_arguments(predicate_gold)
                for case in cases:
                    argument_pred: List[Argument] = arguments_pred[case]
                    argument_gold: List[Argument] = arguments_gold[case]
                    if not argument_pred:
                        continue
                    assert len(argument_pred) == 1
                    arg = argument_pred[0]
                    # if any(arg in argument_gold for arg in argument_pred):
                    if arg in argument_gold:
                        score.measures[case].tp += 1
                    score.measures[case].denom_pred += 1

        # calculate recall
        for predicate_gold in dtid2pred_gold.values():
            arguments_gold = document_gold.get_arguments(predicate_gold)
            for case in cases:
                argument_gold: List[Argument] = arguments_gold[case]
                if argument_gold:
                    print(case + ': ' + argument_gold[0].midasi)
                    score.measures[case].denom_gold += 1

    score.print_result()


if __name__ == '__main__':
    main()
