""" コーパスでアノテーションされている述語/名詞が KNP の feature を利用してどれだけ拾えるか """
# usage:
# python scripts/coverage.py /path/to/corpus/directory

import sys; sys.path.append('src')
# from pathlib import Path
from dataclasses import dataclass, field
from typing import List, NamedTuple

from pyknp import Tag
from kyoto_reader import KyotoReader, Document, Sentence, Predicate
from scorer import Measure

PRED_CASES = ('ガ', 'ヲ', 'ニ', 'ガ２')
NOUN_CASES = ('ノ',)
FEATS = ('用言', '準用言', '弱用言', '機能的用言', '用言一部', '体言', '非用言格解析', 'サ変', 'サ変動詞', '強サ変', 'サ変動詞化',
         '述語化', '判定詞', '判定詞句', '状態述語', '動態述語', '格解析なし', '省略解析なし', '省略格指定')


class Example(NamedTuple):
    document: Document
    predicate: Predicate

    def __str__(self):
        sentence: Sentence = self.document[self.predicate.sid]
        return f'{sentence.sid}, {sentence}【{self.predicate}】, ' \
               f'{"/".join(k + (":" + v if v is not True else "") for k, v in self._feats().items())}'

    def _feats(self):
        feats = self.predicate.tag.features
        return {k: v for k, v in feats.items() if k in FEATS}


@dataclass
class RetValue:
    measure_pred: Measure = field(default_factory=Measure)
    measure_noun: Measure = field(default_factory=Measure)
    examples: List[Example] = field(default_factory=list)

    def __add__(self, other: 'RetValue'):
        return RetValue(self.measure_pred + other.measure_pred,
                        self.measure_noun + other.measure_noun,
                        self.examples + other.examples)


def hiyougen(tag: Tag):
    return '非用言格解析' in tag.features
    # return '非用言格解析' in tag.features or any(('連用形名詞化' in mrph.fstring) for mrph in tag.mrph_list())


def judge_pred(tag: Tag) -> bool:
    return '用言' in tag.features or hiyougen(tag)


def judge_noun(tag: Tag) -> bool:
    return '体言' in tag.features and not hiyougen(tag)


def coverage(doc: Document) -> RetValue:
    ret = RetValue()
    for predicate in doc.get_predicates():
        ex = Example(doc, predicate)
        arguments = doc.get_arguments(predicate)
        is_pred_gold = any(arguments[case] for case in PRED_CASES)
        is_noun_gold = any(arguments[case] for case in NOUN_CASES)
        is_pred_pred = judge_pred(predicate.tag)
        is_noun_pred = judge_noun(predicate.tag)

        if is_pred_gold:
            ret.measure_pred.denom_gold += 1
        if is_pred_pred:
            ret.measure_pred.denom_pred += 1
        if is_pred_gold and is_pred_pred:
            ret.measure_pred.correct += 1
        if is_pred_gold and not is_pred_pred:
            ret.examples.append(ex)
        # if is_noun_gold and is_pred_pred:
        #     print('***', ex, '***')

        if is_noun_gold:
            ret.measure_noun.denom_gold += 1
        if is_noun_pred:
            ret.measure_noun.denom_pred += 1
        if is_noun_gold and is_noun_pred:
            ret.measure_noun.correct += 1
    return ret


def main():
    reader = KyotoReader(sys.argv[1])
    ret = RetValue()
    for doc in reader.process_all_documents():
        ret += coverage(doc)
    print('pred:')
    print(f'  precision: {ret.measure_pred.precision:.4f} ({ret.measure_pred.correct}/{ret.measure_pred.denom_pred})')
    print(f'  recall   : {ret.measure_pred.recall:.4f} ({ret.measure_pred.correct}/{ret.measure_pred.denom_gold})')
    print(f'  F        : {ret.measure_pred.f1:.4f}')
    print('noun:')
    print(f'  precision: {ret.measure_noun.precision:.4f} ({ret.measure_noun.correct}/{ret.measure_noun.denom_pred})')
    print(f'  recall   : {ret.measure_noun.recall:.4f} ({ret.measure_noun.correct}/{ret.measure_noun.denom_gold})')
    print(f'  F        : {ret.measure_noun.f1:.4f}')
    # for ex in ret.examples:
    #     print(ex)


if __name__ == '__main__':
    main()
