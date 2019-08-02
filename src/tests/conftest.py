import pytest
from pathlib import Path

from kwdlc_reader import KyotoCorpus
from kwdlc_reader import ALL_CASES, ALL_COREFS, ALL_EXOPHORS


data_dir = Path(__file__).parent / 'data'


@pytest.fixture()
def fixture_kwdlc_readers():
    kyoto_corpus = KyotoCorpus(data_dir,
                               target_cases=ALL_CASES,
                               target_corefs=ALL_COREFS,
                               target_exophors=ALL_EXOPHORS)
    yield list(kyoto_corpus.load_files())