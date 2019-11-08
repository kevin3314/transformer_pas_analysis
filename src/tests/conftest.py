import pytest
from pathlib import Path

from kwdlc_reader import KWDLCReader
from kwdlc_reader import ALL_CASES, ALL_COREFS


data_dir = Path(__file__).parent / 'data'


@pytest.fixture()
def fixture_kwdlc_reader():
    reader = KWDLCReader(data_dir,
                         target_cases=ALL_CASES,
                         target_corefs=ALL_COREFS)
    yield reader
