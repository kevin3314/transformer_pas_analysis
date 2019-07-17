import pytest
from pathlib import Path

from kwdlc_reader import KWDLCReader
from kwdlc_reader import ALL_CASES, ALL_COREFS, ALL_EXOPHORS


data_dir = Path(__file__).parent / 'data'


@pytest.fixture()
def fixture_kwdlc_reader():
    file_path = data_dir / 'w201106-0000060050.knp'
    yield KWDLCReader(file_path,
                      target_cases=ALL_CASES,
                      target_corefs=ALL_COREFS,
                      target_exophors=ALL_EXOPHORS)
