import sys
import pytest
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent / 'src'))

INF = 2 ** 10


@pytest.fixture()
def fixture_input_tensor():
    # コイン ト ##ス を 行う [NULL] [NA]
    # (b, seq, case, seq) = (1, 7, 4, 7)
    input_tensor = torch.tensor(
        [
            [
                [
                    [-INF, -INF, -INF, -INF, -INF, -INF, -INF],  # ガ
                    [-INF, -INF, -INF, -INF, -INF, -INF, -INF],  # ヲ
                    [-INF, -INF, -INF, -INF, -INF, -INF, -INF],  # ニ
                    [-INF, -INF, -INF, -INF, -INF, -INF, -0.2],  # coref
                ],  # コイン
                [
                    [-INF, -INF, -INF, -INF, -INF, -INF, -INF],  # ガ
                    [-INF, -INF, -INF, -INF, -INF, -INF, -INF],  # ヲ
                    [-INF, -INF, -INF, -INF, -INF, -INF, -INF],  # ニ
                    [-0.5, -INF, -INF, -INF, -INF, -INF, -0.1],  # coref
                ],  # ト
                [
                    [-INF, -INF, -INF, -INF, -INF, -INF, -INF],  # ガ
                    [-INF, -INF, -INF, -INF, -INF, -INF, -INF],  # ヲ
                    [-INF, -INF, -INF, -INF, -INF, -INF, -INF],  # ニ
                    [-INF, -INF, -INF, -INF, -INF, -INF, -INF],  # coref
                ],  # ##ス
                [
                    [-INF, -INF, -INF, -INF, -INF, -INF, -INF],  # ガ
                    [-INF, -INF, -INF, -INF, -INF, -INF, -INF],  # ヲ
                    [-INF, -INF, -INF, -INF, -INF, -INF, -INF],  # ニ
                    [-INF, -INF, -INF, -INF, -INF, -INF, -INF],  # coref
                ],  # を
                [
                    [-0.2, -0.5, -INF, -INF, -INF, 0.50, -INF],  # ガ
                    [-0.5, -0.4, -INF, -INF, -INF, -0.9, -INF],  # ヲ
                    [-0.3, -0.3, -INF, -INF, -INF, -0.1, -INF],  # ニ
                    [-INF, -INF, -INF, -INF, -INF, -INF, -INF],  # coref
                ],  # 行う
                [
                    [-INF, -INF, -INF, -INF, -INF, -INF, -INF],  # ガ
                    [-INF, -INF, -INF, -INF, -INF, -INF, -INF],  # ヲ
                    [-INF, -INF, -INF, -INF, -INF, -INF, -INF],  # ニ
                    [-INF, -INF, -INF, -INF, -INF, -INF, -INF],  # coref
                ],  # [NULL]
                [
                    [-INF, -INF, -INF, -INF, -INF, -INF, -INF],  # ガ
                    [-INF, -INF, -INF, -INF, -INF, -INF, -INF],  # ヲ
                    [-INF, -INF, -INF, -INF, -INF, -INF, -INF],  # ニ
                    [-INF, -INF, -INF, -INF, -INF, -INF, -INF],  # coref
                ],  # [NA]
            ],  # コイントスを行う
        ]
    )
    yield input_tensor
