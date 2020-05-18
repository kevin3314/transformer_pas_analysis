
import torch

from model.sub.conditional_model import AttentionConditionalModel

INF = 2 ** 10


def test_hard_output_aggr(fixture_input_tensor: torch.Tensor):
    mask = fixture_input_tensor != -INF
    output_tensor = AttentionConditionalModel._hard_output_aggr(fixture_input_tensor, mask)

    # (b, seq, 1+case*2, seq) = (1, 7, 9, 7)
    expected_output_tensor = torch.tensor(
        [
            [
                [
                    [1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ],  # コイン
                [
                    [1, 1, 1, 1, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ],  # ト
                [
                    [1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ],  # ##ス
                [
                    [1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ],  # を
                [
                    [1, 0, 1, 1, 1, 0, 1],
                    [0, 0, 0, 0, 0, 0.5, 0],
                    [0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0.5, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ],  # 行う
                [
                    [1, 1, 1, 1, 0, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0.5, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0.5, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ],  # [NULL]
                [
                    [0, 0, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0],
                ],  # [NA]
            ],
        ]
    )

    # test_hard_output_aggr
    for idx, (output, expected) in enumerate(zip(output_tensor[0], expected_output_tensor[0])):
        assert (output - expected).abs().sum().item() < 1e-3, f'failed at {idx}'


def test_soft_output_aggr(fixture_input_tensor: torch.Tensor):
    mask = fixture_input_tensor != -INF
    output_tensor = AttentionConditionalModel._soft_output_aggr(fixture_input_tensor, mask)

    # (b, seq, 1+case*2, seq) = (1, 7, 9, 7)
    expected_output_tensor = torch.tensor(
        [
            [
                [
                    [1, 0, 1, 1, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1],  # [0, 0, 0, 0, 0, 0, -0.2],
                    [0, 0, 0, 0, 0.378, 0, 0],  # [0, 0, 0, 0, -0.5, 0, 0],
                    [0, 0, 0, 0, 0.280, 0, 0],  # [0, 0, 0, 0, -0.3, 0, 0],
                    [0, 0, 0, 0, 0.342, 0, 0],  # [0, 0, 0, 0, -0.2, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0],
                ],  # コイン
                [
                    [0, 1, 1, 1, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 1],  # [0, 0, 0, 0, 0, 0, -0.1],
                    [0, 0, 0, 0, 0.3006, 0, 0],  # [0, 0, 0, 0, -0.8, 0, 0],
                    [0, 0, 0, 0, 0.3322, 0, 0],  # [0, 0, 0, 0, -0.3, 0, 0],
                    [0, 0, 0, 0, 0.3672, 0, 0],  # [0, 0, 0, 0, -0.5, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ],  # ト
                [
                    [1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ],  # ##ス
                [
                    [1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ],  # を
                [
                    [0, 0, 1, 1, 1, 0, 1],
                    [0.378, 0.3006, 0, 0, 0, 0.5570, 0],  # [-0.5, -0.8, 0, 0, 0, -0.9, 0],
                    [0.280, 0.3322, 0, 0, 0, 0.1373, 0],  # [-0.3, -0.3, 0, 0, 0, -0.1, 0],
                    [0.342, 0.3672, 0, 0, 0, 0.3057, 0],  # [-0.2, -0.5, 0, 0, 0, 0.10, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ],  # 行う
                [
                    [1, 1, 1, 1, 0, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0.5570, 0, 0],  # [0, 0, 0, 0, -0.9, 0, 0],
                    [0, 0, 0, 0, 0.1373, 0, 0],  # [0, 0, 0, 0, -0.1, 0, 0],
                    [0, 0, 0, 0, 0.3057, 0, 0],  # [0, 0, 0, 0, 0.10, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ],  # [NULL]
                [
                    [0, 0, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0],  # [-0.2, -0.1, 0, 0, 0, 0, 0],
                ],  # [NA]
            ],
        ]
    )

    # test_soft_output_aggr
    for idx, (output, expected) in enumerate(zip(output_tensor[0], expected_output_tensor[0])):
        assert (output - expected).abs().sum().item() < 1e-3, f'failed at {idx}'


def test_confidence_output_aggr(fixture_input_tensor: torch.Tensor):
    mask = fixture_input_tensor != -INF
    output_tensor = AttentionConditionalModel._confidence_output_aggr(fixture_input_tensor, mask)

    # (b, seq, 1+case*2, seq) = (1, 7, 9, 7)
    expected_output_tensor = torch.tensor(
        [
            [
                [
                    [1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ],  # コイン
                [
                    [1, 1, 1, 1, 0.6018, 1, 0.4013],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0.5987],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0.3982, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ],  # ト
                [
                    [1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ],  # ##ス
                [
                    [1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ],  # を
                [
                    [1, 0.6018, 1, 1, 1, 0.0845, 1],
                    [0, 0, 0, 0, 0, 0.5363, 0],
                    [0, 0.3982, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0.3792, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ],  # 行う
                [
                    [1, 1, 1, 1, 0.0845, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0.5363, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0.3792, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ],  # [NULL]
                [
                    [0, 0.4013, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [1, 0.5987, 0, 0, 0, 0, 0],
                ],  # [NA]
            ],
        ]
    )

    # test_confidence_output_aggr
    for idx, (output, expected) in enumerate(zip(output_tensor[0], expected_output_tensor[0])):
        assert (output - expected).abs().sum().item() < 1e-3, f'failed at {idx}'
