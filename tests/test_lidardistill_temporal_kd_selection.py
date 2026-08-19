import pytest
import torch

from labeldistill.models.lidardistill import select_temporal_kd_channels


def make_frame(value, channels=4):
    return torch.full((1, channels, 2, 2), float(value))


def test_selects_current_t2_and_half_t6_without_reordering_head_input():
    frame_channels = 4
    temporal_feature = torch.cat(
        [make_frame(value, frame_channels) for value in range(5)],
        dim=1,
    )

    selected = select_temporal_kd_channels(
        temporal_feature,
        frame_channels,
    )

    expected = torch.cat(
        [
            make_frame(0, frame_channels),
            make_frame(1, frame_channels),
            make_frame(3, frame_channels)[:, :frame_channels // 2],
        ],
        dim=1,
    )
    assert selected.shape[1] == temporal_feature.shape[1] // 2
    assert torch.equal(selected, expected)


def test_rejects_non_five_frame_layout():
    with pytest.raises(ValueError, match='5 x 4'):
        select_temporal_kd_channels(
            torch.zeros(1, 16, 2, 2),
            frame_channels=4,
        )


def test_rejects_odd_frame_channel_count():
    with pytest.raises(ValueError, match='must be even'):
        select_temporal_kd_channels(
            torch.zeros(1, 15, 2, 2),
            frame_channels=3,
        )
