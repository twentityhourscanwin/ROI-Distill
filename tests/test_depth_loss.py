import pytest
import torch
import torch.nn.functional as F

from labeldistill.exps.nuscenes.base_exp import LabelDistillModel


def _depth_experiment():
    experiment = LabelDistillModel.__new__(LabelDistillModel)
    experiment.downsample_factor = 2
    experiment.dbound = [2.0, 6.0, 1.0]
    experiment.depth_channels = 4
    return experiment


def test_r50_depth_loss_uses_continuous_gaussian_targets_and_kl():
    experiment = _depth_experiment()
    # The first 2x2 patch has nearest depth 2.5 -> continuous bin 0.5.
    # The second patch has no projected LiDAR depth and must stay background.
    depth_labels = torch.tensor([[[
        [0.0, 2.5, 0.0, 0.0],
        [3.5, 0.0, 0.0, 0.0],
    ]]])
    depth_indices, fg_mask = experiment.get_downsampled_gt_depth(depth_labels)
    assert depth_indices.tolist() == pytest.approx([0.5, 0.0])
    assert fg_mask.tolist() == [True, False]

    soft_labels = experiment.create_soft_labels(
        depth_indices, fg_mask, sigma=1.5)
    assert soft_labels.shape == (1, 4)
    assert soft_labels.sum(dim=1).item() == pytest.approx(1.0)
    assert soft_labels[0, 0].item() == pytest.approx(
        soft_labels[0, 1].item())

    logits = torch.tensor(
        [[[[2.0, -1.0]], [[1.0, -1.0]], [[0.0, -1.0]], [[-1.0, -1.0]]]],
        requires_grad=True,
    )
    loss = experiment.get_depth_loss(depth_labels, logits, sigma=1.5)
    expected = 3.0 * F.kl_div(
        F.log_softmax(logits[0, :, 0, 0].float(), dim=0),
        soft_labels[0].float(),
        reduction='sum',
    )
    assert loss.item() == pytest.approx(expected.item())
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad[0, :, 0, 1]) == 0


def test_depth_loss_returns_graph_connected_zero_without_foreground():
    experiment = _depth_experiment()
    depth_labels = torch.zeros(1, 1, 2, 2)
    logits = torch.randn(1, 4, 1, 1, requires_grad=True)

    loss = experiment.get_depth_loss(depth_labels, logits)
    assert loss.item() == 0.0
    loss.backward()
    assert torch.count_nonzero(logits.grad) == 0


def test_gaussian_depth_sigma_must_be_positive():
    experiment = _depth_experiment()
    with pytest.raises(ValueError, match='sigma must be positive'):
        experiment.create_soft_labels(
            torch.tensor([0.5]), torch.tensor([True]), sigma=0.0)
