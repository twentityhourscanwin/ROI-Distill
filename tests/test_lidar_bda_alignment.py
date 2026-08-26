import torch

from labeldistill.datasets.nusc_det_dataset_lidar import (
    apply_bda_to_lidar_points_,
    bev_transform,
)


def test_lidar_xyz_uses_the_same_bda_transform_as_gt_center():
    original_xyz = torch.tensor([1.5, -2.0, 0.8])
    points = torch.tensor([[1.5, -2.0, 0.8, 17.0, 0.25]])
    gt_boxes = torch.tensor([[1.5, -2.0, 0.8, 4.0, 2.0, 1.5, 0.3, 1.0, -0.5]])

    transformed_gt, bda_rot = bev_transform(
        gt_boxes.clone(), rotate_angle=17.0, scale_ratio=1.05,
        flip_dx=True, flip_dy=False)
    transformed_points = apply_bda_to_lidar_points_(
        points.clone(), bda_rot)

    expected_xyz = bda_rot @ original_xyz
    assert torch.allclose(transformed_points[0, :3], expected_xyz)
    assert torch.allclose(transformed_points[0, :3], transformed_gt[0, :3])
    assert torch.equal(transformed_points[0, 3:], points[0, 3:])


def test_lidar_bda_scales_z_instead_of_leaving_it_unchanged():
    points = torch.tensor([[0.0, 0.0, 2.0, 1.0, 0.0]])
    _, bda_rot = bev_transform(
        torch.zeros((0, 9)), rotate_angle=0.0, scale_ratio=0.95,
        flip_dx=False, flip_dy=False)

    apply_bda_to_lidar_points_(points, bda_rot)

    assert points[0, 2] == torch.tensor(1.9)
