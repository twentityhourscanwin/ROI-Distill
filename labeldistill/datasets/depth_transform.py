"""Shared continuous image augmentation and versioned depth rasterization."""

import numpy as np
import torch


def depth_transform(cam_depth, resize, resize_dims, crop, flip, rotate,
                    *, rasterization='nearest'):
    """Transform depth based on ida augmentation configuration.

    Args:
        cam_depth (np array): Nx3, 3: x,y,d.
        resize (float): Resize factor.
        resize_dims (list): Final dimension.
        crop (list): x1, y1, x2, y2
        flip (bool): Whether to flip.
        rotate (float): Rotation value.
        rasterization (str): Nearest valid depth, or explicit legacy replay.

    Returns:
        Tensor: Sparse depth map [H, W]; zero denotes missing supervision.
    """

    if rasterization not in {'legacy', 'nearest'}:
        raise ValueError(f'Unsupported depth rasterization: {rasterization!r}')

    H, W = resize_dims
    cam_depth[:, :2] = cam_depth[:, :2] * resize
    cam_depth[:, 0] -= crop[0]
    cam_depth[:, 1] -= crop[1]
    if flip:
        cam_depth[:, 0] = resize_dims[1] - cam_depth[:, 0]

    cam_depth[:, 0] -= W / 2.0
    cam_depth[:, 1] -= H / 2.0

    h = rotate / 180 * np.pi
    rot_matrix = [
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ]
    cam_depth[:, :2] = np.matmul(rot_matrix, cam_depth[:, :2].T).T

    cam_depth[:, 0] += W / 2.0
    cam_depth[:, 1] += H / 2.0

    if rasterization == 'nearest':
        # Check continuous coordinates first: truncating -0.2 would admit it
        # as pixel zero. Ignore missing/nonfinite/nonpositive depth samples.
        valid_mask = (
            np.isfinite(cam_depth).all(axis=1)
            & (cam_depth[:, 0] >= 0) & (cam_depth[:, 0] < W)
            & (cam_depth[:, 1] >= 0) & (cam_depth[:, 1] < H)
            & (cam_depth[:, 2] > 0)
        )
        valid_points = cam_depth[valid_mask]
        depth_coords = np.floor(valid_points[:, :2]).astype(np.int64)
        depth_map = np.full(resize_dims, np.inf, dtype=np.float32)
        # A repeated pixel must retain the visible nearest surface regardless
        # of point order. Block downsampling cannot recover overwritten points.
        np.minimum.at(
            depth_map, (depth_coords[:, 1], depth_coords[:, 0]),
            valid_points[:, 2],
        )
        depth_map[~np.isfinite(depth_map)] = 0.0
        return torch.from_numpy(depth_map)

    # Explicit legacy mode is reserved for reproducing saved experiments.
    depth_coords = cam_depth[:, :2].astype(np.int16)

    depth_map = np.zeros(resize_dims)
    valid_mask = ((depth_coords[:, 1] < resize_dims[0])
                  & (depth_coords[:, 0] < resize_dims[1])
                  & (depth_coords[:, 1] >= 0)
                  & (depth_coords[:, 0] >= 0))
    depth_map[depth_coords[valid_mask, 1],
              depth_coords[valid_mask, 0]] = cam_depth[valid_mask, 2]

    return torch.Tensor(depth_map)
