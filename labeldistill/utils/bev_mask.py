import numpy as np
import torch
from torch.nn.functional import one_hot
from skimage.draw import polygon


def gen_labelinput(bev_mask,
                   bev_box,
                   bev_label,
                   center,
                   width,
                   length,
                   label_box,
                   label_class):
    instance_bev_mask = np.zeros_like(bev_mask)

    mask_x = center[0]
    mask_y = center[1]
    mask_w = width * 0.5
    mask_l = length * 0.5
    yaw = label_box[6] - np.pi / 2
    W, H = bev_mask.shape

    # Compute the 4 corners of the bounding box in BEV coordinates
    corners = torch.stack([
        torch.stack([+mask_l / 2, +mask_w / 2]),
        torch.stack([+mask_l / 2, -mask_w / 2]),
        torch.stack([-mask_l / 2, -mask_w / 2]),
        torch.stack([-mask_l / 2, +mask_w / 2]),
    ]).to(dtype=torch.float32, device=label_box.device)
    cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
    rot_mat = torch.stack([
        torch.stack([cos_yaw, -sin_yaw]),
        torch.stack([sin_yaw, cos_yaw]),
    ]).to(dtype=torch.float32, device=label_box.device)
    corners = torch.matmul(corners, rot_mat.transpose(1,0))

    corners = corners + torch.stack((mask_x, mask_y))

    # Draw a polygon for the bounding box on the mask
    corners = corners.cpu().numpy().astype(np.int32)

    #########################################################################################################
    rows, cols = polygon(corners[:, 1], corners[:, 0])
    rows = np.asarray(rows).clip(0, W - 1)
    cols = np.asarray(cols).clip(0, H - 1)
    bev_mask[rows, cols] = 1

    # One hot encoding for class information
    label_class = one_hot(
        label_class.long(), num_classes=bev_label.shape[-1]).float()

    # NumPy advanced indices rely on backend-specific broadcasting when used
    # directly against a CUDA tensor.  Materialize torch indices and expanded
    # values explicitly so CUDA and PPU backends receive identical shapes.
    row_index = torch.as_tensor(rows, dtype=torch.long, device=bev_label.device)
    col_index = torch.as_tensor(cols, dtype=torch.long, device=bev_label.device)
    num_pixels = row_index.numel()

    # Fill labels into bev_label
    bev_label[row_index, col_index] = label_class.unsqueeze(0).expand(
        num_pixels, -1)
    bev_box[row_index, col_index] = label_box.to(
        dtype=bev_box.dtype).unsqueeze(0).expand(num_pixels, -1)
    #########################################################################################################

    return bev_mask, bev_box, bev_label


