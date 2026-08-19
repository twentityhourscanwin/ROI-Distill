import numpy as np
import torch

from labeldistill.utils.bev_mask import gen_labelinput


def test_label_and_box_values_are_explicitly_expanded_per_polygon_pixel():
    bev_mask = np.zeros((16, 16), dtype=np.uint8)
    bev_box = torch.zeros((16, 16, 9), dtype=torch.float32)
    bev_label = torch.zeros((16, 16, 10), dtype=torch.float32)
    center = torch.tensor([8.0, 8.0])
    label_box = torch.tensor([0.0, 0.0, 0.0, 8.0, 4.0, 1.0, 0.0, 0.0, 0.0])

    mask, boxes, labels = gen_labelinput(
        bev_mask, bev_box, bev_label, center,
        width=torch.tensor(8.0), length=torch.tensor(4.0),
        label_box=label_box, label_class=torch.tensor(3),
    )

    selected = torch.from_numpy(mask.astype(bool))
    assert selected.any()
    assert torch.all(labels[selected][:, 3] == 1)
    assert torch.all(labels[selected].sum(dim=-1) == 1)
    assert torch.all(boxes[selected] == label_box)
