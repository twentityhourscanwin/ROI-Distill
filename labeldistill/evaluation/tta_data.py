"""Deterministic views using the existing historical-frame policy."""
import importlib.util
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "roi_tta_history_policy", ROOT / "scripts/dlc/evaluate_history.py")
_history = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _history
_spec.loader.exec_module(_history)


def partition_indices(indices, rank, world_size):
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("invalid rank/world size")
    if len(set(indices)) != len(indices):
        raise ValueError("duplicate input indices")
    return list(indices[rank::world_size])


def image_parameters(conf, scale, flip):
    height, width = conf["H"], conf["W"]
    final_h, final_w = conf["final_dim"]
    dims = (int(width * scale), int(height * scale))
    crop_h = int((1 - np.mean(conf["bot_pct_lim"])) * dims[1]) - final_h
    crop_w = int(max(0, dims[0] - final_w) / 2)
    return scale, dims, (crop_w, crop_h, crop_w + final_w, crop_h + final_h), flip, 0


class TTADataset(_history.HistoryPaddedEvalDataset):
    def __init__(self, *, view, **kwargs):
        self.tta_view = view
        super().__init__(**kwargs)
        if self.is_train:
            raise ValueError("TTA is evaluation-only")

    def sample_ida_augmentation(self):
        v = self.tta_view
        if v.image_scale == 1 and not v.image_flip:
            return super().sample_ida_augmentation()
        return image_parameters(self.ida_aug_conf, v.image_scale, v.image_flip)

    def sample_bda_augmentation(self):
        v = self.tta_view
        return v.bev_rotation, v.bev_scale, v.flip_x, v.flip_y


def make_dataset(experiment, view):
    return TTADataset(
        view=view, ida_aug_conf=experiment.ida_aug_conf,
        bda_aug_conf=experiment.bda_aug_conf, classes=experiment.class_names,
        data_root=experiment.data_root, info_paths=experiment.val_info_paths,
        is_train=False, img_conf=experiment.img_conf,
        num_sweeps=experiment.num_sweeps, sweep_idxes=experiment.sweep_idxes,
        key_idxes=experiment.key_idxes, return_depth=False, use_fusion=False)
