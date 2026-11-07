"""Evaluate camera checkpoints with an explicit history padding policy."""
import argparse
import importlib.util
import os
from pathlib import Path
import sys

from labeldistill.datasets.nusc_det_dataset import NuscDetDataset as OriginalDataset

ROOT = Path(__file__).resolve().parents[2]
POLICY_ENV = "ROI_EVAL_HISTORY_FALLBACK"


class HistoryFrameSelection:
    @staticmethod
    def indices(infos, idx, key_idxes, mode):
        if mode not in {"previous_valid", "current"}:
            raise ValueError(f"Unsupported history fallback: {mode!r}")
        last_idx = idx
        selected = []
        for offset in key_idxes:
            candidate = idx + offset
            if candidate < 0 or infos[candidate]["scene_token"] != infos[idx]["scene_token"]:
                candidate = idx if mode == "current" else last_idx
            else:
                last_idx = candidate
            selected.append(candidate)
        return selected


class HistoryPaddedEvalDataset(OriginalDataset):
    def __getitem__(self, idx):
        if self.is_train or self.return_depth or self.use_fusion or self.sweeps_idx:
            raise ValueError("History override supports camera-only keyframe evaluation.")
        self._history_sample_idx = self.sample_indices[idx] if self.use_cbgs else idx
        try:
            return super().__getitem__(idx)
        finally:
            del self._history_sample_idx

    def get_image(self, cam_infos, cams, lidar_infos=None):
        indices = HistoryFrameSelection.indices(
            self.infos, self._history_sample_idx, self.key_idxes, "previous_valid")
        selected = [self.infos[idx]["cam_infos"] for idx in indices]
        return super().get_image(selected, cams, lidar_infos)


def install_policy(mode):
    from labeldistill.exps.nuscenes import base_exp
    if mode == "current":
        base_exp.NuscDetDataset = OriginalDataset
    elif mode == "previous_valid":
        base_exp.NuscDetDataset = HistoryPaddedEvalDataset
    else:
        raise ValueError(f"Unsupported history fallback: {mode!r}")


def load_native_evaluator():
    spec = importlib.util.spec_from_file_location(
        "native_labeldistill_evaluate", ROOT / "tools/evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--history-fallback", choices=["previous_valid", "current"],
                        default=os.environ.get(POLICY_ENV, "current"))
    args, native_args = parser.parse_known_args()
    if args.history_fallback not in {"previous_valid", "current"}:
        parser.error("Invalid inherited history fallback.")
    # Lightning relaunches sys.argv[0]; each rank inherits this policy.
    os.environ[POLICY_ENV] = args.history_fallback
    sys.argv = [sys.argv[0], *native_args]
    print(f"eval_history_fallback={args.history_fallback} local_rank={os.environ.get('LOCAL_RANK', '0')}", flush=True)
    install_policy(args.history_fallback)
    load_native_evaluator().main()


if __name__ == "__main__":
    main()
