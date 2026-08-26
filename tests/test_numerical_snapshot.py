import torch

from labeldistill.acceptance.snapshot import compare_snapshots, tensor_summary


def test_snapshot_comparison_reports_tensor_path_and_delta():
    left = {"losses_raw": {"feature": torch.tensor(1.0)}}
    right = {"losses_raw": {"feature": torch.tensor(1.1)}}
    report = compare_snapshots(left, right, atol=1e-6, rtol=1e-6)
    assert report["passed"] is False
    assert report["failed"] == ["losses_raw.feature"]
    assert report["tensors"]["losses_raw.feature"]["max_abs"] > 0


def test_snapshot_summary_has_content_hash():
    summary = tensor_summary({"mask": torch.arange(4).reshape(1, 2, 2)})
    assert summary["mask"]["shape"] == [1, 2, 2]
    assert len(summary["mask"]["sha256"]) == 64
