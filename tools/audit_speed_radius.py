"""Audit the 2x2 speed/radius experiment using cached teacher candidate graphs.

No teacher/student forward, optimizer, training or dataset mutation. Graphs
must come from the identity-BDA, exact-class, 10-sweep extraction pipeline.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import torch
import torch.nn.functional as F

from labeldistill.builders import build_experiment
from labeldistill.config import load_and_resolve_config
from labeldistill.models.distill_outputs import ProposalBatch


CONFIGS = {
    'A': 'b1_teacher_value_no_scale.yaml',
    'B': 'b2_speed_half_centered_circular.yaml',
    'C': 'r1_teacher_value_radius_cap.yaml',
    'D': 's1_speed_half_radius_cap.yaml',
}
PAIRS = [('B', 'A'), ('D', 'C'), ('C', 'A'), ('D', 'B'), ('D', 'A')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('--samples must be positive')
    if args.output.exists():
        parser.error('--output must be a new directory')
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    meta = json.loads((args.cache / 'metadata.json').read_text())
    assert meta['schema_version'] == 2 and meta['identity_bda']
    assert meta['teacher_sweep_indices'] == list(range(10))
    assert meta['split'] == 'train'

    class DummyModel:
        def __init__(self, *args, **kwargs):
            pass

    class Parts:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    bundles, parts = {}, {}
    for key, name in CONFIGS.items():
        bundle = load_and_resolve_config(
            root / 'configs/experiments' / name, project_root=root)
        assert bundle.config.derived.teacher_checkpoint_sha256 == meta['checkpoint_sha256']
        assert bundle.config.region.scaler.center_mode == 'fixed'
        bundles[key] = bundle
        parts[key] = build_experiment(bundle, model_cls=DummyModel, experiment_cls=Parts)
    config = bundles['A'].config
    info = root / config.data.root / config.data.train_info
    assert hashlib.sha256(info.read_bytes()).hexdigest() == meta['info_sha256']
    assert meta['edge_radius_m'] >= max(config.matching.trust_radius.values())
    for bundle in bundles.values():
        for key in ('matching', 'geometry', 'loss'):
            assert OmegaConf.to_container(bundle.config[key]) == OmegaConf.to_container(config[key])
    selected = sorted(np.random.default_rng(args.seed).choice(
        meta['global_samples'], min(args.samples, meta['global_samples']), replace=False).tolist())
    gt_frames, edge_frames, tokens = [], [], {}
    for path in sorted(args.cache.glob('graph_rank*_*.pt')):
        shard = torch.load(path, map_location='cpu', weights_only=False)
        for key, destination in [('gt', gt_frames), ('edges', edge_frames)]:
            array = shard[key].numpy()
            destination.append(pd.DataFrame(
                array[np.isin(array[:, 0], selected)], columns=shard['gt_columns' if key == 'gt' else 'edge_columns']))
        tokens.update({int(k): v for k, v in shard['sample_tokens'].items() if int(k) in selected})
    assert set(selected) <= set(tokens), 'Missing selected samples in cache'
    gt_frame, edge_frame = pd.concat(gt_frames), pd.concat(edge_frames)
    edge_groups = {int(k): v for k, v in edge_frame.groupby('sample_index')}
    args.output.mkdir(parents=True, exist_ok=False)
    gt_rows, scene_rows = [], []
    count_samples = 0
    for sample, group in gt_frame.groupby('sample_index', sort=True):
        sample = int(sample)
        group = group.sort_values('gt_index')
        gt = torch.tensor(group[['x', 'y', 'z', 'dx', 'dy', 'dz', 'yaw', 'vx', 'vy']].values, dtype=torch.float32)
        labels = torch.tensor(group.label.values, dtype=torch.long)
        edges = edge_groups.get(sample, edge_frame.iloc[:0])
        edges = edges.sort_values('proposal_index').drop_duplicates('proposal_index')
        boxes = torch.tensor(edges[['proposal_' + k for k in
            ['x', 'y', 'z', 'dx', 'dy', 'dz', 'yaw', 'vx', 'vy']]].values, dtype=torch.float32)
        proposals = ProposalBatch([boxes], [torch.tensor(edges.score.values, dtype=torch.float32)],
                                  [torch.tensor(edges.label.values, dtype=torch.long)])
        match = parts['A'].matcher(proposals, [gt], [labels])
        assert torch.equal(match.effective_gt_mask[0], torch.tensor(group.effective.values, dtype=torch.bool))
        results = {k: part.scaler.forward(match) if bundles[k].config.region.scaler.enabled
                   else match for k, part in parts.items()}
        for result in results.values():
            assert result.teacher_values is match.teacher_values
            assert result.matched_proposal_indices is match.matched_proposal_indices
            assert torch.equal(result.gt_boxes[0][:, :3], match.gt_boxes[0][:, :3])
        active = torch.nonzero(match.effective_gt_mask[0] & (match.teacher_values[0] > 0)).flatten().tolist()
        merged = {k: torch.zeros(128, 128) for k in CONFIGS}
        for i in active:
            q = float(match.teacher_values[0][i])
            masks = {k: parts[k].feature_loss_reducer._draw_base_mask(
                result.gt_boxes[0][i], device='cpu') for k, result in results.items()}
            assert all(m is not None for m in masks.values())
            assert torch.all(masks['D'] >= masks['C'])
            for k, mask in masks.items():
                merged[k] = torch.maximum(merged[k], q * mask)
            speed = float(torch.linalg.vector_norm(gt[i, 7:9]))
            for new, old in PAIRS:
                gt_rows.append(dict(sample_index=sample, gt_index=int(group.iloc[i].gt_index),
                    class_name=config.classes.names[int(labels[i])], speed=speed,
                    comparison=new + '/' + old, changed=not torch.equal(masks[new], masks[old]),
                    relative_l1=float((masks[new] - masks[old]).abs().sum() / masks[old].sum()), q=q))
        if active:
            for level in (128, 64):
                masks = merged if level == 128 else {k: F.interpolate(
                    m[None, None], (64, 64), mode='bilinear', align_corners=True)[0, 0]
                    for k, m in merged.items()}
                for new, old in PAIRS:
                    a, b = masks[new], masks[old]
                    scene_rows.append(dict(sample_index=sample, level=level,
                        comparison=new + '/' + old, changed=not torch.equal(a, b),
                        relative_l1=float((a-b).abs().sum()/b.sum()),
                        normalized_l1=float((a/a.sum()-b/b.sum()).abs().sum())))
        count_samples += 1
        if count_samples % 128 == 0:
            print('Processed', count_samples, 'samples with GT', flush=True)
    frame, scenes = pd.DataFrame(gt_rows), pd.DataFrame(scene_rows)
    if frame.empty:
        raise RuntimeError('No positive-q GTs in selected samples')
    summary = frame.groupby('comparison').agg(
        n=('changed', 'size'), changed_rate=('changed', 'mean'), relative_l1=('relative_l1', 'mean'))
    fast = frame[frame.speed >= 0.8].groupby('comparison').agg(
        n=('changed', 'size'), changed_rate=('changed', 'mean'))
    per_class = frame.groupby(['comparison', 'class_name']).agg(
        n=('changed', 'size'), changed_rate=('changed', 'mean'))
    scene_summary = scenes.groupby(['comparison', 'level']).agg(
        n=('changed', 'size'), changed_rate=('changed', 'mean'),
        relative_l1=('relative_l1', 'mean'), normalized_l1=('normalized_l1', 'mean'))
    frame.to_csv(args.output / 'per_gt.csv.gz', index=False)
    scenes.to_csv(args.output / 'per_scene.csv', index=False)
    for name, table in [('summary', summary), ('moving_summary', fast),
                        ('per_class', per_class), ('scene_summary', scene_summary)]:
        table.to_csv(args.output / (name + '.csv'))
    manifest = dict(selected_samples=selected, sample_tokens=tokens, samples_with_gt=count_samples,
        seed=args.seed, cache=str(args.cache.resolve()), cache_metadata=meta,
        configs={k: OmegaConf.to_container(v.config, resolve=True) for k, v in bundles.items()},
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        git_status=subprocess.check_output(['git', 'status', '--short'], cwd=root, text=True),
        source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), root/'labeldistill/refine_head/target_assigner/raw_gaussian_feature_loss.py',
                      root/'labeldistill/refine_head/target_assigner/velocity_only_half_scaler.py',
                      root/'labeldistill/refine_head/target_assigner/roi_distill.py']},
        limitation='Identity BDA, fixed sampled frames; geometry/masks only, no student feature or training result.')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(summary.to_string(), flush=True)
    print(scene_summary.to_string(), flush=True)


if __name__ == '__main__':
    main()
