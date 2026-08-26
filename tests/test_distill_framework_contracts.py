from copy import deepcopy

import torch

from labeldistill.builders import build_experiment
from labeldistill.config import load_and_resolve_config
from labeldistill.exps.nuscenes import base_exp
from labeldistill.models.distill_outputs import (
    LabelDistillOutput,
    ProposalBatch,
    StudentOutput,
    TeacherOutput,
)
from labeldistill.layers.heads.kd_head import _align_gt_values_to_tasks


def test_base_experiment_can_skip_model_build_and_owns_config_copies(
        monkeypatch, tmp_path):
    source_backbone = deepcopy(base_exp.backbone_conf)
    source_head = deepcopy(base_exp.head_conf)

    def fail_if_built(*args, **kwargs):
        raise AssertionError('BaseBEVDepth should not be constructed')

    monkeypatch.setattr(base_exp, 'BaseBEVDepth', fail_if_built)
    experiment = base_exp.LabelDistillModel(
        build_model=False,
        backbone_conf=source_backbone,
        head_conf=source_head,
        default_root_dir=str(tmp_path),
    )

    assert experiment.model is None
    experiment.backbone_conf['output_channels'] = 999
    experiment.head_conf['train_cfg']['max_objs'] = 1
    assert source_backbone['output_channels'] != 999
    assert source_head['train_cfg']['max_objs'] != 1
    assert base_exp.backbone_conf['output_channels'] != 999
    assert base_exp.head_conf['train_cfg']['max_objs'] != 1


def test_structured_distillation_output_exposes_named_contracts():
    proposals = ProposalBatch(
        boxes=[torch.zeros((0, 9))],
        scores=[torch.zeros(0)],
        labels=[torch.zeros(0, dtype=torch.long)],
    )
    student = StudentOutput(
        raw_preds='student-preds',
        depth=None,
        distill_features=[torch.zeros((1, 1, 1, 1))],
    )
    teacher = TeacherOutput(
        raw_preds='teacher-preds',
        backbone_features=[torch.zeros((1, 1, 1, 1))],
        neck_features=[torch.zeros((1, 1, 1, 1))],
        proposals=proposals,
    )
    outputs = LabelDistillOutput(student=student, teacher=teacher)

    assert outputs.student.raw_preds == 'student-preds'
    assert outputs.teacher.raw_preds == 'teacher-preds'
    assert outputs.teacher.proposals is proposals


def test_response_valid_masks_follow_centerhead_task_and_class_order():
    gt_labels = [torch.tensor([8, 1, 2, 1, 0])]
    valid = [torch.tensor([True, False, True, True, True])]
    class_names = [
        ['car'],
        ['truck', 'construction_vehicle'],
        ['bus', 'trailer'],
        ['barrier'],
        ['motorcycle', 'bicycle'],
        ['pedestrian', 'traffic_cone'],
    ]
    target_masks = [torch.zeros((1, 6), dtype=torch.uint8) for _ in class_names]

    aligned = _align_gt_values_to_tasks(
        gt_labels, valid, class_names, target_masks)

    assert aligned[0][0, :1].tolist() == [1.0]
    # CenterHead groups task GTs by class: truck entries first, then
    # construction_vehicle entries, preserving order within each class.
    assert aligned[1][0, :3].tolist() == [0.0, 1.0, 1.0]
    assert aligned[5][0, :1].tolist() == [1.0]


def test_j4_builder_builds_only_the_final_structured_model(monkeypatch):
    calls = []

    def fail_base_model(*args, **kwargs):
        raise AssertionError('J4 must not build the parent BaseBEVDepth model')

    class DummyLabelDistill(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            calls.append((args, kwargs))

    monkeypatch.setattr(base_exp, 'BaseBEVDepth', fail_base_model)
    bundle = load_and_resolve_config(
        'configs/experiments/j4_wl05_wh07.yaml',
        project_root='.',
        require_checkpoint=False,
    )
    experiment = build_experiment(bundle, model_cls=DummyLabelDistill)

    assert isinstance(experiment.model, DummyLabelDistill)
    assert len(calls) == 1
    assert calls[0][1]['structured_output'] is True
    assert experiment.backbone_conf['output_channels'] == 150
    assert experiment.resolved_config['student']['output_channels'] == 150
