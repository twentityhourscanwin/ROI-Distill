from copy import deepcopy

import torch

from labeldistill.exps.nuscenes import base_exp
from labeldistill.exps.nuscenes.ablation_param import param_J4_wl05_wh08
from labeldistill.models.distill_outputs import (
    LabelDistillOutput,
    ProposalBatch,
    StudentOutput,
    TeacherOutput,
)


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


def test_j4_builds_only_the_final_structured_model(monkeypatch, tmp_path):
    calls = []

    def fail_base_model(*args, **kwargs):
        raise AssertionError('J4 must not build the parent BaseBEVDepth model')

    class DummyLabelDistill(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            calls.append((args, kwargs))

    monkeypatch.setattr(base_exp, 'BaseBEVDepth', fail_base_model)
    monkeypatch.setattr(param_J4_wl05_wh08, 'LabelDistill', DummyLabelDistill)
    experiment = param_J4_wl05_wh08.LabelDistillModel(
        default_root_dir=str(tmp_path))

    assert isinstance(experiment.model, DummyLabelDistill)
    assert len(calls) == 1
    assert calls[0][1]['structured_output'] is True
    assert experiment.backbone_conf['output_channels'] == 150
    assert experiment.hparams['backbone_conf']['output_channels'] == 150
