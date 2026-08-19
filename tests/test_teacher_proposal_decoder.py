from copy import deepcopy
from types import SimpleNamespace

import torch
from mmdet3d.registry import TASK_UTILS

from labeldistill.layers.heads.kd_head import KDHead
from labeldistill.models.distill_outputs import ProposalBatch
from labeldistill.models.teacher_proposal_decoder import TeacherProposalDecoder


def _prediction(num_classes):
    shape = (1, num_classes, 2, 2)
    return [{
        'heatmap': torch.full(shape, -10.0),
        'reg': torch.zeros((1, 2, 2, 2)),
        'height': torch.zeros((1, 1, 2, 2)),
        'dim': torch.zeros((1, 3, 2, 2)),
        'rot': torch.cat([
            torch.zeros((1, 1, 2, 2)),
            torch.ones((1, 1, 2, 2)),
        ], dim=1),
        'vel': torch.zeros((1, 2, 2, 2)),
    }]


def test_explicit_threshold_nms_and_task_label_offsets():
    bbox_coder = dict(
        type='CenterPointBBoxCoder',
        post_center_range=[-10, -10, -10, 10, 10, 10],
        max_num=4,
        score_threshold=0.1,
        out_size_factor=1,
        voxel_size=[1.0, 1.0],
        pc_range=[0, 0, -5, 2, 2, 5],
        code_size=9,
    )
    decoder = TeacherProposalDecoder(
        bbox_coder=bbox_coder,
        num_classes=[1, 2],
        nms_type='circle',
        min_radius=[2.0, 0.1],
        post_max_size=4,
        norm_bbox=True,
    )

    task0 = _prediction(1)
    task1 = _prediction(2)
    task0[0]['heatmap'][0, 0, 0, 0] = torch.logit(torch.tensor(0.9))
    task0[0]['heatmap'][0, 0, 0, 1] = torch.logit(torch.tensor(0.8))
    task1[0]['heatmap'][0, 1, 1, 1] = torch.logit(torch.tensor(0.7))
    # This candidate is explicitly below the teacher's 0.1 threshold.
    task1[0]['heatmap'][0, 0, 0, 0] = torch.logit(torch.tensor(0.09))

    proposals = decoder([task0, task1])
    assert isinstance(proposals, ProposalBatch)
    boxes, scores, labels = proposals[0]

    assert boxes.shape == (2, 9)
    assert torch.allclose(scores, torch.tensor([0.9, 0.7]), atol=1e-6)
    assert labels.tolist() == [0, 2]


def test_matches_the_previous_kd_head_decode_behavior():
    bbox_coder = dict(
        type='CenterPointBBoxCoder',
        post_center_range=[-10, -10, -10, 10, 10, 10],
        max_num=4,
        score_threshold=0.1,
        out_size_factor=1,
        voxel_size=[1.0, 1.0],
        pc_range=[0, 0, -5, 2, 2, 5],
        code_size=9,
    )
    min_radius = [2.0, 0.1]
    decoder = TeacherProposalDecoder(
        bbox_coder=bbox_coder,
        num_classes=[1, 2],
        min_radius=min_radius,
        post_max_size=4,
    )
    previous_decoder_owner = SimpleNamespace(
        bbox_coder=TASK_UTILS.build(deepcopy(bbox_coder)),
        num_classes=[1, 2],
        norm_bbox=True,
        test_cfg={
            'nms_type': 'circle',
            'min_radius': min_radius,
            'post_max_size': 4,
        },
    )

    torch.manual_seed(7)
    predictions = [_prediction(1), _prediction(2)]
    for task in predictions:
        task[0]['heatmap'].normal_()
        task[0]['reg'].uniform_(-0.25, 0.25)
        task[0]['height'].uniform_(-1.0, 1.0)
        task[0]['dim'].uniform_(-0.5, 0.5)
        task[0]['rot'].normal_()
        task[0]['vel'].normal_()

    expected = KDHead.get_bboxes(
        previous_decoder_owner, predictions, img_metas=None)
    actual = decoder(predictions)

    assert len(actual) == len(expected)
    for actual_sample, expected_sample in zip(actual, expected):
        for actual_tensor, expected_tensor in zip(
                actual_sample, expected_sample):
            assert torch.allclose(actual_tensor, expected_tensor)
