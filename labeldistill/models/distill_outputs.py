from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import torch


@dataclass(frozen=True)
class ProposalBatch:
    """Ragged decoded proposals, represented as one tensor per sample."""

    boxes: List[torch.Tensor]
    scores: List[torch.Tensor]
    labels: List[torch.Tensor]

    def __post_init__(self):
        batch_size = len(self.boxes)
        if len(self.scores) != batch_size or len(self.labels) != batch_size:
            raise ValueError('ProposalBatch fields must have equal batch size')
        for idx, (boxes, scores, labels) in enumerate(
                zip(self.boxes, self.scores, self.labels)):
            if len(boxes) != len(scores) or len(boxes) != len(labels):
                raise ValueError(
                    f'ProposalBatch sample {idx} has inconsistent lengths')

    def __len__(self):
        return len(self.boxes)

    def __getitem__(self, index):
        """Compatibility view matching the previous [boxes, scores, labels]."""
        return [self.boxes[index], self.scores[index], self.labels[index]]


@dataclass(frozen=True)
class StudentOutput:
    """Student-side outputs needed by training and optional diagnostics."""

    raw_preds: Any
    depth: Optional[torch.Tensor]
    distill_features: Sequence[torch.Tensor]
    neck_features: Any = None
    decoded_boxes: Optional[ProposalBatch] = None


@dataclass(frozen=True)
class TeacherOutput:
    """Frozen teacher predictions, features and decoded proposals."""

    raw_preds: Any
    backbone_features: Sequence[torch.Tensor]
    neck_features: Sequence[torch.Tensor]
    proposals: ProposalBatch


@dataclass(frozen=True)
class LabelDistillOutput:
    """Structured training output of the LiDAR distillation model."""

    student: StudentOutput
    teacher: TeacherOutput
