"""Retain selected epoch checkpoints without adding checkpoint writes."""

from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


class MilestoneCheckpoint(ModelCheckpoint):
    """Keep native top-k checkpoints plus milestones expressed as one-based epochs.

    Milestones must coincide with the native save schedule: this callback never
    triggers an extra save. The trainer uses epoch-end saves, monitor="epoch",
    mode="max", and a distinct epoch filename to retain the most recent k files.
    Serialization and last links remain owned by Lightning's checkpoint strategy.
    """

    def __init__(self, milestones: list[int], **kwargs: Any) -> None:
        if any(type(epoch) is not int or epoch <= 0 for epoch in milestones):
            raise ValueError("milestones must contain positive integers (one-based epochs)")
        super().__init__(**kwargs)
        self.milestones = frozenset(milestones)
        self.milestone_paths: dict[str, int] = {}

    def _save_checkpoint(self, trainer: pl.Trainer, filepath: str) -> None:
        epoch = trainer.current_epoch + 1
        previous_epoch = self.milestone_paths.get(filepath)
        if previous_epoch is not None and previous_epoch != epoch:
            raise ValueError(f"Cannot overwrite retained milestone checkpoint: {filepath}")
        retain = epoch in self.milestones and filepath != self.last_model_path
        if retain:
            # Include the current file in its own saved callback state so that
            # resuming directly from a milestone retains it when top-k evicts it.
            self.milestone_paths[filepath] = epoch
        try:
            super()._save_checkpoint(trainer, filepath)
        except BaseException:
            if retain and previous_epoch is None:
                self.milestone_paths.pop(filepath, None)
            raise

    def _remove_checkpoint(self, trainer: pl.Trainer, filepath: str) -> None:
        if filepath not in self.milestone_paths:
            super()._remove_checkpoint(trainer, filepath)

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["milestone_paths"] = dict(self.milestone_paths)
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        # Match native top-k restoration: paths from a different output directory
        # must not become retention decisions in this run's directory.
        self.milestone_paths = (
            dict(state_dict.get("milestone_paths", {}))
            if state_dict.get("dirpath", self.dirpath) == self.dirpath
            else {}
        )
