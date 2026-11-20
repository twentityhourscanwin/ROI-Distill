"""训练蒸馏模型，成功后逐轮验证配置中连续保留的普通权重。"""

import student_pipeline as base


class DistillPipeline(base.StudentPipeline):
    """复用训练、不可变产物核验及可续评的完整验证流程。"""

    def _validate_config(self, config):
        epochs = config['runtime']['max_epochs']
        if epochs not in (30, 36):
            raise ValueError('蒸馏预算必须为30或36轮')
        self._validate_runtime(config, epochs)
        if config['student']['training_mode'] != 'distillation':
            raise ValueError('该入口要求distillation模式')
        if config['loss']['response_weight'] != 1.0:
            raise ValueError('本轮响应蒸馏系数固定为1.0')
        checkpoint = config['checkpoint']
        milestones = checkpoint['milestone_epochs']
        allowed_windows = (list(range(24, epochs + 1)),
                           list(range(28, epochs + 1)))
        if (checkpoint['every_n_epochs'] != 1
                or checkpoint['save_top_k'] == 0
                or milestones not in allowed_windows):
            raise ValueError('必须逐轮保存并保留第24或28轮到末轮的普通权重')
        self.epochs = tuple(epoch - 1 for epoch in milestones)


class TrainvalDistillPipeline(DistillPipeline):
    """保留原训练预算和保存窗口，仅核验预先指定的第 30 轮普通权重。"""

    default_config = 'convnext_wide_depth_trainval_s0.yaml'
    training_split = 'trainval'
    selected_epoch = 29

    def _validate_config(self, config):
        super()._validate_config(config)
        if config['runtime']['max_epochs'] != 36 or self.selected_epoch not in self.epochs:
            raise ValueError('Trainval requires the 36-epoch schedule and an epoch 30 checkpoint')
        self.epochs = (self.selected_epoch,)


if __name__ == '__main__':
    raise SystemExit(base.main(
        pipeline_class=DistillPipeline, config_name='convnext_distill.yaml', description=__doc__,
        train_only_class=TrainvalDistillPipeline,
    ))
