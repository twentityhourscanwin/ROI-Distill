"""训练蒸馏模型，成功后逐轮验证第24轮起保留的普通权重。"""

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
        if (checkpoint['every_n_epochs'] != 1
                or checkpoint['save_top_k'] == 0
                or checkpoint['milestone_epochs'] != list(range(24, epochs + 1))):
            raise ValueError('必须逐轮保存并保留第24轮到末轮的普通权重')
        self.epochs = tuple(range(23, epochs))


if __name__ == '__main__':
    raise SystemExit(base.main(
        pipeline_class=DistillPipeline, config_name='convnext_distill.yaml', description=__doc__,
    ))
