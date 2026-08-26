# 训练命令

## B0

```bash
python tools/train.py \
  --config configs/experiments/b0_full_gt_uniform_no_scale.yaml \
  runtime.gpus=8 \
  runtime.batch_size_per_device=4 \
  runtime.precision=16
```

## B1

```bash
python tools/train.py \
  --config configs/experiments/b1_teacher_value_no_scale.yaml \
  runtime.gpus=8 \
  runtime.batch_size_per_device=4 \
  runtime.precision=16
```

## B1T

```bash
python tools/train.py \
  --config configs/experiments/b1t_teacher_value_elliptical_mask.yaml \
  runtime.gpus=8 \
  runtime.batch_size_per_device=4 \
  runtime.precision=16
```

## B2

```bash
python tools/train.py \
  --config configs/experiments/b2_teacher_value_adaptive_scale.yaml \
  runtime.gpus=8 \
  runtime.batch_size_per_device=4 \
  runtime.precision=16
```
