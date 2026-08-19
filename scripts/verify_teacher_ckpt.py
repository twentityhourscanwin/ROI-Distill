"""校验「单独训练得到的教师 checkpoint」能否被蒸馏脚本直接当作教师权重加载。

完全复刻蒸馏脚本 (如 Lidar_gt_label_r50_128x128_e_24.py / LidarDistill_r50_128x128_e24_roi.py)
里加载教师的逻辑::

    prefix = 'model.centerpoint.'
    load_keys = [k for k in sd if k.startswith(prefix)]
    centerpoint.load_state_dict({k[len(prefix):]: sd[k] for k in load_keys})

用法::

    python scripts/verify_teacher_ckpt.py ./outputs/train_teacher_centerpoint/checkpoints/last.ckpt
"""
import sys

import torch

import mmdet.models.losses  # noqa: F401
import mmdet3d.models       # noqa: F401
from mmdet3d.registry import MODELS

# 复用训练脚本里的 CenterPoint 配置，确保结构完全一致
from labeldistill.exps.nuscenes.labeldistill.train_teacher_centerpoint import \
    lidar_conf


def main(ckpt_path):
    print(f'[verify] loading checkpoint: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

    prefix = 'model.centerpoint.'
    load_keys = [k for k in state_dict if k.startswith(prefix)]
    print(f'[verify] keys with prefix "{prefix}": {len(load_keys)}')
    if len(load_keys) == 0:
        print('[verify] FAILED: 没有找到 model.centerpoint.* 前缀的权重，'
              '说明保存的 key 结构不对。')
        sys.exit(1)

    new_state = {k[len(prefix):]: state_dict[k] for k in load_keys}

    # 构建一个全新的 CenterPoint（去掉 voxel_layer，和蒸馏脚本里一致）
    conf = {k: v for k, v in lidar_conf.items() if k != 'voxel_layer'}
    centerpoint = MODELS.build(conf)

    # 先用 strict=True 模拟蒸馏脚本的加载方式
    try:
        centerpoint.load_state_dict(new_state)
        print('[verify] SUCCESS: strict=True 加载成功，'
              '该 checkpoint 可被蒸馏脚本直接当作教师权重使用。')
    except RuntimeError as e:
        print('[verify] strict=True 加载失败，下面用 strict=False 给出差异明细：')
        ret = centerpoint.load_state_dict(new_state, strict=False)
        print(f'  missing_keys ({len(ret.missing_keys)}):')
        for k in ret.missing_keys[:50]:
            print('    -', k)
        print(f'  unexpected_keys ({len(ret.unexpected_keys)}):')
        for k in ret.unexpected_keys[:50]:
            print('    +', k)
        print('[verify] 原始错误信息：')
        print(str(e)[:1000])
        sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('usage: python scripts/verify_teacher_ckpt.py <ckpt_path>')
        sys.exit(1)
    main(sys.argv[1])
