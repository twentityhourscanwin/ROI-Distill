import argparse
import os
import pickle
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from tqdm import tqdm


def generate_info(nusc, scenes, max_cam_sweeps=6, max_lidar_sweeps=9):
    infos = list()
    for cur_scene in tqdm(nusc.scene):
        if cur_scene['name'] not in scenes:
            continue
        first_sample_token = cur_scene['first_sample_token']
        cur_sample = nusc.get('sample', first_sample_token)
        while True:
            info = dict()
            sweep_cam_info = dict()
            cam_datas = list()
            lidar_datas = list()
            info['sample_token'] = cur_sample['token']
            info['timestamp'] = cur_sample['timestamp']
            info['scene_token'] = cur_sample['scene_token']
            cam_names = [
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'
            ]
            lidar_names = ['LIDAR_TOP']
            cam_infos = dict()
            lidar_infos = dict()
            for cam_name in cam_names:
                cam_data = nusc.get('sample_data',
                                    cur_sample['data'][cam_name])
                cam_datas.append(cam_data)
                sweep_cam_info = dict()
                sweep_cam_info['sample_token'] = cam_data['sample_token']
                sweep_cam_info['ego_pose'] = nusc.get(
                    'ego_pose', cam_data['ego_pose_token'])
                sweep_cam_info['timestamp'] = cam_data['timestamp']
                sweep_cam_info['is_key_frame'] = cam_data['is_key_frame']
                sweep_cam_info['height'] = cam_data['height']
                sweep_cam_info['width'] = cam_data['width']
                sweep_cam_info['filename'] = cam_data['filename']
                sweep_cam_info['calibrated_sensor'] = nusc.get(
                    'calibrated_sensor', cam_data['calibrated_sensor_token'])
                cam_infos[cam_name] = sweep_cam_info
            for lidar_name in lidar_names:
                lidar_data = nusc.get('sample_data',
                                      cur_sample['data'][lidar_name])
                lidar_datas.append(lidar_data)
                sweep_lidar_info = dict()
                sweep_lidar_info['sample_token'] = lidar_data['sample_token']
                sweep_lidar_info['ego_pose'] = nusc.get(
                    'ego_pose', lidar_data['ego_pose_token'])
                sweep_lidar_info['timestamp'] = lidar_data['timestamp']
                sweep_lidar_info['filename'] = lidar_data['filename']
                sweep_lidar_info['calibrated_sensor'] = nusc.get(
                    'calibrated_sensor', lidar_data['calibrated_sensor_token'])
                lidar_infos[lidar_name] = sweep_lidar_info

            # CenterPoint uses the key sweep plus nine causal (past) sweeps.
            # Keep only previous LiDAR sample_data records here; future sweeps
            # leak information that is unavailable to the student at inference.
            lidar_sweeps = [dict() for _ in range(max_lidar_sweeps)]
            cam_sweeps = [dict() for _ in range(max_cam_sweeps)]
            info['cam_infos'] = cam_infos
            info['lidar_infos'] = lidar_infos
            # for i in range(max_cam_sweeps):
            #     cam_sweeps.append(dict())
            for k, cam_data in enumerate(cam_datas):
                sweep_cam_data = cam_data
                for j in range(max_cam_sweeps):
                    if sweep_cam_data['prev'] == '':
                        break
                    else:
                        sweep_cam_data = nusc.get('sample_data',
                                                  sweep_cam_data['prev'])
                        sweep_cam_info = dict()
                        sweep_cam_info['sample_token'] = sweep_cam_data[
                            'sample_token']
                        if sweep_cam_info['sample_token'] != cam_data[
                                'sample_token']:
                            break
                        sweep_cam_info['ego_pose'] = nusc.get(
                            'ego_pose', cam_data['ego_pose_token'])
                        sweep_cam_info['timestamp'] = sweep_cam_data[
                            'timestamp']
                        sweep_cam_info['is_key_frame'] = sweep_cam_data[
                            'is_key_frame']
                        sweep_cam_info['height'] = sweep_cam_data['height']
                        sweep_cam_info['width'] = sweep_cam_data['width']
                        sweep_cam_info['filename'] = sweep_cam_data['filename']
                        sweep_cam_info['calibrated_sensor'] = nusc.get(
                            'calibrated_sensor',
                            cam_data['calibrated_sensor_token'])
                        cam_sweeps[j][cam_names[k]] = sweep_cam_info

            for k, lidar_data in enumerate(lidar_datas):
                sweep_lidar_data = lidar_data
                for j in range(max_lidar_sweeps):
                    if sweep_lidar_data['prev'] == '':
                        break
                    else:
                        sweep_lidar_data = nusc.get('sample_data',
                                                    sweep_lidar_data['prev'])
                        sweep_lidar_info = dict()
                        sweep_lidar_info['sample_token'] = sweep_lidar_data[
                            'sample_token']
                        sweep_lidar_info['ego_pose'] = nusc.get(
                            'ego_pose', sweep_lidar_data['ego_pose_token'])
                        sweep_lidar_info['timestamp'] = sweep_lidar_data[
                            'timestamp']
                        sweep_lidar_info['is_key_frame'] = sweep_lidar_data[
                            'is_key_frame']
                        sweep_lidar_info['filename'] = sweep_lidar_data[
                            'filename']
                        sweep_lidar_info['calibrated_sensor'] = nusc.get(
                            'calibrated_sensor',
                            sweep_lidar_data['calibrated_sensor_token'])
                        lidar_sweeps[j][lidar_names[k]] = sweep_lidar_info

            # Remove empty sweeps.
            for i, sweep in enumerate(cam_sweeps):
                if len(sweep.keys()) == 0:
                    cam_sweeps = cam_sweeps[:i]
                    break
            for i, sweep in enumerate(lidar_sweeps):
                if len(sweep.keys()) == 0:
                    lidar_sweeps = lidar_sweeps[:i]
                    break
            info['cam_sweeps'] = cam_sweeps
            info['lidar_sweeps'] = lidar_sweeps
            ann_infos = list()
            if 'anns' in cur_sample:
                for ann in cur_sample['anns']:
                    ann_info = nusc.get('sample_annotation', ann)
                    velocity = nusc.box_velocity(ann_info['token'])
                    if np.any(np.isnan(velocity)):
                        velocity = np.zeros(3)
                    ann_info['velocity'] = velocity
                    ann_infos.append(ann_info)
                info['ann_infos'] = ann_infos
            infos.append(info)
            if cur_sample['next'] == '':
                break
            else:
                cur_sample = nusc.get('sample', cur_sample['next'])
    return infos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='./data/nuScenes')
    parser.add_argument('--max-lidar-sweeps', type=int, default=9)
    parser.add_argument('--output-suffix', default='10sweeps_past9')
    args = parser.parse_args()
    dataroot = args.data_root

    trainval_nusc = NuScenes(version='v1.0-trainval',
                             dataroot=dataroot,
                             verbose=True)

    train_infos = generate_info(
        trainval_nusc, splits.train,
        max_lidar_sweeps=args.max_lidar_sweeps)
    train_path = os.path.join(
        dataroot, f'nuscenes_infos_train_{args.output_suffix}.pkl')
    with open(train_path, 'wb') as f:
        pickle.dump(train_infos, f)

    val_infos = generate_info(
        trainval_nusc, splits.val,
        max_lidar_sweeps=args.max_lidar_sweeps)
    val_path = os.path.join(
        dataroot, f'nuscenes_infos_val_{args.output_suffix}.pkl')
    with open(val_path, 'wb') as f:
        pickle.dump(val_infos, f)

    print(f'Wrote {train_path}')
    print(f'Wrote {val_path}')


if __name__ == '__main__':
    main()
