python labeldistill/exps/nuscenes/labeldistill/Lidar_gt_label_r50_128x128_e_24.py \
    --gpus 2 \
    --evaluate \
    --ckpt_path outputs/Lidar_gt_label_r50_128x128_e_24/checkpoints/epoch_epoch=23.ckpt

AP: 0.3976                                                                                                                                          
mATE: 0.6202
mASE: 0.2608
mAOE: 0.3943
mAVE: 0.3670
mAAE: 0.2235
NDS: 0.5122
Eval time: 48.6s

Per-class results:
Object Class            AP      ATE     ASE     AOE     AVE     AAE   
car                     0.599   0.422   0.155   0.089   0.375   0.205 
truck                   0.357   0.581   0.188   0.079   0.325   0.199 
bus                     0.401   0.721   0.193   0.050   0.801   0.300 
trailer                 0.190   0.885   0.227   0.345   0.234   0.197 
construction_vehicle    0.103   1.110   0.484   1.038   0.130   0.416 
pedestrian              0.393   0.655   0.280   0.719   0.396   0.216 
motorcycle              0.393   0.576   0.254   0.519   0.512   0.249 
bicycle                 0.386   0.447   0.250   0.589   0.162   0.005 
traffic_cone            0.575   0.411   0.319   nan     nan     nan   
barrier                 0.579   0.394   0.257   0.120   nan     nan   


python labeldistill/exps/nuscenes/labeldistill/Lidar_gt_label_r50_128x128_e24_full.py \
    --gpus 2 \
    --evaluate \
    --ckpt_path outputs/Lidar_gt_label_r50_128x128_e24_full/checkpoints/epoch_epoch=23.ckpt

AP: 0.4015                                                                                                                                          
mATE: 0.6206
mASE: 0.2584
mAOE: 0.4466
mAVE: 0.3579
mAAE: 0.2177
NDS: 0.5107
Eval time: 50.3s

Per-class results:
Object Class            AP      ATE     ASE     AOE     AVE     AAE   
car                     0.602   0.426   0.156   0.094   0.374   0.208 
truck                   0.354   0.577   0.187   0.085   0.335   0.202 
bus                     0.406   0.681   0.188   0.056   0.714   0.277 
trailer                 0.204   0.979   0.229   0.555   0.259   0.194 
construction_vehicle    0.120   1.014   0.454   1.217   0.139   0.397 
pedestrian              0.397   0.651   0.284   0.740   0.393   0.221 
motorcycle              0.402   0.563   0.250   0.535   0.489   0.239 
bicycle                 0.394   0.486   0.259   0.618   0.160   0.004 
traffic_cone            0.559   0.431   0.318   nan     nan     nan   
barrier                 0.578   0.397   0.259   0.120   nan     nan   
Testing DataLoader 0: 100%|███████████████████████████████████████████████████████████████████████████████████████| 377/377 [15:46<00:00,  0.40it/s]


python labeldistill/exps/nuscenes/labeldistill/LidarDistill_r50_128x128_e24_roi.py \
    --gpus 2 \
    --evaluate \
    --ckpt_path outputs/LidarDistill_r50_128x128_e24_roi/checkpoints/epoch_epoch=23.ckpt

Testing DataLoader 0: 100%|███████████████████████████████████████████████████████████████████████████████████████| 377/377 [12:01<00:00,  0.52it/s]
Formating bboxes of img_bbox
Start to convert detection format...
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 6019/6019, 208.4 task/s, elapsed: 29s, ETA:     0s
Results writes to ./outputs/LidarDistill_r50_128x128_e24_roi/results_nusc.json
Evaluating bboxes of img_bbox
mAP : 0.4034                                                                                                                                                 mAP: 0.4035                                                                                                                                          
mATE: 0.5944
mASE: 0.2605
mAOE: 0.3870
mAVE: 0.3697
mAAE: 0.2228
NDS: 0.5183
Eval time: 47.4s

Per-class results:
Object Class            AP      ATE     ASE     AOE     AVE     AAE   
car                     0.605   0.419   0.154   0.089   0.377   0.204 
truck                   0.360   0.570   0.189   0.077   0.318   0.205 
bus                     0.394   0.621   0.187   0.043   0.832   0.288 
trailer                 0.190   0.996   0.229   0.353   0.239   0.218 
construction_vehicle    0.116   0.882   0.486   1.040   0.124   0.428 
pedestrian              0.397   0.654   0.280   0.734   0.403   0.226 
motorcycle              0.408   0.548   0.254   0.470   0.493   0.208 
bicycle                 0.406   0.440   0.249   0.554   0.172   0.006 
traffic_cone            0.572   0.424   0.318   nan     nan     nan   
barrier                 0.588   0.389   0.260   0.121   nan     nan   
Testing DataLoader 0: 100%|███████████████████████████████████████████████████████████████████████████████████████| 377/377 [15:22<00:00,  0.41it/s]
(base) root@labeldistill-0:/mnt/drtraining/user/qiupenggu/ROI_LABEL_DISTILL# 


python labeldistill/exps/nuscenes/labeldistill/LidarDistill_r50_128x128_e24_roi_noscale.py \
    --gpus 2 \
    --evaluate \
    --ckpt_path outputs/LidarDistill_r50_128x128_e24_roi_noscale/checkpoints/epoch_epoch=22.ckpt


/mnt/drtraining/user/qiupenggu/ROI_LABEL_DISTILL/labeldistill/exps/nuscenes/labeldistill/LidarDistill_r50_128x128_e24_roi_noscale.py
outputs/LidarDistill_r50_128x128_e24_roi_noscale/checkpoints/epoch_epoch=23.ckpt
labeldistill/LidarDistill_r50_128x128_e24_roi_noscale.py



# ConvNeXt-B ROI蒸馏（无FP）训练
python labeldistill/exps/nuscenes/labeldistill/LidarDistill_convnextb_900x1600_e24_roi.py --gpus 8 --batch_size_per_device 2



/mnt/public-data/user/qiupenggu


cp /mnt/drtraining/user/qiupenggu/ROI_LABEL_DISTILL/main.tex /mnt/public-data/user/qiupenggu





python labeldistill/exps/nuscenes/labeldistill/LidarDistill_r50_128x128_e24_roi_v2.py --gpus 8 -b 8


outputs/LidarDistill_r50_128x128_e24_roi_v2/checkpoints/epoch_epoch=23.ckpt

labeldistill/exps/nuscenes/labeldistill/LidarDistill_r50_128x128_e24_roi_v2.py



python labeldistill/exps/nuscenes/find_best_way/LidarDistill_r50_128x128_e24_roi_v2_exp8.py \
    --gpus 2 \
    --evaluate \
    --ckpt_path outputs/LidarDistill_r50_128x128_e24_roi_v2_exp8/checkpoints/epoch_epoch=23.ckpt




outputs/LidarDistill_r50_128x128_e24_roi/checkpoints/epoch_epoch=23.ckpt




# Exp1: 类别相关 tau
python labeldistill/exps/nuscenes/find_best_way/LidarDistill_r50_128x128_e24_roi_v2_exp1.py --gpus 8 -b 8

# Exp2: 未匹配基础扩展
python labeldistill/exps/nuscenes/find_best_way/LidarDistill_r50_128x128_e24_roi_v2_exp2.py --gpus 8 -b 8

# Exp3: mask权重 w_l=0.7, w_u=0.3  tjob
python labeldistill/exps/nuscenes/find_best_way/LidarDistill_r50_128x128_e24_roi_v2_exp3.py --gpus 8 -b 8

# Exp4: tau_ratio + base_expand 组合
python labeldistill/exps/nuscenes/find_best_way/LidarDistill_r50_128x128_e24_roi_v2_exp4.py --gpus 8 -b 8

# Exp5: 全部改动
python labeldistill/exps/nuscenes/find_best_way/LidarDistill_r50_128x128_e24_roi_v2_exp5.py --gpus 8 -b 8



LidarDistill_r101_128x128_e24_roi_v2_exp7.py



/mnt/drtraining/user/qiupenggu/ROI_LABEL_DISTILL/labeldistill/exps/nuscenes/labeldistill/big_backbone/LidarDistill_convnextb_900x1600_e24_roi_v2_exp7.py




labeldistill/exps/nuscenes/find_best_v3/LidarDistill_r50_128x128_e24_roi_v3_exp2.py


labeldistill/exps/nuscenes/find_best_v3/LidarDistill_r50_128x128_e24_roi_v3_exp2.py




outputs/LidarDistill_r50_128x128_e24_roi/checkpoints/epoch_epoch=23.ckpt



python labeldistill/exps/nuscenes/find_best_v3/LidarDistill_r50_128x128_e24_roi_v3.py \
    --gpus 2 \
    --evaluate \
    --ckpt_path outputs/LidarDistill_r50_128x128_e24_roi/checkpoints/epoch_epoch=23.ckpt







labeldistill/exps/nuscenes/ablation_module/ablation_A0_baseline_full_gt.py

labeldistill/exps/nuscenes/ablation_module/ablation_A1_channel_split.py

labeldistill/exps/nuscenes/ablation_module/ablation_A2_cs_roi.py

labeldistill/exps/nuscenes/ablation_module/ablation_A3_full_method.py

labeldistill/exps/nuscenes/ablation_module/ablation_A4_roi_only.py

labeldistill/exps/nuscenes/ablation_module/ablation_A5_roi_scale.py



outputs/ablation_A0_baseline_full_gt/checkpoints/epoch_epoch=23.ckpt

outputs/ablation_A1_channel_split/checkpoints/epoch_epoch=23.ckpt

outputs/ablation_A2_cs_roi/checkpoints/epoch_epoch=23.ckpt

outputs/ablation_A3_full_method/checkpoints/epoch_epoch=23.ckpt
outputs/ablation_A4_roi_only/checkpoints/epoch_epoch=23.ckpt
outputs/ablation_A5_roi_scale/checkpoints/epoch_epoch=23.ckpt


python labeldistill/exps/nuscenes/ablation_module/ablation_A5_roi_scale.py \
    --gpus 2 \
    --evaluate \
    --ckpt_path outputs/ablation_A5_roi_scale/checkpoints/epoch_epoch=23.ckpt







python labeldistill/exps/nuscenes/ablation_param/param_J1_wl02_wh05.py --gpus 8 -b 4
python labeldistill/exps/nuscenes/ablation_param/param_J2_wl03_wh06.py --gpus 8 -b 4
python labeldistill/exps/nuscenes/ablation_param/param_J3_wl035_wh07.py --gpus 8 -b 4
python labeldistill/exps/nuscenes/ablation_param/param_J4_wl05_wh08.py --gpus 8 -b 4
python labeldistill/exps/nuscenes/ablation_param/param_J5_wl03_wh08.py --gpus 8 -b 4





python labeldistill/exps/nuscenes/ablation_param/param_P3_lambda_020.py --gpus 8 -b 4
python labeldistill/exps/nuscenes/ablation_param/param_P3_lambda_040.py --gpus 8 -b 4
python labeldistill/exps/nuscenes/ablation_param/param_P3_lambda_050.py --gpus 8 -b 4
python labeldistill/exps/nuscenes/ablation_param/param_P3_lambda_070.py --gpus 8 -b 4
python labeldistill/exps/nuscenes/ablation_param/param_P3_lambda_080.py --gpus 8 -b 4
python labeldistill/exps/nuscenes/ablation_param/param_P3_lambda_100.py --gpus 8 -b 4






python labeldistill/exps/nuscenes/ablation_module/ablation_A0_baseline_full_gt.py --gpus 8 -b 4
python labeldistill/exps/nuscenes/ablation_module/ablation_A1_channel_split.py --gpus 8 -b 4
python labeldistill/exps/nuscenes/ablation_module/ablation_A2_cs_roi.py --gpus 8 -b 4
python labeldistill/exps/nuscenes/ablation_module/ablation_A3_full_method.py --gpus 8 -b 4
python labeldistill/exps/nuscenes/ablation_module/ablation_A4_roi_only.py --gpus 8 -b 4
python labeldistill/exps/nuscenes/ablation_module/ablation_A5_roi_scale.py --gpus 8 -b 4



labeldistill/exps/nuscenes/ablation_param/param_mu_010.py
labeldistill/exps/nuscenes/ablation_param/param_mu_020.py

/mnt/drtraining/user/qiupenggu/ROI_LABEL_DISTILL/labeldistill/exps/nuscenes/ablation_param/param_J1_wl02_wh05.py


/mnt/public-data/user/qiupenggu/temp


cp /mnt/drtraining/user/qiupenggu/ROI_LABEL_DISTILL/main.tex /mnt/public-data/user/qiupenggu/temp





cd /mnt/workspace/guqiupeng/code/ROI_LABEL_DISTILL
python labeldistill/exps/nuscenes/labeldistill/train_teacher_centerpoint.py \
  -e \
  --ckpt_path ./outputs/train_teacher_centerpoint/checkpoints/step_step=50200.ckpt \
  --gpus 4 -b 16



