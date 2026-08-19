# labeldistill/refine_head/target_assigner/adaptive_weight_mask.py
# 可学习的质量感知蒸馏权重掩码生成器
# 替代 draw_heatmap_mask.py 中的 BEVDistillationMaskGenerator
# 核心改进：用 4 个可学习参数替代所有 hard-coded 权重
#   - matched GT:  k = clamp(alpha * score + beta, 0, 1)
#   - unmatched GT: k = clamp(k0 * (1 - d/d_max), 0, 1)
import torch
import torch.nn as nn
from mmdet3d.models.utils.gaussian import gaussian_radius


class AdaptiveWeightMaskGenerator(nn.Module):

    def __init__(self):
        super().__init__()

        grid_size = [1024, 1024, 40]
        voxel_size = [0.1, 0.1, 0.2]
        out_size_factor = 8
        point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

        self.register_buffer('pc_range', torch.tensor(point_cloud_range))
        self.register_buffer('voxel_size', torch.tensor(voxel_size))
        self.out_factor = out_size_factor
        self.gaussian_overlap = 0.1
        self.min_radius = 2
        self.feature_map_size = (
            grid_size[0] // out_size_factor,
            grid_size[1] // out_size_factor,
        )  # (W, H) = (128, 128)

        # ── 可学习参数 ──
        # matched GT: k = clamp(alpha * score + beta, 0, 1)
        # 初始化: alpha=1, beta=0 → 直接用 score 作为权重
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

        # unmatched GT: k = clamp(k0 * (1 - d / d_max), 0, 1)
        # 初始化: k0=0.4, d_max=50 → d=0→0.4, d=25→0.2, d=50→0
        self.k0 = nn.Parameter(torch.tensor(0.4))
        self.d_max = nn.Parameter(torch.tensor(50.0))

    # ──────────────────────── 权重计算 ────────────────────────

    def compute_matched_weight(self, scores):
        """matched GT 的自适应权重 (可微)"""
        return torch.clamp(self.alpha * scores + self.beta, 0.0, 1.0)

    def compute_unmatched_weight(self, gt_boxes):
        """unmatched GT 的距离衰减权重 (可微)"""
        distances = torch.sqrt(gt_boxes[:, 0] ** 2 + gt_boxes[:, 1] ** 2)
        d_max_safe = torch.clamp(self.d_max, min=1.0)
        return torch.clamp(self.k0 * (1.0 - distances / d_max_safe), 0.0, 1.0)

    # ──────────────────────── 高斯模板 ────────────────────────

    def _gaussian_params(self, box):
        """从框坐标计算特征图上的高斯参数，返回 (cx, cy, radius) 或 None"""
        x, y = box[0], box[1]
        width, length = box[3], box[4]

        cx = (x - self.pc_range[0]) / self.voxel_size[0] / self.out_factor
        cy = (y - self.pc_range[1]) / self.voxel_size[1] / self.out_factor

        w_fm = width / self.voxel_size[0] / self.out_factor
        l_fm = length / self.voxel_size[1] / self.out_factor

        if w_fm.item() <= 0 or l_fm.item() <= 0 or w_fm.item() > 1000 or l_fm.item() > 1000:
            return None

        r = gaussian_radius((l_fm, w_fm), min_overlap=self.gaussian_overlap)
        radius = max(self.min_radius, int(r.item()))
        return cx.item(), cy.item(), radius

    def _make_template(self, cx, cy, radius, H, W, device):
        """生成一个空间高斯模板 (不带梯度), 峰值=1, 形状 (H, W)"""
        x0 = max(0, int(cx) - radius)
        x1 = min(W, int(cx) + radius + 1)
        y0 = max(0, int(cy) - radius)
        y1 = min(H, int(cy) + radius + 1)
        if x0 >= x1 or y0 >= y1:
            return None

        lx = torch.arange(x0, x1, device=device, dtype=torch.float32) - cx
        ly = torch.arange(y0, y1, device=device, dtype=torch.float32) - cy
        gy, gx = torch.meshgrid(ly, lx, indexing='ij')
        sigma = (2 * radius + 1) / 6.0
        local = torch.exp(-(gx ** 2 + gy ** 2) / (2 * sigma ** 2))

        tpl = torch.zeros(H, W, device=device)
        tpl[y0:y1, x0:x1] = local
        return tpl

    # ──────────────────────── mask 生成 ────────────────────────

    def _collect_weighted_gaussians(self, boxes, weights, H, W, device):
        """对一组框生成加权高斯列表 (可微: 梯度通过 weight 流回)"""
        maps = []
        for i in range(len(boxes)):
            if weights[i].item() <= 0:
                continue
            params = self._gaussian_params(boxes[i])
            if params is None:
                continue
            tpl = self._make_template(*params, H, W, device)
            if tpl is not None:
                maps.append(weights[i] * tpl)
        return maps

    def generate_mask(self, matched_results, batch_size):
        feat_W, feat_H = self.feature_map_size

        high_rois = matched_results['refined_high_quality_rois']
        high_scores = matched_results['refined_high_quality_roi_scores']
        med_gts = matched_results['medium_quality_gt']
        med_scores = matched_results['medium_quality_roi_scores']
        unmatch_near = matched_results['unmatched_gt_near']
        unmatch_mid = matched_results['unmatched_gt_medium']
        unmatch_far = matched_results['unmatched_gt_far']

        mask_list = []

        for idx in range(batch_size):
            device = high_rois[idx].device
            all_maps = []

            # 1. 高质量匹配 → 绘制 ROI
            if len(high_rois[idx]) > 0:
                w = self.compute_matched_weight(high_scores[idx])
                all_maps.extend(
                    self._collect_weighted_gaussians(
                        high_rois[idx], w, feat_H, feat_W, device))

            # 2. 中等质量匹配 → 绘制 GT
            if len(med_gts[idx]) > 0:
                w = self.compute_matched_weight(med_scores[idx])
                all_maps.extend(
                    self._collect_weighted_gaussians(
                        med_gts[idx], w, feat_H, feat_W, device))

            # 3. 未匹配 GT → 距离衰减权重
            for unmatch in (unmatch_near[idx], unmatch_mid[idx], unmatch_far[idx]):
                if len(unmatch) > 0:
                    w = self.compute_unmatched_weight(unmatch)
                    all_maps.extend(
                        self._collect_weighted_gaussians(
                            unmatch, w, feat_H, feat_W, device))

            # 合并: 取逐像素最大值 (torch.max 对最大元素可微)
            if all_maps:
                stacked = torch.stack(all_maps)            # (N, H, W)
                mask = stacked.max(dim=0)[0].unsqueeze(0)  # (1, H, W)
            else:
                mask = torch.zeros(1, feat_H, feat_W, device=device)

            mask_list.append(mask)

        return torch.stack(mask_list)  # (B, 1, H, W)

    def forward(self, matched_results, batch_size):
        return self.generate_mask(matched_results, batch_size)

    def log_params(self):
        return {
            'aw/alpha': self.alpha.item(),
            'aw/beta': self.beta.item(),
            'aw/k0': self.k0.item(),
            'aw/d_max': self.d_max.item(),
        }
