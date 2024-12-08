# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models.task_modules.samplers import SamplingResult
from mmdet.models.utils import multi_apply
from mmdet.utils import InstanceList
from mmdet.registry import MODELS
from mmdet.structures.bbox import get_box_tensor
from .bbox_head import BBoxHead


@MODELS.register_module()
class ConvFCBBoxHead(BBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs: int = 0,
                 num_shared_fcs: int = 0,
                 num_cls_convs: int = 0,
                 num_cls_fcs: int = 0,
                 num_reg_convs: int = 0,
                 num_reg_fcs: int = 0,
                 conv_out_channels: int = 256,
                 fc_out_channels: int = 1024,
                 conv_cfg: Optional[Union[dict, ConfigDict]] = None,
                 norm_cfg: Optional[Union[dict, ConfigDict]] = None,
                 init_cfg: Optional[Union[dict, ConfigDict]] = None,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, init_cfg=init_cfg, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = self.num_classes + 1
            cls_predictor_cfg_ = self.cls_predictor_cfg.copy()
            cls_predictor_cfg_.update(
                in_features=self.cls_last_dim, out_features=cls_channels)
            self.fc_cls = MODELS.build(cls_predictor_cfg_)
        if self.with_reg:
            box_dim = self.bbox_coder.encode_size
            out_dim_reg = box_dim if self.reg_class_agnostic else \
                box_dim * self.num_classes
            reg_predictor_cfg_ = self.reg_predictor_cfg.copy()
            if isinstance(reg_predictor_cfg_, (dict, ConfigDict)):
                reg_predictor_cfg_.update(
                    in_features=self.reg_last_dim, out_features=out_dim_reg)
            self.fc_reg = MODELS.build(reg_predictor_cfg_)

        if init_cfg is None:
            # when init_cfg is None,
            # It has been set to
            # [[dict(type='Normal', std=0.01, override=dict(name='fc_cls'))],
            #  [dict(type='Normal', std=0.001, override=dict(name='fc_reg'))]
            # after `super(ConvFCBBoxHead, self).__init__()`
            # we only need to append additional configuration
            # for `shared_fcs`, `cls_fcs` and `reg_fcs`
            self.init_cfg += [
                dict(
                    type='Xavier',
                    distribution='uniform',
                    override=[
                        dict(name='shared_fcs'),
                        dict(name='cls_fcs'),
                        dict(name='reg_fcs')
                    ])
            ]

    def _add_conv_fc_branch(self,
                            num_branch_convs: int,
                            num_branch_fcs: int,
                            in_channels: int,
                            is_shared: bool = False) -> tuple:
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def forward(self, x: Tuple[Tensor]) -> tuple:
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_score (Tensor): Classification scores for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * num_classes.
                - bbox_pred (Tensor): Box energies / deltas for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * 4.
        """
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred


@MODELS.register_module()
class Shared2FCBBoxHead(ConvFCBBoxHead):

    def __init__(self, fc_out_channels: int = 1024, *args, **kwargs) -> None:
        super().__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)


@MODELS.register_module()
class Shared4Conv1FCBBoxHead(ConvFCBBoxHead):

    def __init__(self, fc_out_channels: int = 1024, *args, **kwargs) -> None:
        super().__init__(
            num_shared_convs=4,
            num_shared_fcs=1,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)


@MODELS.register_module()
class Shared2FCBBoxHeadWithEmission(Shared2FCBBoxHead):
    def __init__(self,
                 emission_loss: Optional[Union[dict, ConfigDict]] = None,
                 normalize_emission: bool = False,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_emission = MODELS.build(emission_loss)
        self.normalize_emission = normalize_emission

        # 添加 emission 回归分支
        self.emission_fc = nn.Sequential(
            nn.Linear(self.fc_out_channels, self.fc_out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.fc_out_channels, 1)  # 预测标量 emission
        )
    
    def forward(self, x: Tuple[Tensor]) -> tuple:
        # 共享特征提取
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)
        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.flatten(1)
            for fc in self.shared_fcs:
                x = self.relu(fc(x))

        # 分类和回归分支
        x_cls, x_reg, x_emission = x, x, x  # 共享特征
        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))
        
        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        # 排放量回归分支
        # breakpoint()
        if x_emission.dim() > 2:
            if self.with_avg_pool:
                x_emission = self.avg_pool(x_emission)
            x_emission = x_emission.flatten(1)
        emission_pred = self.emission_fc(x_emission).squeeze(-1)

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred, emission_pred


    def _get_targets_single(self, pos_priors: Tensor, neg_priors: Tensor,
                pos_gt_bboxes: Tensor, pos_gt_labels: Tensor,
                pos_gt_emissions: Optional[Tensor],
                cfg: ConfigDict) -> tuple:
        """计算每张图片的分类、回归和排放目标。

        Args:
            pos_priors (Tensor): 正样本候选框，形状为 (num_pos, 4)。
            neg_priors (Tensor): 负样本候选框，形状为 (num_neg, 4)。
            pos_gt_bboxes (Tensor): 正样本的真实边界框，形状为 (num_pos, 4)。
            pos_gt_labels (Tensor): 正样本的真实类别标签，形状为 (num_pos,)。
            pos_gt_emissions (Tensor, optional): 正样本的排放量，形状为 (num_pos,)。
            cfg (ConfigDict): R-CNN 的训练配置。

        Returns:
            tuple: 分类、回归和排放的目标值，包括：
                - labels: 分类标签。
                - label_weights: 分类权重。
                - bbox_targets: 边界框回归目标。
                - bbox_weights: 回归权重。
                - emission_targets: 排放量目标。
        """
        num_pos = pos_priors.size(0)
        num_neg = neg_priors.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_priors.new_full((num_samples, ),
                                     self.num_classes,
                                     dtype=torch.long)
        reg_dim = pos_gt_bboxes.size(-1) if self.reg_decoded_bbox \
            else self.bbox_coder.encode_size
        label_weights = pos_priors.new_zeros(num_samples)
        emission_weights = pos_priors.new_zeros(num_samples)
        bbox_targets = pos_priors.new_zeros(num_samples, reg_dim)
        bbox_weights = pos_priors.new_zeros(num_samples, reg_dim)
        # 排放目标
        emission_targets = pos_priors.new_zeros(num_samples)
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    pos_priors, pos_gt_bboxes)
            else:
                # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
                # is applied directly on the decoded bounding boxes, both
                # the predicted boxes and regression targets should be with
                # absolute coordinate format.
                pos_bbox_targets = get_box_tensor(pos_gt_bboxes)
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1

            emission_targets[:num_pos] = pos_gt_emissions
            emission_weights[:num_pos] = pos_weight
            
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        num_pos = pos_priors.size(0)
        num_neg = neg_priors.size(0)
        num_samples = num_pos + num_neg

        return labels, label_weights, bbox_targets, bbox_weights, emission_targets, emission_weights



    def get_targets_with_emissions(self,
            sampling_results: List[SamplingResult],
            rcnn_train_cfg: ConfigDict,
            concat: bool = True) -> tuple:
        """扩展目标生成，包含 emission_targets。"""
        pos_priors_list = [res.pos_priors for res in sampling_results]
        neg_priors_list = [res.neg_priors for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        pos_gt_emissions_list = [res.pos_gt_emissions for res in sampling_results]
        
        # breakpoint()

        # 调用原始目标生成逻辑
        labels, label_weights, bbox_targets, bbox_weights, emission_targets, emission_weights = multi_apply(
            self._get_targets_single,
            pos_priors_list,
            neg_priors_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            pos_gt_emissions_list,
            cfg=rcnn_train_cfg)

        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
            emission_targets = torch.cat(emission_targets, 0)
            emission_weights = torch.cat(emission_weights, 0)

        # 归一化（如果需要）
        if self.normalize_emission and concat:
            emission_targets = (emission_targets - emission_targets.mean()) / (
                emission_targets.std() + 1e-6)

        return labels, label_weights, bbox_targets, bbox_weights, emission_targets, emission_weights

    
    def loss_and_target(self,
        cls_score: Tensor,
        bbox_pred: Tensor,
        emission_pred: Tensor,
        rois: Tensor,
        sampling_results: List[SamplingResult],
        rcnn_train_cfg: ConfigDict,
        concat: bool = True,
        reduction_override: Optional[str] = None) -> dict:
        """扩展损失计算，加入 emission 分支。"""
        cls_reg_targets = self.get_targets_with_emissions(
            sampling_results, rcnn_train_cfg, concat=concat)

        labels, label_weights, bbox_targets, bbox_weights, emission_targets, emission_weights  = cls_reg_targets

        # 分类和回归损失
        losses = self.loss(
            cls_score,
            bbox_pred,
            rois,
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            reduction_override=reduction_override)

        # Emission 损失
        # breakpoint()
        if emission_pred is not None:
            loss_emission = self.loss_emission(emission_pred,emission_targets, emission_weights, reduction_override=reduction_override)
            # self.loss_emission(emission_pred,emission_targets,reduction_override=reduction_override)
            losses['loss_emission'] = loss_emission

        return dict(loss_bbox=losses, bbox_targets=cls_reg_targets)


    def predict_by_feat(self,
            rois: Tuple[Tensor],
            cls_scores: Tuple[Tensor],
            bbox_preds: Tuple[Tensor],
            emission_preds: Tuple[Tensor],
            batch_img_metas: List[dict],
            rcnn_test_cfg: Optional[ConfigDict] = None,
            rescale: bool = False) -> InstanceList:
        """扩展推理，包含 emission 的预测结果。"""
        assert len(cls_scores) == len(bbox_preds) == len(emission_preds)
        result_list = []
        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            results = self._predict_by_feat_single(
                roi=rois[img_id],
                cls_score=cls_scores[img_id],
                bbox_pred=bbox_preds[img_id],
                emission_pred=emission_preds[img_id],
                img_meta=img_meta,
                rescale=rescale,
                rcnn_test_cfg=rcnn_test_cfg)
            result_list.append(results)
        return result_list
    
    def _predict_by_feat_single(self,
            roi: Tensor,
            cls_score: Tensor,
            bbox_pred: Tensor,
            emission_pred: Tensor,
            img_meta: dict,
            rescale: bool = False,
            rcnn_test_cfg: Optional[ConfigDict] = None) -> InstanceData:
        """将单张图片的特征转换为检测结果，包括 bbox 和 emission。

        Args:
            roi (Tensor): RoI 的信息，形状为 (num_boxes, 5)。
            cls_score (Tensor): 分类分数，形状为 (num_boxes, num_classes+1)。
            bbox_pred (Tensor): 边界框预测结果，形状为 (num_boxes, num_classes*4)。
            emission_pred (Tensor): 排放量预测结果，形状为 (num_boxes,)。
            img_meta (dict): 图片元信息。
            rescale (bool): 是否对检测结果进行缩放。
            rcnn_test_cfg (ConfigDict): R-CNN 测试配置。

        Returns:
            InstanceData: 检测结果，包括 bbox、scores、labels 和 emission。
        """
        results = super()._predict_by_feat_single(
            roi, cls_score, bbox_pred, img_meta, rescale, rcnn_test_cfg)
        # NMS 和结果筛选
        if rcnn_test_cfg is None:
            results.emissions = emission_pred  # 直接输出排放预测
        else:
            # 关联 emission_pred 到最终检测结果
            results.emissions = emission_pred[results.labels]

        return results

