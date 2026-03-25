# --------------------------------------------------------
# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Feng Li (fliay@connect.ust.hk)
# --------------------------------------------------------
from typing import Tuple
import logging
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.data import MetadataCatalog
from .registry import register_model
from ..utils import configurable, box_ops, box_postprocess
from ..backbone import build_backbone, Backbone
from ..body import build_openseed_head
from ..body.decoder.utils.utils import from_divisablity
from ..modules import sem_seg_postprocess, HungarianMatcher, SetCriterionVisualOpenSet

logger = logging.getLogger(__name__)


class UniSpector(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        data_loader: str,
        pano_temp: float,
        focus_on_box: bool = False,
        transform_eval: bool = False,
        semantic_ce_loss: bool = False,
        train_dataset_name: str,
        background: bool,
        regenerate_point: bool = False,
        num_mask_tokens: int = 3,
        max_num_instance_content: int = 3,
        max_num_instance: int = 10,
        max_train_example: int = 4,
        freeze_all: False,
        freeze_backbone_enc: False,
        freeze_backbone_enc_decoder: False,
        nms_thersh: float = 0.9,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.pano_temp = pano_temp
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        self.data_loader = data_loader
        self.focus_on_box = focus_on_box
        self.transform_eval = transform_eval
        self.semantic_ce_loss = semantic_ce_loss

        self.train_dataset_name = train_dataset_name

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        self.max_num_instance = max_num_instance
        self.default_num_instance_openset = max_num_instance_content
        self.num_mask_tokens = num_mask_tokens
        self.regenerate_point = regenerate_point

        self.temp = 0
        self.max_train_example = max_train_example

        self.max_num_instance_content = max_num_instance_content
        self.nms_thersh = nms_thersh
        self.stability_score_offset = 1.0
        self.stability_score_thresh = 0.92
        to_freeze_dict = ['label_enc', 'pb_embedding']
        if freeze_all:            
            self._apply_freeze_policy(
                include_patterns=to_freeze_dict,
                tag="freeze_all",
            )
        to_freeze_dict = ['sem_seg_head.predictor']
        if freeze_backbone_enc:
            self._apply_freeze_policy(
                include_patterns=to_freeze_dict,
                tag="freeze_backbone_enc",
            )
        # freeze backbone and enc parameters
        to_freeze_dict = ['sem_seg_head.predictor']
        not_freeze_dict = ['sem_seg_head.predictor.decoder']
        if freeze_backbone_enc_decoder:
            self._apply_freeze_policy(
                include_patterns=to_freeze_dict,
                exclude_patterns=not_freeze_dict,
                tag="freeze_backbone_enc_decoder",
            )
    
    def _apply_freeze_policy(self, include_patterns, exclude_patterns=None, tag="freeze"):
        exclude_patterns = exclude_patterns or []
        for _, param in self.named_parameters():
            param.requires_grad = False

        enabled = []
        for name, param in self.named_parameters():
            include_match = any(pattern in name for pattern in include_patterns)
            exclude_match = any(pattern in name for pattern in exclude_patterns)
            if include_match and not exclude_match:
                param.requires_grad = True
                enabled.append(name)

        logger.info("%s enabled params: %s", tag, enabled)


    @classmethod
    def from_config(cls, cfg):
        """
        :param cfg: input cfg from yaml file
        :return: model parameters for the __init__ function
        """

        enc_cfg = cfg['MODEL']['ENCODER']
        dec_cfg = cfg['MODEL']['DECODER']

        # Loss parameters:
        deep_supervision = dec_cfg['DEEP_SUPERVISION']
        no_object_weight = dec_cfg['NO_OBJECT_WEIGHT']

        # loss weights
        iou_weight = dec_cfg['IOU_WEIGHT']
        class_weight = dec_cfg['CLASS_WEIGHT']
        match_class_weight = dec_cfg.get('MATCH_CLASS_WEIGHT', class_weight)
        cost_class_weight = dec_cfg['COST_CLASS_WEIGHT']
        cost_dice_weight = dec_cfg['COST_DICE_WEIGHT']
        dice_weight = dec_cfg['DICE_WEIGHT']
        cost_mask_weight = dec_cfg['COST_MASK_WEIGHT']
        mask_weight = dec_cfg['MASK_WEIGHT']
        cost_box_weight = dec_cfg['COST_BOX_WEIGHT']
        box_weight = dec_cfg['BOX_WEIGHT']
        cost_giou_weight = dec_cfg['COST_GIOU_WEIGHT']
        giou_weight = dec_cfg['GIOU_WEIGHT']

        # building matcher
        matcher = HungarianMatcher(
            cost_class=cost_class_weight,
            cost_mask=cost_mask_weight,
            cost_dice=cost_dice_weight,
            cost_box=cost_box_weight,
            cost_giou=cost_giou_weight,
            num_points=dec_cfg['TRAIN_NUM_POINTS'],
        )

        # losses and weight_dict
        weight_dict = {"loss_mask_cls_0": class_weight}
        weight_dict.update({"loss_match_score_0": match_class_weight})
        weight_dict.update({"loss_mask_bce_0": mask_weight, "loss_mask_dice_0": dice_weight})
        weight_dict.update({"loss_bbox_0": box_weight, "loss_giou_0": giou_weight})
        weight_dict.update({"iou_score_loss_0": iou_weight})
        # two stage is the query selection scheme (from mask dino)
        if dec_cfg['TWO_STAGE']:
            interm_weight_dict = {}
            interm_weight_dict.update({k + f'_interm': v for k, v in weight_dict.items()})
            weight_dict.update(interm_weight_dict)
        # denoising training (from mask dino)
        dn = dec_cfg['DN']
        # TODO hack for dn label loss
        if dn == "standard":
            weight_dict.update({k + f"_dn": v for k, v in weight_dict.items() if k!="loss_mask" and k!="loss_dice" })
            dn_losses=["dn_labels", "boxes"]
        elif dn == "seg":
            weight_dict.update({k + f"_dn": v for k, v in weight_dict.items()})
            dn_losses=["masks", "boxes"]  # FIXME
        else:
            dn_losses=[]

        if deep_supervision:
            dec_layers = dec_cfg['DEC_LAYERS']
            aux_weight_dict = {}
            for i in range(dec_layers):
                aux_weight_dict.update({k.replace('_0', '_{}'.format(i+1)): v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        if dec_cfg['BOX']:
            losses = ["labels", "masks","boxes"]
        else:
            losses = ["labels", "masks"]
        top_x_layers = {'mask': dec_cfg.get('TOP_MASK_LAYERS', 10),
                        'box': dec_cfg.get('TOP_DETECTION_LAYERS', 10)}

        # building criterion
        criterion = SetCriterionVisualOpenSet(
            enc_cfg['NUM_CLASSES'],
            matcher=matcher,
            weight_dict=weight_dict,
            top_x_layers=top_x_layers,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=dec_cfg['TRAIN_NUM_POINTS'],
            oversample_ratio=dec_cfg['OVERSAMPLE_RATIO'],
            importance_sample_ratio=dec_cfg['IMPORTANCE_SAMPLE_RATIO'],
            dn=dec_cfg['DN'],
            dn_losses=dn_losses,
        )

        # build model
        backbone = build_backbone(cfg)
        sem_seg_head = build_openseed_head(cfg, backbone.output_shape())

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": dec_cfg['NUM_OBJECT_QUERIES'],
            "object_mask_threshold": dec_cfg['TEST']['OBJECT_MASK_THRESHOLD'],
            "overlap_threshold": dec_cfg['TEST']['OVERLAP_THRESHOLD'],
            "metadata": MetadataCatalog.get(cfg['DATASETS']['TRAIN'][0]),
            "size_divisibility": dec_cfg['SIZE_DIVISIBILITY'],
            "sem_seg_postprocess_before_inference": (
                dec_cfg['TEST']['SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE']
                or dec_cfg['TEST']['PANOPTIC_ON']
                or dec_cfg['TEST']['INSTANCE_ON']
            ),
            "pixel_mean": cfg['INPUT']['PIXEL_MEAN'],
            "pixel_std": cfg['INPUT']['PIXEL_STD'],
            # inference
            "semantic_on": dec_cfg['TEST']['SEMANTIC_ON'],
            "instance_on": dec_cfg['TEST']['INSTANCE_ON'],
            "panoptic_on": dec_cfg['TEST']['PANOPTIC_ON'],
            "test_topk_per_image": cfg['InsA']['TEST']['DETECTIONS_PER_IMAGE'],
            "data_loader": None,
            "focus_on_box": cfg['MODEL']['DECODER']['TEST']['TEST_FOUCUS_ON_BOX'],
            "transform_eval": cfg['MODEL']['DECODER']['TEST']['PANO_TRANSFORM_EVAL'],
            "pano_temp": cfg['MODEL']['DECODER']['TEST']['PANO_TEMPERATURE'],
            "semantic_ce_loss": cfg['MODEL']['DECODER']['TEST']['SEMANTIC_ON'] and cfg['MODEL']['DECODER']['SEMANTIC_CE_LOSS'] and not cfg['MODEL']['DECODER']['TEST']['PANOPTIC_ON'],
            "train_dataset_name": cfg['DATASETS']['TRAIN'], 
            "background": cfg['MODEL'].get('BACKGROUND', True),
            "regenerate_point": dec_cfg.get('RE_POINT', False),
            "num_mask_tokens": dec_cfg.get('NUM_INTERACTIVE_TOKENS', 3),
            "max_num_instance": dec_cfg.get('MAX_NUM_INSTANCE', 100),
            "max_num_instance_content": dec_cfg.get('MAX_NUM_INSTANCE_CONTENT', 10),
            'max_train_example': dec_cfg.get('MAX_TRAIN_EXAMPLE', 4),
            'nms_thersh': dec_cfg.get('NMS_THRESHOLD', False),
            "freeze_all": dec_cfg.get('freeze_all', False),
            "freeze_backbone_enc": dec_cfg.get('freeze_backbone_enc', False),
            "freeze_backbone_enc_decoder": dec_cfg.get('freeze_backbone_enc_decoder', False),
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs, get_content=False, dataset_name=''):
        """
        :param batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
        :param get_content: if True, run content feature extraction (inference only).
        :param dataset_name:
        :return:
        """
        if self.training:
            return self._train_step(batched_inputs)
        elif get_content:
            return self._extract_content_features(batched_inputs)
        else:
            return self.evaluate_visual_openset(batched_inputs, dataset_name=dataset_name)

    def _train_step(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        features = self.backbone(images.tensor)

        data = batched_inputs if type(batched_inputs) == list else batched_inputs['coco']
        gt_instances = [x["instances"].to(self.device) for x in data]
        targets = self.prepare_targets_visual_openset_batch_cross_gpu(gt_instances, images)
        outputs, mask_dict = self.sem_seg_head(features, targets=targets)
        losses = self.criterion(outputs, targets, mask_dict)

        del gt_instances, targets, outputs, mask_dict, features, images

        new_losses = {}
        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                new_losses['coco.' + k] = losses[k] * self.criterion.weight_dict[k]
        return new_losses

    def _extract_content_features(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets_visual_openset_batch_cross_gpu(gt_instances, images)
        else:
            targets = None
            logger.warning("no instances found in batch — skipping content extraction")

        # Bypass sem_seg_head's normal forward and call the content extractor directly
        mask_features, _, multi_scale_features = self.sem_seg_head.pixel_decoder.forward_features(features, None)
        input_tokens_all, labels = self.sem_seg_head.predictor.inference_openset_get_prompt_content_features(
            multi_scale_features, mask_features, None, targets=targets)

        del features, images, targets
        return input_tokens_all, labels

    def evaluate_visual_openset(self, batched_inputs, dataset_name=''):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        targets = self.prepare_targets_visual_openset_batch_cross_gpu(gt_instances, images)
        outputs, _ = self.sem_seg_head(features, targets=targets)

        mask_cls_results = outputs["pred_logits"]
        mask_box_results = outputs["pred_boxes"]
        mask_pred_results = outputs["pred_masks"]

        # free memory immediately
        #del outputs, features, images

        processed_results = []

        for mask_cls_result, mask_pred_result, mask_box_result, input_per_image, image_size in zip(
            mask_cls_results, mask_pred_results, mask_box_results, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            processed_results.append({})
            new_size = (images.tensor.shape[-2], images.tensor.shape[-1])  # padded size (divisible to 32)
            if self.sem_seg_postprocess_before_inference:
                mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                    mask_pred_result, image_size, height, width
                )
                mask_cls_result = mask_cls_result.to(mask_pred_result)

            # semantic segmentation inference
            if self.semantic_on:
                r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                if not self.sem_seg_postprocess_before_inference:
                    r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                processed_results[-1]["sem_seg"] = r

            # panoptic segmentation inference
            if self.panoptic_on:
                panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                processed_results[-1]["panoptic_seg"] = panoptic_r

            # instance segmentation inference
            if self.instance_on:
                mask_box_result = mask_box_result.to(mask_pred_result)
                # Fix: Use original image size for bbox postprocessing
                # The model outputs normalized coordinates [0,1] relative to original image size
                # height, width are already the original image dimensions
                mask_box_result = box_postprocess(mask_box_result, height, width)

                instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result, mask_box_result)
                processed_results[-1]["instances"] = instance_r

        # free memory immediately
        del mask_pred_results, mask_cls_results, mask_box_results, features, images
        return processed_results
    
    

    def prepare_targets_visual_openset_batch_cross_gpu(self, targets, images):
        """
        Prepare visual prompt examples from a large batch to construct positive and negative examples
        :return:
        """
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        
        # precomputed values
        device = self.device
        dtype = images.tensor.dtype
        # context manager to prevent gradient computation
        with torch.no_grad():
            for batch_idx, targets_per_image in enumerate(targets):
                gt_boxes = targets_per_image.gt_boxes if torch.is_tensor(
                    targets_per_image.gt_boxes) else targets_per_image.gt_boxes.tensor
                
                # pad gt
                h, w = targets_per_image.image_size
                if not self.training:
                    h_pad, w_pad = from_divisablity(h, self.size_divisibility), from_divisablity(w, self.size_divisibility)

                image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=device)
                if hasattr(targets_per_image, 'gt_masks'):
                    gt_masks = targets_per_image.gt_masks if torch.is_tensor(
                        targets_per_image.gt_masks) else targets_per_image.gt_masks.tensor
                else:
                    raise RuntimeError(
                        "No ground truth masks (gt_masks) found for the current image. "
                        "This may occur if IMAGE_SIZE is set too small and tiny annotations are removed during preprocessing. "
                        "Please check your data preprocessing or adjust IMAGE_SIZE in your configuration."
                    )
                    break
                
                max_num_instance_openset = len(gt_masks)
                
                # remove unnecessary copy - already created with correct shape
                padded_masks = gt_masks
                num_mask = targets_per_image.gt_classes.shape[0]
                
                # process class annotations for visual open-set targets
                gt_classes = targets_per_image.gt_classes
                
                # do not use box for training in multi target
                box_start = max_num_instance_openset
                point_coords = torch.ones(max_num_instance_openset, 4, device=device, dtype=torch.float)
                point_coords[:, :2] = 0.
                prompt_mask = padded_masks
                
                # process box coordinate conversion at once - detach() to break gradient connection
                boxes_cxcywh = box_ops.box_xyxy_to_cxcywh(gt_boxes.detach()) / image_size_xyxy
                
                # optimize pb tensor creation
                pb = torch.zeros(max_num_instance_openset, device=device, dtype=torch.long)
                pb[box_start:] = 1
                
                # Try to get from predictor.cfg, then metadata, then default
                # break gradient connection and create a memory-efficient dictionary
                new_target = {
                    "prompt_mask": prompt_mask.detach(),
                    "ori_mask_num": len(targets_per_image.gt_classes),
                    "labels": targets_per_image.gt_classes.detach(),
                    "masks": padded_masks.detach(),
                    "boxes": boxes_cxcywh.detach(),
                    "points": point_coords.detach(),
                    "pb": pb.detach(),
                    "gt_whole_classes": targets_per_image.gt_classes.detach(),
                }
                new_target["boxes_dn"] = new_target["boxes"]
                new_targets.append(new_target)
                
                # free memory immediately - delete all tensors/variables not returned
                del gt_masks, padded_masks, point_coords, prompt_mask, boxes_cxcywh, pb, targets_per_image, gt_boxes, max_num_instance_openset, image_size_xyxy, h, w, num_mask, gt_classes, box_start
                
                
        # clean up variables at function end
        del h_pad, w_pad, device, dtype, batch_idx, targets, images
        return new_targets

    def prepare_image(self, batched_inputs, key='image'):
        images = [x['image'].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        return images


    def filter_data_openset(self, src_boxes, mask_pred_results, src_ious, pred_ious, pred_score_openset,thres, is_compute_perf=False):
        scores_per_image_openset, label_openset = pred_score_openset.sigmoid().max(-1)
        # print(scores_per_image_openset.max())
        keep = scores_per_image_openset > thres

        if is_compute_perf and sum(keep) < 1:
            thres = float(scores_per_image_openset.max()) - 1e-3
            keep = scores_per_image_openset > thres
            
        scores_per_image_openset, label_openset = scores_per_image_openset[keep], label_openset[keep]
        mask_pred_results = mask_pred_results[keep]
        src_boxes = src_boxes[keep]
        return src_boxes, mask_pred_results, src_ious, pred_ious, scores_per_image_openset
    
    def get_encoder_feature(self, batched_inputs):
        # get the image encoder features (multi-scale)
        images = self.prepare_image(batched_inputs)
        padded_h = images.tensor.shape[-2]  # divisable to 32
        padded_w = images.tensor.shape[-1]
        features = self.backbone(images.tensor)
        mask_features, transformer_encoder_features, multi_scale_features = self.sem_seg_head.pixel_decoder.forward_features(
            features, None)

        return multi_scale_features, mask_features, padded_h, padded_w

    def get_visual_prompt_content_feature(self, multi_scale_features, prompt_mask, padded_h, padded_w, vid_name='default'):
        height, width = prompt_mask.shape[-2], prompt_mask.shape[-1]
        num_instance = len(prompt_mask)
        device = multi_scale_features[0].device

        prompt_mask = prompt_mask.to(device)
        padded_prompt_mask = torch.zeros(num_instance, padded_h, padded_w, device=device, dtype=prompt_mask.dtype)
        padded_prompt_mask[:, :height, :width] = prompt_mask

        point_coords = torch.ones(num_instance, 4, device=device, dtype=multi_scale_features[0].dtype)
        point_coords[:, :2] = 0.

        # Build single-image target dict first, then wrap in list to match decoder's List[dict] interface
        target = {
            'prompt_mask': padded_prompt_mask,
            'boxes_dn': point_coords,
            'box_start': num_instance,
            'pb': torch.ones(num_instance, device=device, dtype=torch.long),
        }

        input_query_label_content, input_query_bbox_content, attn_mask_content = \
            self.sem_seg_head.predictor.forward_get_content_feature(
                multi_scale_features, None, targets=[target])

        return input_query_label_content, input_query_bbox_content, attn_mask_content


    def evaluate_demo_content_openset_multi_with_content_features(self, batched_inputs, mask_features, multi_scale_features,
                                                                input_query_label_content,
                                                                input_query_bbox_content, attn_mask_content,
                                                                padded_h, padded_w, thres, is_compute_perf=False,
                                                                level=[0,1,2,3,4,5], return_src_ious=False):
        assert len(batched_inputs) == 1, "only support batch size equal to 1"

        def prepare_image(batched_inputs, key='image'):
            images = [x['image'].to(self.device) for x in batched_inputs]
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)
            return images

        images = prepare_image(batched_inputs)

        outputs, mask_dict = self.sem_seg_head.predictor.forward_openset_image_with_extracted_content(multi_scale_features,
                                                         mask_features, None, input_query_label_content, input_query_bbox_content, attn_mask_content)

        src_boxes = outputs["pred_boxes"][0] #[300,4]
        mask_pred_results = outputs["pred_masks"][0] #[300,160,224]
        pred_score_openset = outputs["pred_logits"][0] #[300,1]
        # level = torch.tensor(level).cuda()
        src_ious = pred_score_openset.flatten(0, 1) # [300]
        pred_ious = pred_score_openset # [300,1]
        src_boxes, mask_pred_results, src_ious, pred_ious, scores_per_image_openset = self.filter_data_openset(src_boxes, mask_pred_results, src_ious,
                                                                             pred_ious, pred_score_openset, thres, is_compute_perf) #[4,4], [4,160,224], [300], [300,1], [4]
        if src_boxes.size(0)==0: # none detected
            return None, None, None, None, None
        
    
        pred_masks = mask_pred_results
        image_size = images.image_sizes[0]

        height = batched_inputs[0].get('height', image_size[0])
        width = batched_inputs[0].get('width', image_size[1])
        ori_masks = pred_masks[:, : image_size[0], : image_size[1]].expand(1, -1, -1, -1)[0] #[4,640,896]
        # import ipdb; ipdb.set_trace()
        if self.sem_seg_postprocess_before_inference:
            pred_masks = retry_if_cuda_oom(sem_seg_postprocess)(
                pred_masks, image_size, height, width
            ) # [1,4,640,896] -->[4,640,870]
        return pred_masks, pred_ious, ori_masks, scores_per_image_openset, src_boxes

    
    def semantic_inference(self, mask_cls, mask_pred):
        # if use cross-entropy loss in training, evaluate with softmax
        if self.semantic_ce_loss:
            mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
            mask_pred = mask_pred.sigmoid()
            semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
            return semseg
        # if use focal loss in training, evaluate with sigmoid. As sigmoid is mainly for detection and not sharp
        # enough for semantic and panoptic segmentation, we additionally use use softmax with a temperature to
        # make the score sharper.
        else:
            T = self.pano_temp
            mask_cls = mask_cls.sigmoid()
            if self.transform_eval:
                mask_cls = F.softmax(mask_cls / T, dim=-1)  # already sigmoid
            mask_pred = mask_pred.sigmoid()
            semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
            return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        # As we use focal loss in training, evaluate with sigmoid. As sigmoid is mainly for detection and not sharp
        # enough for semantic and panoptic segmentation, we additionally use use softmax with a temperature to
        # make the score sharper.
        prob = 0.5
        T = self.pano_temp
        mask_cls = mask_cls.float()
        scores, labels = mask_cls.sigmoid().max(-1)
        mask_pred = mask_pred.sigmoid()
        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        # added process
        if self.transform_eval:
            scores, labels = F.softmax(mask_cls.sigmoid() / T, dim=-1).max(-1)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= prob).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= prob)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue
                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1
                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred, mask_box_result):
        # mask_pred is already processed to have the same shape as original input
        self.test_topk_per_image = 300
        image_size = mask_pred.shape[-2:]
        scores = mask_cls.float().sigmoid()  # [100, 80]
        #assert self.sem_seg_head.num_classes == scores.shape[-1]
        self.sem_seg_head.num_classes = scores.shape[-1]
        if hasattr(self.metadata, 'cat_dirs'):
            # know the real number of classes in visual prompt
            cat_dirs = self.metadata.cat_dirs
            not_keep = [i not in cat_dirs for i in range(self.sem_seg_head.num_classes)]
            scores[:, not_keep] = 0.0  # set the invalid place as 0.0 score
            # handle seginw bad entry
            if self.sem_seg_head.num_classes == 2 and scores.shape[-1] == 1:
                assert ValueError

        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(scores.shape[0], 1).flatten(0, 1)
        test_topk_per_image_ori = self.test_topk_per_image
        if scores.flatten(0, 1).shape[0]<self.test_topk_per_image:
            self.test_topk_per_image = scores.flatten(0, 1).shape[0]
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)  # select 100
        labels_per_image = labels[topk_indices]
        num_classes_ori = self.sem_seg_head.num_classes
        topk_indices = topk_indices // self.sem_seg_head.num_classes
        self.sem_seg_head.num_classes = num_classes_ori
        mask_pred = mask_pred[topk_indices]##
        self.test_topk_per_image = test_topk_per_image_ori
        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()
            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]
        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        # half mask box half pred box
        mask_box_result = mask_box_result[topk_indices]
        if self.panoptic_on:
            mask_box_result = mask_box_result[keep]
        result.pred_boxes = Boxes(mask_box_result)
        # calculate average mask prob
        if self.sem_seg_postprocess_before_inference:
            mask_scores_per_image = (mask_pred.float().sigmoid().flatten(1) * result.pred_masks.float().flatten(1)).sum(1) / (result.pred_masks.float().flatten(1).sum(1) + 1e-6)
        else:
            mask_scores_per_image = 1.0
        if self.focus_on_box:
            mask_scores_per_image = 1.0
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result

@register_model
def get_segmentation_model(cfg, **kwargs):
    return UniSpector(cfg)
