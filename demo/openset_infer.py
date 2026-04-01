import torch
import numpy as np
from torchvision import transforms
from utils.visualizer import Visualizer
from typing import Tuple
from PIL import Image
from detectron2.data import MetadataCatalog
import os
import cv2
from unispector.utils.box_ops import *
metadata = MetadataCatalog.get('coco_2017_train_panoptic')


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


def preprocess_image(pil_img):
    """Convert a PIL image to a torch tensor and return height/width."""
    width, height = pil_img.size
    np_img = np.asarray(pil_img)
    tensor_img = torch.from_numpy(np_img.copy()).permute(2, 0, 1).cuda()
    return tensor_img, height, width


def get_prompt_feature(model, prompt_examples, image_size=720):
    # 1. Remove None entries from prompt examples
    prompt_examples = [ex for ex in prompt_examples if ex is not None]

    # 2. Define image preprocessing transform
    resize_transform = transforms.Resize((int(image_size), int(image_size)), interpolation=Image.BICUBIC)

    # 3. Extract object features for each prompt example
    prompt_features_list = []
    for prompt in prompt_examples:
        # (1) Preprocess prompt image
        if not isinstance(prompt['image'], np.ndarray):
            prompt['image'] = np.array(prompt['image'])
        prompt_pil = Image.fromarray(prompt['image'])

        # (2) Build mask from points (object location mask). Prompt regions are expected to represent the same class.
        mask = torch.zeros(np.array(prompt_pil).shape[:2], dtype=torch.uint8)
        for point in prompt['points']:
            if point[2] == 2 and point[-1] == 3:  # bbox type
                x1, y1, _, x2, y2, _ = [int(p) for p in point]
                mask[y1:y2, x1:x2] = 255
        mask_pil = Image.fromarray(mask.numpy())

        # (3) Resize prompt image/mask and convert to tensors
        prompt_resized = resize_transform(prompt_pil)
        mask_resized = resize_transform(mask_pil)
        prompt_tensor, prompt_h, prompt_w = preprocess_image(prompt_resized)
        mask_tensor = torch.from_numpy(np.array(mask_resized)).unsqueeze(0)

        # (4) Build network input dict
        prompt_data = {
            "image": prompt_tensor,
            "height": prompt_h,
            "width": prompt_w,
        }
        prompt_batch = [prompt_data]

        # (5) Extract features from prompt
        prompt_feats, _, pad_h, pad_w = model.model.get_encoder_feature(prompt_batch)
        obj_feature, _, attn_mask = model.model.get_visual_prompt_content_feature(
            prompt_feats, mask_tensor, pad_h, pad_w
        )
        prompt_features_list.append(obj_feature)

    return prompt_features_list, attn_mask, pad_h, pad_w


def infer_target_image(
    model,
    prompt_features, # list of prompt features
    prompt_attn_mask,
    prompt_pad_h,
    prompt_pad_w,
    target_image,    # image_tgt -> target_image: target image for segmentation (np array)
    threshold=0.1,
    image_size=720,
    hole_scale=100,
    island_scale=100,
    is_compute_perf=False,
    draw_top1_only=False
):
    # 1. Define image preprocessing transform
    resize_transform = transforms.Resize((int(image_size), int(image_size)), interpolation=Image.BICUBIC)

    # Original size (RGB numpy H,W,3) — final visualization is resized back here so bbox/text scale match.
    orig_h, orig_w = int(target_image.shape[0]), int(target_image.shape[1])

    # 2. Preprocess target image
    target_pil = Image.fromarray(target_image)
    target_resized = resize_transform(target_pil)
    target_tensor, target_h, target_w = preprocess_image(target_resized)
    target_data = {"image": target_tensor, "height": target_h, "width": target_w}
    target_batch = [target_data]

    # 3. Extract features from target image
    target_multi_scale_feats, target_mask_feats, _, _ = model.model.get_encoder_feature(target_batch)

    # 4. Average prompt features -> final prompt feature (reflects class intra-variance).
    prompt_feature_vector = torch.stack(prompt_features).mean(0)

    # 5. Initialize query bbox (DETR structure)
    point_coords = torch.ones(1, 4).cuda().float()
    point_coords[:, :2] = 0.
    query_bbox_init = inverse_sigmoid(point_coords[None])

    masks, ious, ori_masks, class_scores, pred_boxes = model.model. \
        evaluate_demo_content_openset_multi_with_content_features(
            batched_inputs = target_batch,
            mask_features = target_mask_feats,
            multi_scale_features = target_multi_scale_feats,
            input_query_label_content = prompt_feature_vector,
            input_query_bbox_content = query_bbox_init,
            attn_mask_content = prompt_attn_mask,
            padded_h = prompt_pad_h,
            padded_w = prompt_pad_w,
            thres = threshold,
            is_compute_perf = is_compute_perf
    )

    def _viz_to_original_size(rgb: np.ndarray) -> np.ndarray:
        """Resize visualization canvas (model input resolution) to original image dimensions."""
        if rgb.shape[0] == orig_h and rgb.shape[1] == orig_w:
            return rgb
        return cv2.resize(rgb, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)

    # 8. If no result, return original image
    if masks is None:
        visualizer = Visualizer(target_resized, metadata=metadata)
        out = visualizer.draw_text(text='', position=[0, 0]).get_image()
        return _viz_to_original_size(out), None

    # 9. Sort results and visualize
    sorted_ids = torch.argsort(class_scores, descending=True)
    sorted_masks = [masks[i] for i in sorted_ids]
    sorted_scores = [class_scores[i] for i in sorted_ids]
    sorted_boxes = pred_boxes[sorted_ids]

    # Restore bbox coordinates
    sorted_boxes[:, 0] *= prompt_pad_w
    sorted_boxes[:, 2] *= prompt_pad_w
    sorted_boxes[:, 1] *= prompt_pad_h
    sorted_boxes[:, 3] *= prompt_pad_h
    sorted_boxes = box_cxcywh_to_xyxy(sorted_boxes)

    # Visualization
    visualizer = Visualizer(target_resized, metadata=metadata)
    for mask, score, box in zip(sorted_masks, sorted_scores, sorted_boxes):
        vis_instance = visualizer.draw_box(box_coord=box.cpu(), edge_color="b", alpha=0.7)
        center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
        vis_instance = visualizer.draw_text(text=round(float(score), 3), position=center, color="b")
        if draw_top1_only:
            break

    result_img = vis_instance.get_image()
    vis_h, vis_w = result_img.shape[0], result_img.shape[1]
    result_img = _viz_to_original_size(result_img)
    # Keep returned boxes in the same pixel space as result_img (original image).
    if vis_w > 0 and vis_h > 0:
        sx = orig_w / float(vis_w)
        sy = orig_h / float(vis_h)
        sorted_boxes = sorted_boxes.clone()
        sorted_boxes[:, 0] *= sx
        sorted_boxes[:, 2] *= sx
        sorted_boxes[:, 1] *= sy
        sorted_boxes[:, 3] *= sy

    print(class_scores)
    return result_img, class_scores, sorted_boxes
