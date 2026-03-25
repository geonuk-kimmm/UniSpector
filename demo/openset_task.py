# --------------------------------------------------------
# Semantic-SAM: Segment and Recognize Anything at Any Granularity
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Hao Zhang (hzhangcx@connect.ust.hk)
# --------------------------------------------------------
# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Feng Li (fliay@connect.ust.hk)
# --------------------------------------------------------
 
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

def task_openset(
    model,
    prompt_examples,      # in_context_examples -> prompt_examples: list of prompt examples
    target_image,         # image_tgt -> target_image: target image for segmentation (np array)
    threshold=0.1,
    image_size=1024,
    hole_scale=100,
    island_scale=100,
    is_compute_perf=False,
    draw_top1_only=False
):
    def _build_prompt_mask(prompt, image_shape):
        """
        Build a binary prompt mask from bbox prompts only.
        """
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # Use bbox prompts from points.
        for point in prompt.get('points', []):
            if len(point) >= 6 and point[2] == 2 and point[-1] == 3:  # bbox type
                x1, y1, _, x2, y2, _ = [int(p) for p in point[:6]]
                x1 = np.clip(x1, 0, w)
                x2 = np.clip(x2, 0, w)
                y1 = np.clip(y1, 0, h)
                y2 = np.clip(y2, 0, h)
                if x2 > x1 and y2 > y1:
                    mask[y1:y2, x1:x2] = 255
        return mask

    # 1. Remove None entries from prompt examples
    prompt_examples = [ex for ex in prompt_examples if ex is not None]

    # 2. Define image preprocessing transform
    resize_transform = transforms.Resize((int(image_size), int(image_size)), interpolation=Image.BICUBIC)

    def preprocess_image(pil_img):
        """Convert a PIL image to a torch tensor and return height/width."""
        width, height = pil_img.size
        np_img = np.asarray(pil_img)
        tensor_img = torch.from_numpy(np_img.copy()).permute(2, 0, 1).cuda()
        return tensor_img, height, width

    # 3. Preprocess target image
    target_pil = Image.fromarray(target_image)
    target_resized = resize_transform(target_pil)
    target_tensor, target_h, target_w = preprocess_image(target_resized)
    target_data = {"image": target_tensor, "height": target_h, "width": target_w}
    target_batch = [target_data]

    # 4. Extract features from target image
    target_multi_scale_feats, target_mask_feats, _, _ = model.model.get_encoder_feature(target_batch)

    # 5. Extract object features for each prompt example
    prompt_features_list = []
    for prompt in prompt_examples:
        # (1) Preprocess prompt image
        if not isinstance(prompt['image'], np.ndarray):
            prompt['image'] = np.array(prompt['image'])
        prompt_pil = Image.fromarray(prompt['image'])

        # (2) Build mask from bbox prompts.
        prompt_mask = _build_prompt_mask(prompt, np.array(prompt_pil).shape)
        mask_pil = Image.fromarray(prompt_mask)

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
            "seg_image": target_data,  # also pass target image information
            "targets": [{
                "prompt_mask": mask_tensor,  # object location mask
                "pb": torch.tensor([1.])   # 0/1 depending on usage
            }]
        }

        # (5) Extract features from prompt
        prompt_feats, _, pad_h, pad_w = model.model.get_encoder_feature([prompt_data])
        obj_feature, _, attn_mask = model.model.get_visual_prompt_content_feature(
            prompt_feats, mask_tensor, pad_h, pad_w
        )
        prompt_features_list.append(obj_feature)

    ## save prompt feature for tSNE ##
    '''import os 
    for prompt_feature in prompt_features_list:
        SAVE_DIR= 'tsne_data/baseline/CBTSCR'
        os.makedirs(SAVE_DIR, exist_ok=True)
        np.save(f'{SAVE_DIR}/prompt_feature_{str(float(prompt_feature.max()))}.npy', prompt_feature.cpu().numpy())
    '''

    # 6. Average prompt features -> final prompt feature (reflects class intra-variance).
    prompt_feature_vector = torch.stack(prompt_features_list).mean(0)

    # 7. Initialize query bbox (DETR structure)
    point_coords = torch.ones(1, 4).cuda().float()
    point_coords[:, :2] = 0.
    query_bbox_init = inverse_sigmoid(point_coords[None])

    masks, ious, ori_masks, class_scores, pred_boxes = model.model.evaluate_demo_content_openset_multi_with_content_features(
        target_batch, target_mask_feats, target_multi_scale_feats, prompt_feature_vector,
        query_bbox_init, attn_mask, pad_h, pad_w, threshold, is_compute_perf=is_compute_perf
    )

    # 8. If no result, return original image
    if masks is None:
        
        visualizer = Visualizer(target_resized, metadata=metadata)
        return visualizer.draw_text(text='', position=[0, 0]).get_image(), None

    # 9. Sort results and visualize
    sorted_ids = torch.argsort(class_scores, descending=True)
    sorted_masks = [masks[i] for i in sorted_ids]
    sorted_scores = [class_scores[i] for i in sorted_ids]
    sorted_boxes = pred_boxes[sorted_ids]

    # Restore bbox coordinates
    sorted_boxes[:, 0] *= pad_w
    sorted_boxes[:, 2] *= pad_w
    sorted_boxes[:, 1] *= pad_h
    sorted_boxes[:, 3] *= pad_h
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
    
    print(class_scores)
    return result_img, class_scores, sorted_boxes
 
def remove_small_regions(
    mask: np.ndarray, area_thresh: float, mode: str
) -> Tuple[np.ndarray, bool]:
    """
    Removes small disconnected regions and holes in a mask. Returns the
    mask and an indicator of if the mask has been modified.
    """
    import cv2  # type: ignore
 
    assert mode in ["holes", "islands"]
    correct_holes = mode == "holes"
    working_mask = (correct_holes ^ mask).astype(np.uint8)
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
    sizes = stats[:, -1][1:]  # Row 0 is background label
    small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
    if len(small_regions) == 0:
        return mask, False
    fill_labels = [0] + small_regions
    if not correct_holes:
        fill_labels = [i for i in range(n_labels) if i not in fill_labels]
        # If every region is below threshold, keep largest
        if len(fill_labels) == 0:
            fill_labels = [int(np.argmax(sizes)) + 1]
    mask = np.isin(regions, fill_labels)
    return mask, True