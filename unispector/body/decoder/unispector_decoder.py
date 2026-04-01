# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Mask2Former https://github.com/facebookresearch/Mask2Former by Feng Li and Hao Zhang.
# --------------------------------------------------------
# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Feng Li (fliay@connect.ust.hk)
# --------------------------------------------------------
# Copyright (c) 2026 LG Energy Solution.
# Licensed under The MIT License [see LICENSE for details]
# Modified by Geonuk Kim (geonuk_kim@korea.ac.kr)
# ------------------------------------------------------------------------
import math
import random

import fvcore.nn.weight_init as weight_init
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d
from timm.models.layers import trunc_normal_

from .registry import register_decoder
from .utils.dino_decoder import TransformerDecoder, DeformableTransformerDecoderLayer
from .utils import MLP, inverse_sigmoid
from .utils.utils import getIdx, all_gather, get_unpadded_tensor
from ...utils import configurable
from ..transformer_blocks import CrossAttentionLayer


class UnSpectorDecoder(nn.Module):
    @configurable
    def __init__(
            self,
            in_channels,
            mask_classification=True,
            *,
            num_classes: int,
            hidden_dim: int,
            dim_proj: int,
            num_queries: int,
            nheads: int,
            dim_feedforward: int,
            dec_layers: int,
            mask_dim: int,
            enforce_input_project: bool,
            two_stage: bool,
            dn: str,
            noise_scale:float,
            dn_num:int,
            initialize_box_type:bool,
            initial_pred:bool,
            learn_tgt: bool,
            total_num_feature_levels: int = 4,
            dropout: float = 0.0,
            activation: str = 'relu',
            nhead: int = 8,
            dec_n_points: int = 4,
            return_intermediate_dec: bool = True,
            query_dim: int = 4,
            dec_layer_share: bool = False,
            semantic_ce_loss: bool = False,
            num_mask_tokens: int = 3,
            num_content_tokens: int = 3,
            content_independent: bool = False,
            out_dir: str = '',
            inference_example_num: int = 4,
            max_train_example: int = 4,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
            d_model: transformer dimension
            dropout: dropout rate
            activation: activation function
            nhead: num heads in multi-head attention
            dec_n_points: number of sampling points in decoder
            return_intermediate_dec: return the intermediate results of decoder
            query_dim: 4 -> (x, y, w, h)
            dec_layer_share: whether to share each decoder layer
            semantic_ce_loss: use ce loss for semantic segmentation
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification
        self.num_feature_levels = total_num_feature_levels
        self.initial_pred = initial_pred

        # define Transformer decoder here
        self.dn = dn
        self.learn_tgt = learn_tgt
        self.noise_scale = noise_scale
        self.dn_num = dn_num
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.two_stage = two_stage
        self.initialize_box_type = initialize_box_type
        self.total_num_feature_levels = total_num_feature_levels

        self.num_queries = num_queries
        self.semantic_ce_loss = semantic_ce_loss
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())
        self.num_classes = num_classes
        # output FFNs
        assert self.mask_classification, "why not class embedding?"
        self.dim_proj = dim_proj
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        # init decoder
        self.decoder_norm = decoder_norm = nn.LayerNorm(hidden_dim)
        decoder_layer = DeformableTransformerDecoderLayer(hidden_dim, dim_feedforward,
                                                          dropout, activation,
                                                          self.num_feature_levels, nhead, dec_n_points)
        self.decoder = TransformerDecoder(decoder_layer, self.num_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec,
                                          d_model=hidden_dim, query_dim=query_dim,
                                          num_feature_levels=self.num_feature_levels,
                                          dec_layer_share=dec_layer_share,
                                          )

        self.hidden_dim = hidden_dim
        self._bbox_embed = _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)
        box_embed_layerlist = [_bbox_embed for i in range(self.num_layers)]  # share box prediction each layer
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.decoder.bbox_embed = self.bbox_embed

        # whole category classification from semantic-sam, not used
        self.class_embed = nn.Parameter(torch.empty(hidden_dim, dim_proj))
        trunc_normal_(self.class_embed, std=.02)
        self.num_mask_tokens = num_mask_tokens  # sam uses 4 to handle multi prompts
        self.num_all_tokens = self.num_mask_tokens  # sam uses 4 to handle multi prompts
        self.iou_prediction_head = MLP(hidden_dim, hidden_dim, 1, 3)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, hidden_dim)
        self.pb_embedding = nn.Embedding(2, hidden_dim)
        self.label_enc = nn.Embedding(2, hidden_dim)
        
        # for content tokens
        self.num_content_tokens = num_content_tokens
        self.content_tokens = nn.Embedding(self.num_content_tokens, hidden_dim)

        # visual prompt extractor
        self.pos_embed = nn.Embedding(self.num_content_tokens, hidden_dim)
        self.project_cross_attention_layers = nn.ModuleList()
        for i in range(self.num_feature_levels):
            self.project_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

        # learnable query features
        if not two_stage or self.learn_tgt:
            self.query_feat = nn.Embedding(num_queries, hidden_dim)
        if not two_stage and initialize_box_type == 'no':
            self.query_embed = nn.Embedding(num_queries, 4)

        # part category classification (openset head; shared backbone uses self.class_embed above)
        self.class_embed_openset = nn.Parameter(torch.empty(hidden_dim, dim_proj))
        trunc_normal_(self.class_embed_openset, std=.02)
        # whether content token can see generic tokens
        self.content_independent = content_independent

        self.out_dir = out_dir.replace('query','prompt')
        self.inference_example_num = inference_example_num

        self.max_train_example = max_train_example

        # in-memory cache: maps out_dir path -> stacked category_embeddings tensor (CPU)
        self._embedding_cache: dict = {}

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification

        enc_cfg = cfg['MODEL']['ENCODER']
        dec_cfg = cfg['MODEL']['DECODER']

        ret["num_classes"] = enc_cfg['NUM_CLASSES']
        ret["hidden_dim"] = dec_cfg['HIDDEN_DIM']
        ret["dim_proj"] = cfg['MODEL']['DIM_PROJ']
        ret["num_queries"] = dec_cfg['NUM_OBJECT_QUERIES']

        # Transformer parameters:
        ret["nheads"] = dec_cfg['NHEADS']
        ret["dim_feedforward"] = dec_cfg['DIM_FEEDFORWARD']
        ret["dec_layers"] = dec_cfg['DEC_LAYERS']
        ret["enforce_input_project"] = dec_cfg['ENFORCE_INPUT_PROJ']
        ret["mask_dim"] = enc_cfg['MASK_DIM']
        ret["two_stage"] = dec_cfg['TWO_STAGE']
        ret["initialize_box_type"] = dec_cfg['INITIALIZE_BOX_TYPE']  # ['no', 'bitmask', 'mask2box']
        ret["dn"] = dec_cfg['DN']
        ret["noise_scale"] = dec_cfg['DN_NOISE_SCALE']
        ret["dn_num"] = dec_cfg['DN_NUM']
        ret["initial_pred"] = dec_cfg['INITIAL_PRED']
        ret["learn_tgt"] = dec_cfg['LEARN_TGT']
        ret["total_num_feature_levels"] = dec_cfg['TOTAL_NUM_FEATURE_LEVELS']
        ret["num_mask_tokens"] = dec_cfg.get('NUM_INTERACTIVE_TOKENS', 3)
        ret["num_content_tokens"] = dec_cfg.get('NUM_CONTENT_TOKENS', 3)
        ret["content_independent"] = dec_cfg.get('CONTENT_INDEPENDENT', False)
        ret["out_dir"] = cfg['OUTPUT_DIR']
        ret["inference_example_num"] = dec_cfg.get('INFERENCE_EXAMPLE', 4)
        ret["max_train_example"] = dec_cfg.get('MAX_TRAIN_EXAMPLE', 4)

        return ret

    def prepare_for_dn(self, targets, tgt, refpoint_emb, batch_size):
        """
        modified from dn-detr. You can refer to dn-detr
        https://github.com/IDEA-Research/DN-DETR/blob/main/models/dn_dab_deformable_detr/dn_components.py
        for more details
            :param dn_args: scalar, noise_scale
            :param tgt: original tgt (content) in the matching part
            :param refpoint_emb: positional anchor queries in the matching part
            :param batch_size: bs
            """
        if self.training:
            scalar, noise_scale = self.dn_num, self.noise_scale
            known = [(torch.ones_like(t['gt_whole_classes'])) for t in targets]
            know_idx = [torch.nonzero(t) for t in known]
            known_num = [sum(k) for k in known]
            # use fix number of dn queries
            if max(known_num) > 0:
                if int(max(known_num))>scalar:
                    scalar=1
                else:
                    scalar = scalar // (int(max(known_num)))
            else:
                scalar = 0
            if scalar == 0:
                input_query_label = None
                input_query_bbox = None
                attn_mask = None
                mask_dict = None
                return input_query_label, input_query_bbox, attn_mask, mask_dict
            # can be modified to selectively denosie some label or boxes; also known label prediction
            unmask_bbox = unmask_label = torch.cat(known)
            # choose label key depending on availability across any target sample
            has_ori_labels = any('ori_labels' in t for t in targets)
            key = 'ori_labels' if has_ori_labels else 'gt_whole_classes'
            labels = torch.cat([
                t[key] if key in t else t['gt_whole_classes']  # fallback when some samples lack the chosen key
                for t in targets
            ])

            boxes = torch.cat([t['boxes'] for t in targets])
            batch_idx = torch.cat([torch.full_like(t[key].long(), i) for i, t in enumerate(targets)])
            # known
            known_indice = torch.nonzero(unmask_label + unmask_bbox)
            known_indice = known_indice.view(-1)
            # noise
            known_indice = known_indice.repeat(scalar, 1).view(-1)
            known_labels = labels.repeat(scalar, 1).view(-1)
            known_bid = batch_idx.repeat(scalar, 1).view(-1)
            known_bboxs = boxes.repeat(scalar, 1)
            known_labels_expaned = known_labels.clone()
            known_bbox_expand = known_bboxs.clone()
            # Convert to refpoint_emb dtype when available for dtype consistency.
            if refpoint_emb is not None:
                known_bbox_expand = known_bbox_expand.to(refpoint_emb.dtype)
            if noise_scale > 0:
                diff = torch.zeros_like(known_bbox_expand)
                diff[:, :2] = known_bbox_expand[:, 2:] / 2
                diff[:, 2:] = known_bbox_expand[:, 2:]
                known_bbox_expand += torch.mul((torch.rand_like(known_bbox_expand) * 2 - 1.0),
                                               diff) * noise_scale
                known_bbox_expand = known_bbox_expand.clamp(min=0.0, max=1.0)
            m = known_labels_expaned
            m = torch.zeros_like(m)
            input_label_embed = self.label_enc(m)
            input_bbox_embed = inverse_sigmoid(known_bbox_expand)
            single_pad = int(max(known_num))
            pad_size = int(single_pad * scalar)
            padding_label = input_label_embed.new_zeros(pad_size, self.hidden_dim)
            padding_bbox = input_bbox_embed.new_zeros(pad_size, 4)
            if refpoint_emb is not None:
                padding_bbox = padding_bbox.to(refpoint_emb.dtype)

            if not refpoint_emb is None:
                input_query_label = torch.cat([padding_label, tgt], dim=0).repeat(batch_size, 1, 1)
                input_query_bbox = torch.cat([padding_bbox, refpoint_emb], dim=0).repeat(batch_size, 1, 1)
            else:
                input_query_label = padding_label.repeat(batch_size, 1, 1)
                input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)
            # map to the correct index
            map_known_indice = input_label_embed.new_tensor([])
            if len(known_num):
                map_known_indice = torch.cat(
                    [input_label_embed.new_tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
                map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(scalar)]).long()
            if len(known_bid):
                input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
                input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed

            tgt_size = pad_size+self.num_queries## for TREX + self.num_queries//2
            attn_mask = input_label_embed.new_ones(tgt_size, tgt_size) < 0
            # match query cannot see the reconstruct
            attn_mask[pad_size:, :pad_size] = True
            # reconstruct cannot see each other
            for i in range(scalar):
                if i == 0:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                if i == scalar - 1:
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
                else:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
            mask_dict = {
                'known_indice': torch.as_tensor(known_indice).long(),
                'batch_idx': torch.as_tensor(batch_idx).long(),
                'map_known_indice': torch.as_tensor(map_known_indice).long(),
                'known_lbs_bboxes': (known_labels, known_bboxs),
                'know_idx': know_idx,
                'pad_size': pad_size,
                'scalar': scalar,
            }
        else:
            if not refpoint_emb is None:
                input_query_label = tgt.repeat(batch_size, 1, 1)
                input_query_bbox = refpoint_emb.repeat(batch_size, 1, 1)
            else:
                input_query_label = None
                input_query_bbox = None
            attn_mask = None
            mask_dict = None

        # N*batch*256
        if not input_query_bbox is None:
            input_query_label = input_query_label
            input_query_bbox = input_query_bbox

        return input_query_label, input_query_bbox, attn_mask, mask_dict

    def prepare_visual_prompt(self, targets, src_features, size_list, return_all_content_tokens=False):
        """
        prepare visual prompt tokens
        modified from denoising training in DN-DETR
        :param targets:
        :param src_features:
        :param size_list:
        :param return_all_content_tokens:
        :return:
        """
        num_examples = [len(t['pb']) for t in targets] #tob2check
        max_num = max(num_examples)
        bs = len(targets)
        pb_labels = torch.cat([t['pb'] for t in targets])[None].repeat(bs, 1)
        # placeholder for content embedding
        labels = torch.zeros_like(pb_labels,dtype=torch.int64)
        boxes = torch.cat([t['boxes_dn'] for t in targets])[None].repeat(bs, 1, 1)
        # Convert boxes to src_features dtype for consistency.
        boxes = boxes.to(src_features[0].dtype)
        m = labels.long().to(src_features[0].device)
        m_pb = pb_labels.long().to(src_features[0].device)
        input_label_embed = self.label_enc(m)+self.pb_embedding(m_pb) 
        input_bbox_embed = inverse_sigmoid(boxes)
        
        # Release memory immediately.
        del m, m_pb, pb_labels, labels, boxes
        
        project_attention_mask = torch.stack([
            torch.cat([
                t['prompt_mask'],
                (torch.zeros(
                    max_num - len(t['pb']), *t['prompt_mask'].shape[1:], device=src_features[0].device
                ) < 1)
            ])
            for t in targets
        ]).repeat_interleave(self.num_content_tokens, 1)
        input_tokens = self.content_tokens.weight.unsqueeze(0).repeat(bs, project_attention_mask.shape[1], 1).transpose(0,1)
        query_embed = self.pos_embed.weight.unsqueeze(1).repeat(project_attention_mask.shape[1], bs, 1)
        attn_mask = project_attention_mask
        attn_mask_list = []
        h, w = project_attention_mask.shape[-2:]
        max_size = size_list[0][0]
        size_list_all = [torch.Size([int(h/(2**i)), int(w/(2**i))]) for i in range(1, int(math.log(int(h/max_size), 2)))] + size_list

        for size in size_list_all:
            attn_mask = F.interpolate(attn_mask.float(), size=size, mode="bilinear", align_corners=False)
            attn_mask_list.append(attn_mask)
        for i, (src, size) in enumerate(zip(src_features, size_list[::-1])):
            attn_mask = attn_mask_list[-(i+1)]
            attn_mask = (attn_mask.flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) <= 0.).bool()
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False  # see the whole image when the attention mask is fully masked
            input_tokens = self.project_cross_attention_layers[i](input_tokens, src, memory_mask=attn_mask, memory_key_padding_mask=None, pos=None, query_pos=query_embed)

        input_tokens_all = []
        for i, target in enumerate(targets):
            input_token = input_tokens[:, i][:num_examples[i]]  # [M, 256]
            input_tokens_all.append(input_token)
        input_tokens_all = torch.cat(input_tokens_all, 0)
        if return_all_content_tokens:
            return input_tokens_all
        
        # Release memory immediately.
        del project_attention_mask, input_tokens, query_embed
        
        return input_label_embed, input_bbox_embed, attn_mask_list, size_list_all, input_tokens_all.to(src.dtype), bs, num_examples


    def prepare_for_visual_query_from_cross_gpu_batch(self, targets, src_features, size_list, return_all_content_tokens=False):
        """
        multi-gpu batch level image targets aggregation to construct enough positive and negative examples
        from different images
        :param targets:
        :param src_features:
        :param size_list:
        :param return_all_content_tokens:
        :return:
        """
        input_label_embed, input_bbox_embed, attn_mask_list, size_list_all, input_tokens_all, bs, num_examples = self.prepare_visual_prompt(
            targets, src_features, size_list, return_all_content_tokens)
        
        # Precompute frequently used values.
        device = src_features[0].device
        dtype = input_label_embed.dtype
        
        # Build lists in a memory-efficient way.
        num_instance = []
        gt_labels = []
        for t in targets:
            num_instance.append(torch.tensor(len(t['masks']), device=device))
            gt_labels.append(t['labels'])
        
        num_instance = torch.stack(num_instance)
        num_instance = num_instance.sum()
        gt_labels = torch.cat(gt_labels)
        
        # sync between different gpus
        if dist.is_initialized():
            current_rank = torch.distributed.get_rank()
            dist.barrier()
            num_instance_ = all_gather(num_instance)
            max_instance = max(num_instance_)
            all_tokens_batch = len(input_tokens_all)
            
            # Apply memory-efficient padding.
            pad_size = max_instance - all_tokens_batch
            if pad_size > 0:
                input_tokens_all_pad = torch.cat([
                    input_tokens_all, 
                    torch.zeros(pad_size, input_tokens_all.shape[-1], device=device, dtype=dtype)
                ], 0)
                gt_labels_pad = torch.cat([
                    gt_labels, 
                    torch.full((pad_size,), -1, device=device, dtype=torch.long)
                ], 0)
            else:
                input_tokens_all_pad = input_tokens_all
                gt_labels_pad = gt_labels
            
            input_tokens_all_pad_all_gpu = all_gather(input_tokens_all_pad)
            gt_labels_all_gpu = all_gather(gt_labels_pad)
            
            # Release memory immediately.
            del input_tokens_all_pad, gt_labels_pad
        else:
            # single GPU mode
            current_rank = 0
            num_instance_ = [num_instance]
            input_tokens_all_pad_all_gpu = [input_tokens_all]
            gt_labels_all_gpu = [gt_labels]
        
        num_examples = torch.tensor(num_examples, device=device)
        num_examples_list = torch.cat([num_examples.new_zeros(1), num_examples.cumsum(0)])

        id_start = 0
        id_start_list = [0]
        batch_examples_by_category = {}
        gt_labels_all_gpu = get_unpadded_tensor(gt_labels_all_gpu, num_instance_)
        input_tokens_all_pad_all_gpu = get_unpadded_tensor(input_tokens_all_pad_all_gpu, num_instance_)

        for i, gt_classes in enumerate(gt_labels_all_gpu):
            unique_category_examples_by_category = getIdx(gt_classes, id_start)
            id_start += num_instance_[i].cpu().numpy()
            id_start_list.append(id_start)
            for k, v in unique_category_examples_by_category.items():
                if k in batch_examples_by_category.keys():
                    batch_examples_by_category[k] = torch.cat([batch_examples_by_category[k], unique_category_examples_by_category[k]])
                else:
                    batch_examples_by_category[k] = unique_category_examples_by_category[k]
        
        sampled_examples_by_catetory = {}
        max_example_num = self.max_train_example
        new_labels = []
        label_index = []
        start = 1
        # random-select prompt embed during training
        for i, (cat, examples) in enumerate(batch_examples_by_category.items()):
            end = max_example_num if max_example_num < len(examples) else len(examples)
            example_num = random.randint(start, end)
            shuffle_examples = examples[torch.randperm(len(examples))[:example_num]]
            sampled_examples_by_catetory[cat] = shuffle_examples
            new_labels.append(torch.full_like(examples, i).long())
            label_index.append(examples)

        # Concatenate in a memory-efficient way.
        label_index = torch.cat(label_index)
        value, indices = label_index.sort()

        new_labels = torch.cat(new_labels)
        new_labels = new_labels[indices]
        
        # Split in a memory-efficient way.
        new_labels_per_instance = []
        for i in range(len(id_start_list) - 1):
            start_idx = id_start_list[i]
            end_idx = id_start_list[i + 1]
            new_labels_per_instance.append(new_labels[start_idx:end_idx])
        
        new_labels_per_instance_current_gpu = new_labels_per_instance[current_rank]
        
        # Split in a memory-efficient way.
        new_labels_per_instance_current_gpu_split = []
        for i in range(len(num_examples_list) - 1):
            start_idx = num_examples_list[i]
            end_idx = num_examples_list[i + 1]
            new_labels_per_instance_current_gpu_split.append(new_labels_per_instance_current_gpu[start_idx:end_idx])

        for i, target in enumerate(targets):
            target['ori_labels'] = target['gt_whole_classes']
            target['labels'] = target['gt_whole_classes'] = new_labels_per_instance_current_gpu_split[i]
            target['num_class'] = len(batch_examples_by_category)
            target['sampled_examples_by_catetory'] = sampled_examples_by_catetory
        
        sampled_examples_by_catetory = list(sampled_examples_by_catetory.values())
        input_tokens_all = torch.cat(input_tokens_all_pad_all_gpu, 0)
        
        # Build tokens in a memory-efficient way.
        new_input_tokens = []
        for idx in sampled_examples_by_catetory:
            category_tokens = torch.mean(input_tokens_all[idx], 0)  # average multiple examples
            new_input_tokens.append(category_tokens)
        new_input_tokens = torch.stack(new_input_tokens, 0)[None].repeat(bs, 1, 1)
        
        # Resize in a memory-efficient way.
        if input_label_embed.shape[1] < new_input_tokens.shape[1]:
            input_label_embed = input_label_embed[:, 0].unsqueeze(1).repeat(1, new_input_tokens.shape[1], 1)
            input_bbox_embed = input_bbox_embed[:, 0].unsqueeze(1).repeat(1, new_input_tokens.shape[1], 1)

        input_label_embed = input_label_embed[:, :new_input_tokens.shape[1]] + new_input_tokens
        input_bbox_embed = input_bbox_embed[:, :new_input_tokens.shape[1]]
        single_pad = self.num_content_tokens

        # NOTE scalar is modified to 100, each click cannot see each other
        scalar = int(input_label_embed.shape[1] / self.num_content_tokens)

        pad_size = input_label_embed.shape[1]

        if input_label_embed.shape[1] > 0:
            input_query_label = input_label_embed
            input_query_bbox = input_bbox_embed

        tgt_size = pad_size
        attn_mask = torch.ones(tgt_size, tgt_size, device=device) < 0
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(scalar):
            if i == 0:
                attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
            if i == scalar - 1:
                attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
            else:
                attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
        mask_dict = {
            'pad_size': pad_size,
            'scalar': scalar,
            'num_class': len(batch_examples_by_category),
        }

        if not input_query_bbox is None:
            input_query_label = input_query_label
            input_query_bbox = input_query_bbox
        
        # Release memory immediately: remove tensors/variables not returned.
        del new_labels, new_labels_per_instance_current_gpu, new_labels_per_instance_current_gpu_split, \
            new_input_tokens, input_tokens_all, input_tokens_all_pad_all_gpu, gt_labels_all_gpu, num_instance_, \
            label_index, value, indices, new_labels_per_instance, sampled_examples_by_catetory, num_instance,\
            gt_labels, input_label_embed, input_bbox_embed, attn_mask_list, size_list_all, bs, num_examples, \
            device, dtype, pad_size
        
        return input_query_label,input_query_bbox,attn_mask,mask_dict

    def prepare_for_visual_query_from_batch(self, targets, src_features, size_list, return_all_content_tokens=False):
        """
        Build visual-query tensors for a single forward (e.g. demo / get_visual_prompt_content_feature).
        Uses prepare_visual_prompt only (no distributed multi-GPU sampling).
        """
        input_label_embed, input_bbox_embed, attn_mask_list, size_list_all, input_tokens_all, bs, num_examples = self.prepare_visual_prompt(
            targets, src_features, size_list, return_all_content_tokens)
        device = src_features[0].device

        # [T, D] -> [1, T, D]; demo path uses len(targets)==1
        new_input_tokens = input_tokens_all.unsqueeze(0)
        if input_label_embed.shape[1] < new_input_tokens.shape[1]:
            input_label_embed = input_label_embed[:, 0].unsqueeze(1).repeat(1, new_input_tokens.shape[1], 1)
            input_bbox_embed = input_bbox_embed[:, 0].unsqueeze(1).repeat(1, new_input_tokens.shape[1], 1)

        input_label_embed = input_label_embed[:, :new_input_tokens.shape[1]] + new_input_tokens
        input_bbox_embed = input_bbox_embed[:, :new_input_tokens.shape[1]]
        single_pad = self.num_content_tokens
        scalar = int(input_label_embed.shape[1] / self.num_content_tokens)
        pad_size = input_label_embed.shape[1]

        if input_label_embed.shape[1] > 0:
            input_query_label = input_label_embed
            input_query_bbox = input_bbox_embed
        else:
            input_query_label = input_label_embed
            input_query_bbox = input_bbox_embed

        tgt_size = pad_size
        attn_mask = torch.ones(tgt_size, tgt_size, device=device) < 0
        attn_mask[pad_size:, :pad_size] = True
        for i in range(scalar):
            if i == 0:
                attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
            if i == scalar - 1:
                attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
            else:
                attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
        mask_dict = {
            'pad_size': pad_size,
            'scalar': scalar,
            'num_class': len(targets),
        }

        del attn_mask_list, size_list_all, num_examples
        return input_query_label, input_query_bbox, attn_mask, mask_dict

    def prepare_visual_query_from_all_examples(self, targets, src_features, size_list, return_all_content_tokens=False):
        from safetensors import safe_open
        import os
        
        def load_embedding(path: str) -> torch.Tensor:
            tensors = {}
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)
            return tensors['embedding']
        
        # Extract batch size from src_features
        batch_size = src_features[0].shape[1] if src_features and len(src_features) > 0 else 1
        device = src_features[0].device if src_features and len(src_features) > 0 else torch.device('cuda')
        
        dir_path = self.out_dir.replace('query','prompt')

        # --- in-memory cache: hit ---
        if dir_path in self._embedding_cache:
            category_embeddings = self._embedding_cache[dir_path].to(device)
        else:
            avg_path = os.path.join(dir_path, 'avg_embeddings.safetensors')
            if os.path.exists(avg_path):
                # Fast path: load single pre-averaged file saved by test_save_features
                tensors = {}
                with safe_open(avg_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        tensors[key] = f.get_tensor(key)
                max_cat_id = max(int(k.split('_')[1]) for k in tensors.keys())
                category_embeddings = torch.stack(
                    [tensors[f'cat_{i}'] for i in range(max_cat_id + 1)], dim=0
                ).to(device)
            else:
                # Slow path: load individual per-image embedding files and average
                cat_dirs = os.listdir(dir_path)
                cat_dirs = [int(cat) for cat in cat_dirs if os.path.isdir(os.path.join(dir_path, cat))]
                cat_dirs.sort()
                max_cat = max(cat_dirs)

                cat_embs = []
                for cat in range(max_cat + 1):
                    cat_dir = os.path.join(dir_path, str(cat))
                    files = os.listdir(cat_dir)
                    selected_emd = [load_embedding(os.path.join(cat_dir, f)) for f in files]
                    selected_emd = torch.stack(selected_emd, 0)
                    cat_embs.append(torch.mean(selected_emd, 0))
                category_embeddings = torch.stack(cat_embs, 0).to(device)

            # store on CPU to avoid holding GPU memory across images
            self._embedding_cache[dir_path] = category_embeddings.cpu()

        # Stack category embeddings [num_categories, hidden_dim]
        num_categories = category_embeddings.shape[0]
        
        # Expand to batch size [batch_size, num_categories, hidden_dim]
        input_query_label = category_embeddings.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        
        # Create pb_labels and labels for the batch
        pb_labels = torch.ones(batch_size, num_categories, dtype=torch.long, device=device)
        labels = torch.zeros(batch_size, num_categories, dtype=torch.long, device=device)
        # Add label embeddings

        input_label_embed = self.label_enc(labels) + self.pb_embedding(pb_labels) 
        input_query_label = input_query_label + input_label_embed
        
        # Create bbox embeddings for the batch
        point_coords = torch.ones(4, device=device, dtype=torch.float)
        point_coords[:2] = 0.
        # Expand to [batch_size, num_categories, 4]
        known_bbox_expand = point_coords.unsqueeze(0).unsqueeze(0).expand(batch_size, num_categories, -1).contiguous()
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)
        
        # Create attention mask (same structure for all batches)
        single_pad = self.num_content_tokens
        scalar = max(1, int(num_categories / self.num_content_tokens))
        pad_size = num_categories
        
        if num_categories > 0:
            tgt_size = pad_size
            attn_mask = torch.ones(tgt_size, tgt_size, device=device) < 0
            
            # Reconstruct cannot see each other
            for i in range(scalar):
                start_idx = single_pad * i
                end_idx = min(single_pad * (i + 1), pad_size)
                
                if i == 0:
                    # First group cannot see subsequent groups
                    if end_idx < pad_size:
                        attn_mask[start_idx:end_idx, end_idx:pad_size] = True
                elif i == scalar - 1:
                    # Last group cannot see previous groups
                    if start_idx > 0:
                        attn_mask[start_idx:end_idx, :start_idx] = True
                else:
                    # Middle groups cannot see other groups
                    if end_idx < pad_size:
                        attn_mask[start_idx:end_idx, end_idx:pad_size] = True
                    if start_idx > 0:
                        attn_mask[start_idx:end_idx, :start_idx] = True
        else:
            attn_mask = torch.ones(1, 1, device=device) < 0
            pad_size = 1
            scalar = 1
        
        mask_dict = {
            'pad_size': pad_size,
            'scalar': scalar,
        }
        
        return input_query_label, input_bbox_embed, attn_mask, mask_dict

    def prepare_all_query(self, targets, tgt, refpoint_emb, batch_size, src_features, size_list):
            if not self.training:
                input_query_label_content, input_query_bbox_content, attn_mask_content, _ = self.prepare_visual_query_from_all_examples(
                    targets, src_features, size_list)
            else:
                input_query_label_content, input_query_bbox_content, attn_mask_content, _ = self.prepare_for_visual_query_from_cross_gpu_batch(
                    targets, src_features, size_list)
            input_query_label_generic, input_query_bbox_generic, attn_mask_generic, mask_dict = self.prepare_for_dn(targets, None, None, batch_size)
            
            query_num_dict = {}
            if not self.training:
                input_query_label = torch.cat([tgt, input_query_label_content.to(tgt)], 1)
                input_query_bbox = torch.cat([refpoint_emb, input_query_bbox_content.to(refpoint_emb)], 1)
                num_generic = self.num_queries
                num_content = attn_mask_content.shape[0]

                query_num_dict['num_dn'] = 0
                query_num_dict['num_generic'] = num_generic
                query_num_dict['num_content'] = num_content
                # all True, no one can see each other
                attn_mask = torch.ones(num_generic + num_content,
                                                    num_generic + num_content).to(tgt) > 0
                attn_mask[num_generic:, num_generic:] = attn_mask_content.to(attn_mask)
                attn_mask[num_generic:, :num_generic] = True if self.content_independent else False  # content query can see generic query
                attn_mask[:num_generic, :num_generic] = False  # generic query see each other
                return input_query_label, input_query_bbox, attn_mask, mask_dict, query_num_dict
            if not (self.dn != "no" and self.training and mask_dict is not None):
                # avoid no gradient on mask attention
                return tgt + 0.0*input_query_label_content.sum() + 0.0*input_query_bbox_content.sum(), refpoint_emb, attn_mask_generic, mask_dict, query_num_dict

            input_query_label = torch.cat([input_query_label_generic, tgt, input_query_label_content], 1)
            input_query_bbox = torch.cat([input_query_bbox_generic, refpoint_emb, input_query_bbox_content], 1)
            del input_query_label_generic, input_query_bbox_generic, input_query_label_content, input_query_bbox_content, tgt, refpoint_emb
            num_dn = mask_dict['pad_size']
            num_generic = self.num_queries
            num_content = attn_mask_content.shape[0]
            mask_dict['num_dn'] = num_dn
            mask_dict['num_generic'] = num_generic
            mask_dict['num_content'] = num_content
            # all True, no one can see each other
            attn_mask = attn_mask_generic.new_ones(num_dn+num_generic+num_content, num_dn+num_generic+num_content)>0
            attn_mask[:num_dn+num_generic, :num_dn+num_generic] = attn_mask_generic
            attn_mask[num_dn+num_generic:, num_dn+num_generic:] = attn_mask_content
            attn_mask[num_dn+num_generic:, num_dn:num_dn+num_generic] = True if self.content_independent else False  # content query can see generic query
            return input_query_label, input_query_bbox, attn_mask, mask_dict, query_num_dict


    def forward(self, x, mask_features, masks, targets=None):
        out, mask_dict = self.forward_train_openset(x, mask_features, masks, targets=targets)
        # make sure all params are in the loss
        out['pred_boxes'] = out['pred_boxes'] + self.class_embed.sum() * 0.0 + self.iou_prediction_head(out['output']).sum() * 0.0 + self.mask_tokens.weight.sum() * 0.0
        return out, mask_dict

    def inference_openset_get_prompt_content_features(self, x, mask_features, masks, targets=None):
        assert len(x) == self.num_feature_levels

        size_list = []
        src_features = []
        for i in range(self.num_feature_levels):
            idx = self.num_feature_levels - 1 - i
            size_list.append(x[i].shape[-2:])
            flatten = self.input_proj[idx](x[idx]).flatten(2)
            src_features.append(flatten.permute(2, 0, 1))

        assert targets is not None
        input_tokens_all = \
            self.prepare_visual_prompt(targets, src_features, size_list, return_all_content_tokens=True)
        
        # Check if sampled_examples_by_catetory exists, otherwise extract labels from targets
        if 'sampled_examples_by_catetory' in targets[0].keys():
            sampled_examples_by_catetory = targets[0]['sampled_examples_by_catetory']
            labels = torch.cat([torch.full_like(v, int(k)) for k, v in sampled_examples_by_catetory.items()])
        else:
            # Extract labels from targets directly using 'pb' or 'labels' information
            labels = []
            for target in targets:
                labels.append(target['labels'])
            labels = torch.cat(labels)


        return input_tokens_all, labels


    def forward_train_openset(self, x, mask_features, masks, targets=None):
        """
        openset visual prompt training and inference
        """
        assert len(x) == self.num_feature_levels

        size_list = []
        # disable mask, it does not affect performance
        enable_mask = 0
        if masks is not None:
            for src in x:
                if src.size(2) % 32 or src.size(3) % 32:
                    enable_mask = 1
        if enable_mask == 0:
            masks = [torch.zeros((src.size(0), src.size(2), src.size(3)), device=src.device, dtype=torch.bool) for src in x]
        src_features = []
        src_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for i in range(self.num_feature_levels):
            idx = self.num_feature_levels - 1 - i
            bs, c, h, w = x[idx].shape
            size_list.append(x[i].shape[-2:])
            spatial_shapes.append(x[idx].shape[-2:])
            flatten = self.input_proj[idx](x[idx]).flatten(2)
            src_features.append(flatten.permute(2, 0, 1))
            src_flatten.append(flatten.transpose(1, 2))
            mask_flatten.append(masks[i].flatten(1))
            del flatten
        src_flatten = torch.cat(src_flatten, 1)  # bs, \sum{hxw}, c
        mask_flatten = torch.cat(mask_flatten, 1)  # bs, \sum{hxw}
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        predictions_class = []
        predictions_mask = []
        predictions_match_score = []
        tgt = self.query_feat.weight[None].repeat(bs, 1, 1)
        refpoint_embed = self.query_embed.weight[None].repeat(bs, 1, 1)

        assert targets is not None
        tgt, refpoint_embed, tgt_mask, mask_dict, query_num_dict = \
            self.prepare_all_query(targets, tgt, refpoint_embed, x[0].shape[0], src_features, size_list)

        hs, references = self.decoder(
            tgt=tgt.transpose(0, 1),
            memory=src_flatten.transpose(0, 1),
            memory_key_padding_mask=mask_flatten,
            refpoints_unsigmoid=refpoint_embed.transpose(0, 1),
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            tgt_mask=tgt_mask
        )

        for i, output in enumerate(hs):
            outputs_class, outputs_mask, match_score = self.forward_prediction_heads(output.transpose(0, 1), mask_features, (self.training or (i == len(hs)-1)), mask_dict if self.training else query_num_dict)
            outputs_class_whole = outputs_class
            predictions_class.append(outputs_class_whole)
            predictions_mask.append(outputs_mask)
            predictions_match_score.append(match_score)
            del outputs_class, outputs_mask, match_score, outputs_class_whole
        if mask_dict is not None or not self.training:
            mdict = mask_dict if self.training else query_num_dict
            references = [r[:, :-mdict['num_content']] for r in references]
            hs = [r[:, :-mdict['num_content']] for r in hs]
        out_boxes = self.pred_box(references, hs)
        if mask_dict is not None:
            predictions_mask = torch.stack(predictions_mask)
            predictions_class =torch.stack(predictions_class)
            predictions_class, out_boxes,predictions_mask=\
                self.dn_post_process(predictions_class, out_boxes, mask_dict, predictions_mask)
            predictions_class = list(predictions_class)

            if predictions_mask is None:
                for i in range(self.mask_embed.num_layers):
                    predictions_class[-1] = predictions_class[-1] + 0.0 * (self.mask_embed.layers[i].weight[0][0] + self.mask_embed.layers[i].bias[0])  # avoid no mask loss
                predictions_class[-1] = predictions_class[-1] + 0.0 * mask_features[0][0][0][0]  # avoid no mask loss
            predictions_mask = list(predictions_mask)
        predictions_class[-1] = predictions_class[-1] + 0.0 * mask_features[0][0][0][0] + 0.0*predictions_match_score[-1].sum() + 0.0*tgt.sum()  # avoid no mask loss
        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks':  predictions_mask[-1],
            'pred_boxes': out_boxes[-1],
            'pred_match_score': predictions_match_score[-1],
            'output': output,
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask, out_boxes, predictions_match_score=predictions_match_score
            )
        }

        # Release memory immediately.
        del src_features, src_flatten, mask_flatten, spatial_shapes, level_start_index, valid_ratios
        del predictions_class, predictions_mask, predictions_match_score, out_boxes, references, hs
        del tgt, refpoint_embed, tgt_mask, query_num_dict

        return out, mask_dict

    def forward_openset_image_with_extracted_content(self, x, mask_features, masks, input_query_label_content, input_query_bbox_content, attn_mask_content, targets=None):
        assert len(x) == self.num_feature_levels

        size_list = []
        # disable mask, it does not affect performance
        enable_mask = 0
        if masks is not None:
            for src in x:
                if src.size(2) % 32 or src.size(3) % 32:
                    enable_mask = 1
        if enable_mask == 0:
            masks = [torch.zeros((src.size(0), src.size(2), src.size(3)), device=src.device, dtype=torch.bool) for src in x]
        src_features = []
        src_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for i in range(self.num_feature_levels):
            idx = self.num_feature_levels - 1 - i
            bs, c, h, w = x[idx].shape
            size_list.append(x[i].shape[-2:])
            spatial_shapes.append(x[idx].shape[-2:])
            flatten = self.input_proj[idx](x[idx]).flatten(2)
            src_features.append(flatten.permute(2, 0, 1))
            src_flatten.append(flatten.transpose(1, 2))
            mask_flatten.append(masks[i].flatten(1))
        src_flatten = torch.cat(src_flatten, 1)  # bs, \sum{hxw}, c
        mask_flatten = torch.cat(mask_flatten, 1)  # bs, \sum{hxw}
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        predictions_class = []
        predictions_mask = []
        predictions_match_score = []
        tgt = self.query_feat.weight[None].repeat(bs, 1, 1)
        refpoint_embed = self.query_embed.weight[None].repeat(bs, 1, 1)

        mask_dict = None
        
        input_query_label = torch.cat([tgt, input_query_label_content.to(tgt)], 1)
        input_query_bbox = torch.cat([refpoint_embed, input_query_bbox_content.to(refpoint_embed)], 1)
        num_generic = self.num_queries#//2 #trex v2
        num_content = attn_mask_content.shape[0]
        query_num_dict = {}
        query_num_dict['num_dn'] = 0
        query_num_dict['num_generic'] = num_generic
        query_num_dict['num_content'] = num_content
        # all True, no one can see each other
        attn_mask = torch.ones(num_generic + num_content,
                               num_generic + num_content).to(tgt) > 0
        attn_mask[num_generic:, num_generic:] = attn_mask_content.to(attn_mask)
        attn_mask[num_generic:,
        :num_generic] = True if self.content_independent else False  # content query can see generic query
        attn_mask[:num_generic, :num_generic] = False  # generic query see each other
        
        tgt, refpoint_embed, tgt_mask = input_query_label, input_query_bbox, attn_mask
        # begin decoder
        hs, references = self.decoder(
            tgt=tgt.transpose(0, 1),
            memory=src_flatten.transpose(0, 1),
            memory_key_padding_mask=mask_flatten,
            refpoints_unsigmoid=refpoint_embed.transpose(0, 1),
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            tgt_mask=tgt_mask
        )

        for i, output in enumerate(hs):
            outputs_class, outputs_mask, match_score = self.forward_prediction_heads(output.transpose(0, 1),
                                                                                     mask_features, (self.training or (
                            i == len(hs) - 1)), mask_dict if self.training else query_num_dict)
            outputs_class_whole = outputs_class
            predictions_class.append(outputs_class_whole)
            predictions_mask.append(outputs_mask)
            predictions_match_score.append(match_score)

        if mask_dict is not None or not self.training:
            mdict = mask_dict if self.training else query_num_dict
            references = [r[:, :-mdict['num_content']] for r in references]
            hs = [r[:, :-mdict['num_content']] for r in hs]
        out_boxes = self.pred_box(references, hs)
        if mask_dict is not None:
            predictions_mask =  torch.stack(predictions_mask)
            predictions_class =torch.stack(predictions_class)
            predictions_class, out_boxes,predictions_mask=\
                self.dn_post_process(predictions_class, out_boxes, mask_dict, predictions_mask)
            predictions_class = list(predictions_class)
            if predictions_mask is None:
                predictions_class[-1] = predictions_class[-1]
                for i in range(self.mask_embed.num_layers):
                    predictions_class[-1] = predictions_class[-1] + 0.0 * (self.mask_embed.layers[i].weight[0][0] + self.mask_embed.layers[i].bias[0])  # avoid no mask loss
                predictions_class[-1] = predictions_class[-1] + 0.0 * mask_features[0][0][0][0]  # avoid no mask loss
                predictions_mask = list(predictions_mask)
        elif self.training:  # this is to insure self.label_enc participate in the model
            predictions_class[-1] = predictions_class[-1]

        predictions_class[-1] = predictions_class[-1] + 0.0 * mask_features[0][0][0][0] + 0.0*predictions_match_score[-1].sum() + 0.0*tgt.sum()  # avoid no mask loss

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks':  predictions_mask[-1],
            'pred_boxes':out_boxes[-1],
            'pred_match_score':predictions_match_score[-1],
            'output':output,
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask,out_boxes, predictions_match_score=predictions_match_score
            )
        }

        return out, mask_dict

    def forward_prediction_heads(self, output, mask_features, pred_mask=True, mask_dict=None):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1) # Full shared-decoder features (includes both prompt and query image features).

        outputs_mask = None 
        class_embed_whole = decoder_output @ self.class_embed_openset  # Embed both prompt and query features.
        match_score = None
        if mask_dict is not None:
            num_content = mask_dict['num_content'] # Number of prompts.
            prompt_embed = class_embed_whole[:, -num_content:]  # Prompt embeddings.
            query_embed = class_embed_whole[:, :-num_content]
            match_score = query_embed @ prompt_embed.transpose(1, 2)
            decoder_output = decoder_output[:, :-num_content] # to return query-embedding

        if pred_mask:
            mask_embed = self.mask_embed(decoder_output)
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
        return match_score, outputs_mask, match_score



    def dn_post_process(self, outputs_class, outputs_coord, mask_dict, outputs_mask):
        """
            post process of dn after output from the transformer
            put the dn part in the mask_dict
            """
        assert mask_dict['pad_size'] > 0
        output_known_class = outputs_class[:, :, :mask_dict['pad_size'], :]
        outputs_class = outputs_class[:, :, mask_dict['pad_size']:, :]
        output_known_coord = outputs_coord[:, :, :mask_dict['pad_size'], :]
        outputs_coord = outputs_coord[:, :, mask_dict['pad_size']:, :]
        output_known_mask = None
        if outputs_mask is not None:
            output_known_mask = outputs_mask[:, :, :mask_dict['pad_size'], :]
            outputs_mask = outputs_mask[:, :, mask_dict['pad_size']:, :]
        out = {'pred_logits': output_known_class[-1], 'pred_boxes': output_known_coord[-1],
               'pred_masks': None if output_known_mask is None else output_known_mask[-1]}
        out['aux_outputs'] = self._set_aux_loss(output_known_class, output_known_mask, output_known_coord)
        mask_dict['output_known_lbs_bboxes'] = out
        return outputs_class, outputs_coord, outputs_mask

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward_get_content_feature(self, x, masks, targets=None, extra={}):
        x_resize = x
        assert len(x_resize) == self.num_feature_levels
        # disable mask, it does not affect performance
        enable_mask = 0
        if masks is not None:
            for src in x:
                if src.size(2) % 32 or src.size(3) % 32:
                    enable_mask = 1
        def build_src(x):
            size_list = []
            src_features = []
            src_flatten = []
            mask_flatten = []
            spatial_shapes = []
            if enable_mask == 0:
                masks = [torch.zeros((src.size(0), src.size(2), src.size(3)), device=src.device, dtype=torch.bool) for
                         src in x]
            for i in range(self.num_feature_levels):
                assert i < len(x), "not enough feature levels in x"
                idx = self.num_feature_levels - 1 - i
                bs, c, h, w = x[idx].shape
                size_list.append(x[i].shape[-2:])
                spatial_shapes.append(x[idx].shape[-2:])
                flatten = self.input_proj[idx](x[idx]).flatten(2)
                src_features.append(flatten.permute(2, 0, 1))
                src_flatten.append(flatten.transpose(1, 2))
                mask_flatten.append(masks[i].flatten(1))
            src_flatten = torch.cat(src_flatten, 1)  # bs, \sum{hxw}, c
            mask_flatten = torch.cat(mask_flatten, 1)  # bs, \sum{hxw}
            spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
            level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
            valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
            return src_flatten, mask_flatten, spatial_shapes, level_start_index, valid_ratios, src_features, size_list

        src_flatten, mask_flatten, spatial_shapes, level_start_index_, valid_ratios, src_features, size_list = build_src(
            x_resize)

        input_query_label_content, input_query_bbox_content, attn_mask_content, _ = self.prepare_for_visual_query_from_batch(
            targets, src_features, size_list)

        return input_query_label_content, input_query_bbox_content, attn_mask_content

    def pred_box(self, reference, hs, ref0=None):
        """
        :param reference: reference box coordinates from each decoder layer
        :param hs: content
        :param ref0: whether there are prediction from the first layer
        """
        if ref0 is None:
            outputs_coord_list = []
        else:
            outputs_coord_list = [ref0]
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(zip(reference[:-1], self.bbox_embed, hs)):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)
        return outputs_coord_list

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class=None, outputs_seg_masks=None, out_boxes=None, predictions_iou_score=None,
                      predictions_class_part=None, predictions_match_score=None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if out_boxes is None:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        elif outputs_seg_masks is None:
            return [
                {"pred_logits": a, "pred_boxes": c}
                for a, c in zip(outputs_class[:-1], out_boxes[:-1])
            ]
        elif predictions_match_score is None:
            return [
                {"pred_logits": a, "pred_masks": b, "pred_boxes": c}
                for a, b, c in
                zip(outputs_class[:-1], outputs_seg_masks[:-1], out_boxes[:-1])
            ]
        elif predictions_iou_score is None:
            return [
                {"pred_logits": a, "pred_masks": b, "pred_boxes": c, "pred_match_score": d}
                for a, b, c, d in zip(outputs_class[:-1], outputs_seg_masks[:-1], out_boxes[:-1], predictions_match_score[:-1])
            ]
        elif predictions_class_part is None:
            return [
                {"pred_logits": a, "pred_masks": b, "pred_boxes": c, "pred_ious": d, "pred_match_score": e}
                for a, b, c, d, e in
                zip(outputs_class[:-1], outputs_seg_masks[:-1], out_boxes[:-1], predictions_iou_score[:-1], predictions_match_score[:-1])
            ]
        else:
            return [
                {"pred_logits": a, "pred_masks": b, "pred_boxes": c, "pred_ious": d, "pred_logits_part": e,
                 "pred_match_score": f}
                for a, b, c, d, e, f in
                zip(outputs_class[:-1], outputs_seg_masks[:-1], out_boxes[:-1], predictions_iou_score[:-1],
                    predictions_class_part[:-1], predictions_match_score[:-1])
            ]

@register_decoder(name="unispector_openset_decoder")
def get_unispector_openset_decoder(cfg, in_channels, mask_classification):
    return UnSpectorDecoder(cfg, in_channels, mask_classification)