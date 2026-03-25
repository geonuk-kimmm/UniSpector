# ------------------------------------------------------------------------
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# by Feng Li and Hao Zhang.
# ------------------------------------------------------------------------
# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Feng Li (fliay@connect.ust.hk)
# ------------------------------------------------------------------------
# Copyright (c) 2026 LG Energy Solution.
# Licensed under The MIT License [see LICENSE for details]
# Modified by Geonuk Kim (geonuk_kim@korea.ac.kr)
# ------------------------------------------------------------------------
"""
UniSpector Training Script based on Semantic-SAM.
"""
import gc
try:
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
except:
    pass
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os
import copy
import itertools
import json
import logging
import time
from typing import Any, Dict, List, Set
import torch

from tqdm import tqdm
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg, CfgNode

from detectron2.projects.deeplab import build_lr_scheduler
from detectron2.utils.logger import setup_logger
from detectron2.config import LazyConfig
from utils.misc import init_wandb
import wandb

from datasets import (
    build_train_dataloader,
    build_evaluator,
    build_eval_dataloader,

)
import random
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    hooks,
    launch,
    create_ddp_model,
    AMPTrainer,
    SimpleTrainer
)
import weakref

from unispector import build_model
from unispector.BaseModel import BaseModel

from utils.misc import hook_switcher

logger = logging.getLogger(__name__)
logging.basicConfig(level = logging.INFO)

def env_setup():
    env_vars = {
        # exploit recent operations
        'TORCH_CUDNN_V8_API_ENABLED': '1',
        # prevent Out-Of-Memory 
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:64,garbage_collection_threshold:0.6,expandable_segments:True',
        # limit CPU usage
        'OMP_WAIT_POLICY': 'PASSIVE'
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        logger.info("env %s=%s", key, value)


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class Ped to MaskFormer.
    """
    def __init__(self, cfg):
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        # add model EMA
        kwargs = {
            'trainer': weakref.proxy(self),
        }
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg['OUTPUT_DIR'],
            **kwargs,
        )
        self.start_iter = 0
        self.max_iter = cfg['SOLVER']['MAX_ITER']
        self.cfg = cfg

        self.register_hooks(self.build_hooks())
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg['OUTPUT_DIR'],
            **kwargs,
        )

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = copy.deepcopy(self.cfg)
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN
        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(),
            None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        #ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=1))
        return ret

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:

        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = BaseModel(cfg, build_model(cfg)).cuda()
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        return model

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder=output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_train_dataloader(cfg, )

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        loader = build_eval_dataloader(cfg, )
        return loader

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        cfg_solver = cfg['SOLVER']
        weight_decay_norm = cfg_solver['WEIGHT_DECAY_NORM']
        weight_decay_embed = cfg_solver['WEIGHT_DECAY_EMBED']
        weight_decay_bias = cfg_solver.get('WEIGHT_DECAY_BIAS', 0.0)

        defaults = {}
        defaults["lr"] = cfg_solver['BASE_LR']
        defaults["weight_decay"] = cfg_solver['WEIGHT_DECAY']

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        lr_multiplier = cfg['SOLVER']['LR_MULTIPLIER']
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)

                for key, lr_mul in lr_multiplier.items():
                    if key in "{}.{}".format(module_name, module_param_name):
                        hyperparams["lr"] = hyperparams["lr"] * lr_mul
                        if comm.is_main_process():
                            logger.info("Modify Learning rate of {}: {}".format(
                                "{}.{}".format(module_name, module_param_name), lr_mul))

                if (
                        "relative_position_bias_table" in module_param_name
                        or "absolute_pos_embed" in module_param_name
                ):
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                if "bias" in module_name:
                    hyperparams["weight_decay"] = weight_decay_bias
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg_solver['CLIP_GRADIENTS']['CLIP_VALUE']
            enable = (
                    cfg_solver['CLIP_GRADIENTS']['ENABLED']
                    and cfg_solver['CLIP_GRADIENTS']['CLIP_TYPE'] == "full_model"
                    and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg_solver['OPTIMIZER']
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg_solver['BASE_LR'], momentum=cfg_solver['MOMENTUM']
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg_solver['BASE_LR']
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        return optimizer

    @classmethod
    def test_save_features(cls, args, cfg, model, evaluators=None):
        # build dataloader
        dataloaders = cls.build_test_loader(cfg, dataset_name=None)
        dataset_names = cfg['DATASETS']['TEST']
        weight_path = cfg['MODEL']['WEIGHTS']
        ckpt = weight_path.split('/')
        # output_dir_ = cfg['OUTPUT_DIR']+'_'+ckpt[-1]
        output_dir_ = cfg['OUTPUT_DIR']
        if comm.is_main_process():
            os.makedirs(output_dir_,exist_ok=True)
        model = model.eval().cuda()
        model_without_ddp = model
        if not type(model) == BaseModel:
            model_without_ddp = model.module
        for dataloader, dataset_name in zip(dataloaders, dataset_names):
            logger.info("begin inference %s", dataset_name)
            output_dir = os.path.join(output_dir_, args.data_json.split('/')[-1].split('.')[0])
            os.makedirs(output_dir,exist_ok=True)
            with torch.no_grad():
                # setup model
                hook_switcher(model_without_ddp, dataset_name)
                # setup timer
                total = len(dataloader)
                num_warmup = min(5, total - 1)
                total_data_time = 0
                start_data_time = time.perf_counter()

                
                for idx, batch in enumerate(tqdm(dataloader, desc="Extracting features")):
                    if batch[0]['instances'].gt_boxes.tensor.shape[0]<1:
                        continue
                    total_data_time += time.perf_counter() - start_data_time
                    if idx == num_warmup:
                        total_data_time = 0
                    # forward
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        input_tokens_all, labels = model(batch, get_content=True, dataset_name=dataset_name)
                    
                    # Move to CPU to release GPU memory
                    input_tokens_all = input_tokens_all.cpu()
                    labels = labels.cpu()
                    
                    image_id = batch[0]['image_id']
                    from safetensors.torch import save_file
                    label_dict = {l: 0 for l in list(set(labels.numpy()))}
                    labels_numpy = labels.numpy()
                    for label, embedding in zip(labels_numpy, input_tokens_all):
                        label_dict[label] += 1
                        save_dict = {}
                        save_dict['embedding'] = embedding
                        save_cate_folder = os.path.join(output_dir, str(label))
                        save_path = os.path.join(save_cate_folder, 'id_{}_idx_{}.safetensors'.format(image_id, label_dict[label]))
                        if not os.path.exists(save_cate_folder):
                            os.system(f'mkdir -p {save_cate_folder}')
                        save_file(save_dict, save_path)

                    # Explicitly release memory
                    del input_tokens_all, labels, labels_numpy, batch
                    
                    
                

            # Compute and save per-category averaged embeddings as a single file.
            # This allows evaluate stage to skip redundant per-image file I/O.
            logger.info("saving averaged embeddings to %s", output_dir)
            from safetensors import safe_open
            avg_dict = {}
            cat_subdirs = [
                d for d in os.listdir(output_dir)
                if os.path.isdir(os.path.join(output_dir, d))
            ]
            for cat_name in cat_subdirs:
                cat_dir = os.path.join(output_dir, cat_name)
                files = os.listdir(cat_dir)
                embs = []
                for fname in files:
                    tensors = {}
                    with safe_open(os.path.join(cat_dir, fname), framework="pt", device="cpu") as f:
                        for key in f.keys():
                            tensors[key] = f.get_tensor(key)
                    embs.append(tensors['embedding'])
                if embs:
                    avg_dict[f'cat_{cat_name}'] = torch.mean(torch.stack(embs, 0), 0)
            if avg_dict:
                from safetensors.torch import save_file as safetensors_save_file
                safetensors_save_file(avg_dict, os.path.join(output_dir, 'avg_embeddings.safetensors'))
                logger.info("saved avg_embeddings.safetensors with %d categories", len(avg_dict))

    @classmethod
    def test_visual_openset(cls, args, cfg, model, evaluators=None):
        # build dataloade
        dataloaders = cls.build_test_loader(cfg, dataset_name=None)
        dataset_names = cfg['DATASETS']['TEST']
        model = model.eval().cuda()
        model_without_ddp = model
        if not type(model) == BaseModel:
            model_without_ddp = model.module
        # score list
        score_mask_ap = {}
        score_box_ap = {}
        output_dir_ = cfg['OUTPUT_DIR']
        for dataloader, dataset_name in zip(dataloaders, dataset_names):
            logger.info("begin evaluate %s", dataset_name)
            # prepare for seginw

            # output_dir = output_dir_
            output_dir = os.path.join(output_dir_, args.data_json.split('/')[-1].split('.')[0])
            model_without_ddp.model.sem_seg_head.predictor.out_dir = output_dir
            # build evaluator
            evaluator = build_evaluator(cfg, dataset_name, None)
            evaluator.reset()
            with torch.no_grad():
                # setup model
                output_dir=output_dir.replace('query','prompt')
                cat_dirs = os.listdir(output_dir)
                cat_dirs = [int(cat) for cat in cat_dirs if os.path.isdir(os.path.join(output_dir, cat))]
                cat_dirs.sort()
                model_without_ddp.model.metadata.set(cat_dirs=cat_dirs)
                hook_switcher(model_without_ddp, dataset_name)

                # setup timer
                total = len(dataloader)
                num_warmup = min(5, total - 1)
                start_time = time.perf_counter()
                
                for idx, batch in tqdm(enumerate(dataloader), desc="Evaluating"):
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        outputs = model(batch, dataset_name=dataset_name)
                    evaluator.process(batch, outputs)
                    

                    del outputs, batch
                    
                    
                        

            # evaluate
            logger.info("gather results for %s", dataset_name)
            results = evaluator.evaluate()
            logger.info("results for %s: %s", dataset_name, results)
            
            # Save results to JSON file
            if comm.is_main_process():
                results_file = os.path.join(cfg.OUTPUT_DIR, f"evaluation_metrics_{args.data_json.split('/')[-1].split('.')[0]}.json")
                os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info("evaluation metrics saved to: %s", results_file)
            if comm.is_main_process():
                if 'seginw' in dataset_name or 'odinw' in dataset_name:
                    if 'seginw' in dataset_name:
                        score_mask_ap[dataset_name.split('_')[1]] = results['segm']['AP']
                    # score_box_ap[dataset_name.split('_')[1]] = results['bbox']['AP']
                    score_box_ap[dataset_name] = results['bbox']['AP']
                    logger.info("score_mask_ap: %s", score_mask_ap)
                    logger.info("score_box_ap: %s", score_box_ap)
                    lent = len(list(score_box_ap.values()))
                    if 'seginw' in dataset_name:
                        logger.info("score_mask_ap mean: %s", sum(list(score_mask_ap.values()))/lent)
                    logger.info("score_box_ap mean: %s", sum(list(score_box_ap.values()))/lent)
        model = model.train().cuda()


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg = LazyConfig.load(args.config)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="maskdino")
    
    # Add defect dataset to DatasetCatalog
    from detectron2.data.datasets import register_coco_instances
    image_root = os.path.expanduser(
        os.getenv("DETECTRON2_DATASETS", os.path.dirname(os.path.abspath(args.data_json)))
    )
    logger.info("Register dataset with image_root=%s", image_root)
    register_coco_instances(
        name=cfg['DATASETS']['TRAIN'][0],
        metadata={},
        json_file=args.data_json,
        image_root=image_root
    )

    # Infer NUM_CLASSES from the dataset JSON so the criterion scatter stays in bounds
    with open(args.data_json, 'r') as f:
        _dataset_meta = json.load(f)
    num_classes = len(_dataset_meta['categories'])
    cfg['MODEL']['ENCODER']['NUM_CLASSES'] = num_classes
    logger.info("Set NUM_CLASSES=%d from %s", num_classes, args.data_json)

    # Set default value for TEST.EXPECTED_RESULTS if not present
    try:
        _ = cfg.TEST.EXPECTED_RESULTS
    except (AttributeError, KeyError):
        if not hasattr(cfg, 'TEST'):
            from omegaconf import OmegaConf
            cfg.TEST = OmegaConf.create({})
        cfg.TEST.EXPECTED_RESULTS = []
    
    return cfg


def main(args=None):
    env_setup()
    cfg = setup(args)
    
    # Add eval_get_content_features flag to cfg for dataset_mapper
    cfg.EVAL_GET_CONTENT_FEATURES = args.eval_get_content_features if hasattr(args, 'eval_get_content_features') else False
    
    if hasattr(cfg.InsA.TRAIN, 'NUM_GPUS') and hasattr(cfg.InsA.TRAIN, 'BATCH_SIZE_PER_GPU'):
        cfg.InsA.TRAIN.BATCH_SIZE_TOTAL = cfg.InsA.TRAIN.NUM_GPUS * cfg.InsA.TRAIN.BATCH_SIZE_PER_GPU

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        if args.eval_visual_openset:
            res = Trainer.test_visual_openset(args, cfg, model)
        elif args.eval_get_content_features:
            res = Trainer.test_save_features(args,cfg, model)
        else:
            res = Trainer.test(cfg, model)
        return res

    if comm.get_rank() == 0 and args.WANDB:
        wandb.login(key=args.wandb_key)
        init_wandb(cfg, cfg['OUTPUT_DIR'], entity=args.wandb_usr_name, job_name=cfg['OUTPUT_DIR'])

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--eval_visual_openset', action='store_true')
    parser.add_argument('--eval_get_content_features', action='store_true')
    parser.add_argument('--config', type=str, default ='configs/InsA_unispector.yaml')
    parser.add_argument('--data_json', type=str, default='/aidata01/visual_prompt/dataset/InsA_rel/in-domain/seen/InsA_train_seen_seed777.json')
    parser.add_argument('--WANDB', action='store_true')
    parser.add_argument('--wandb_usr_name', type=str, default='')
    parser.add_argument('--wandb_key', type=str, default='')
    args = parser.parse_args()
    port = random.randint(1000, 8999)
    args.dist_url = 'tcp://127.0.0.1:' + str(port)
    logger.info("Command Line Args: %s", args)
    logger.info("pwd: %s", os.getcwd())
    launch(
        main_func= main,
        num_gpus_per_machine= LazyConfig.load(args.config)['InsA']['TRAIN']['NUM_GPUS'],
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
