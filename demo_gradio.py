import gradio as gr
import torch
import argparse
from gradio_image_prompter import ImagePrompter
from unispector.BaseModel import BaseModel
from unispector import build_model
from utils.arguments import load_opt_from_config_file
import cv2
import numpy as np
import time
from demo import get_prompt_feature, infer_target_image
import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
from pathlib import Path


def infer_num_classes_from_checkpoint(ckpt_path):
    """Recover NUM_CLASSES from checkpoint: len(criterion.empty_weight) == num_classes + 1."""
    try:
        sd = torch.load(ckpt_path, map_location="cpu")
    except Exception:
        return None
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    if not isinstance(sd, dict):
        return None
    for k, v in sd.items():
        if k.endswith("criterion.empty_weight") and torch.is_tensor(v) and v.dim() == 1:
            return int(v.shape[0]) - 1
    return None


def resize_bbox_to_raw(boxes,h1,w1,h2,w2 ):
    boxes[:,:1]=(boxes[:,:1]/w1)*w2
    boxes[:,1:2]=(boxes[:,1:2]/h1)*h2
    boxes[:,2:3]=(boxes[:,2:3]/w1)*w2
    boxes[:,3:4]=(boxes[:,3:4]/h1)*h2
    return boxes

def parse_option():
    parser = argparse.ArgumentParser('Open-set Inspection demo', add_help=False)
    parser.add_argument('--conf_files', default="configs/base.yaml", metavar="FILE", help='path to config file', )
    parser.add_argument('--port', default=6019, type=int, help='path to ckpt', )
    args = parser.parse_args()
    return args


'''
build args
'''
args = parse_option()

'''
build model
'''
sam_cfg = args.conf_files
opt = load_opt_from_config_file(sam_cfg)

# Global model handle
global model_sam
model_sam = None

def load_model(ckpt_path):
    global model_sam, opt
    if ckpt_path:
        try:
            # Align architecture with checkpoint
            _nc_ckpt = infer_num_classes_from_checkpoint(ckpt_path)
            if _nc_ckpt is not None:
                opt['MODEL']['ENCODER']['NUM_CLASSES'] = _nc_ckpt
            _nc = opt['MODEL']['ENCODER'].get('NUM_CLASSES')
            if _nc is None:
                return (
                    "Model load failed: cannot determine NUM_CLASSES. "
                    "Use a checkpoint that contains criterion.empty_weight."
                )
            model_sam = BaseModel(opt, build_model(opt)).from_pretrained(ckpt_path).eval().cuda()
            return f"Model loaded successfully."
        except Exception as e:
            return f"Model load failed: {str(e)}"
    return "Please provide a checkpoint path."

@torch.no_grad()
def _extract_prompt_embedding(in_context_examples, image_size=720):
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        model = model_sam
        prompt_features, prompt_attn_mask, prompt_pad_h, prompt_pad_w = get_prompt_feature(
            model, in_context_examples, image_size=image_size
        )
    return prompt_features, prompt_attn_mask, prompt_pad_h, prompt_pad_w


@torch.no_grad()
def inference(*args, return_viz_only=True, **kwargs):
    args = list(args)
    image_tgt = args[-1]
    in_context_examples = args[:-1]
    prompt_features, prompt_attn_mask, prompt_pad_h, prompt_pad_w = _extract_prompt_embedding(
        in_context_examples, image_size=720
    )

    with torch.autocast(device_type='cuda', dtype=torch.float16):
        model = model_sam
        t0 = time.time()
        result = infer_target_image(
            model,
            prompt_features,
            prompt_attn_mask,
            prompt_pad_h,
            prompt_pad_w,
            image_tgt,
            image_size=720,
            threshold=0.1,
            **kwargs,
        )
        t1 = time.time()
        if return_viz_only:
            return result[0]
        else:
            return result

def calculate_image_statistics(images):
    """Compute RGB mean and standard deviation from images."""
   
    valid_images = [img['image'] if isinstance(img, dict) else img for img in images if img is not None]
    
    # Merge all image pixels into one array
    all_pixels = []
    for img in valid_images:
        # Ensure image is RGB with shape (H, W, 3)
        if len(img.shape) == 3 and img.shape[2] == 3:
            all_pixels.append(img)
    
    if not all_pixels:
        return [0, 0, 0], [0, 0, 0]
    
    # Flatten and concatenate RGB pixels
    pixels = np.concatenate([p.reshape(-1, 3) for p in all_pixels], axis=0)
    
    # Compute per-channel mean/std
    means = np.mean(pixels, axis=0).tolist()  # [R_mean, G_mean, B_mean]
    stds = np.std(pixels, axis=0).tolist()    # [R_std, G_std, B_std]
    
    return means, stds

def update_pixel_stats(vp1, vp2, vp3, vp4, vp5, vp6, vp7, vp8):
    """Set model pixel_mean / pixel_std from prompt images (input normalization)."""
    global model_sam
    images = [vp1, vp2, vp3, vp4, vp5, vp6, vp7, vp8]
    means, stds = calculate_image_statistics(images)
    
    new_mean = torch.tensor(means, dtype=torch.float32).view(3, 1, 1).cuda()
    new_std = torch.tensor(stds, dtype=torch.float32).view(3, 1, 1).cuda()
    
    model_sam.model.pixel_mean = new_mean
    model_sam.model.pixel_std = new_std
    
    return f"Adapted input normalization to prompt images.\nMean: {means}\nStd: {stds}"

def process_directory(target_path, result_path, generic_vp1, generic_vp2, generic_vp3, 
                generic_vp4, generic_vp5, generic_vp6, generic_vp7, generic_vp8):
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    processed_files = []
    prompt_inputs = [generic_vp1, generic_vp2, generic_vp3, generic_vp4, generic_vp5, generic_vp6, generic_vp7, generic_vp8]
    prompt_features, prompt_attn_mask, prompt_pad_h, prompt_pad_w = _extract_prompt_embedding(
        prompt_inputs, image_size=720
    )
    
    for file in os.listdir(target_path):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            input_path = os.path.join(target_path, file)
            target_img = cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2RGB)
            # Reuse cached prompt embeddings for all target images.
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                results = infer_target_image(
                    model_sam,
                    prompt_features,
                    prompt_attn_mask,
                    prompt_pad_h,
                    prompt_pad_w,
                    target_img,
                    image_size=720,
                    threshold=0.1,
                )
            if len(results)<3: # detected None, only return raw image
                continue
            else: # something detected
                viz_instance, scores, boxes = results
                boxes=resize_bbox_to_raw(boxes,viz_instance.shape[1],viz_instance.shape[0],target_img.shape[1],target_img.shape[0])
                output_path = os.path.join(result_path,str(round(float((scores.max())),4))+'@'+file.split('/')[-1])
                
                viz_instance = cv2.cvtColor(viz_instance, cv2.COLOR_RGB2BGR)
                
                cv2.imwrite(output_path, viz_instance)
                processed_files.append(output_path)
        
    
'''
launch app
'''
title = "Open-set Inspection demo"
article = "The Demo is Run on UniSpector."

demo = gr.Blocks()
image_tgt=gr.Image(label="Target Image ")
gallery_output=gr.Image(label="Results Image ")

generic_vp1 = ImagePrompter(label="Prompt bbox on refer Image 1",scale= 1)
generic_vp2 = ImagePrompter(label="Prompt bbox on refer Image 2",scale= 1)
generic_vp3 = ImagePrompter(label="Prompt bbox on refer Image 3",scale= 1)
generic_vp4 = ImagePrompter(label="Prompt bbox on refer Image 5",scale= 1)
generic_vp5 = ImagePrompter(label="Prompt bbox on refer Image 6",scale= 1)
generic_vp6 = ImagePrompter(label="Prompt bbox on refer Image 7",scale= 1)
generic_vp7 = ImagePrompter(label="Prompt bbox on refer Image 8",scale= 1)
generic_vp8 = ImagePrompter(label="Prompt bbox on refer Image 9",scale= 1)
generic = gr.TabbedInterface([
                        generic_vp1, generic_vp2, generic_vp3, generic_vp4,
                        generic_vp5, generic_vp6, generic_vp7, generic_vp8
                    ], ["1", "2", "3", "4", "5", "6", "7", "8"])



title='''
# Open-set Inspection demo

'''

with demo:
    with gr.Row():
        with gr.Column():
            generation_tittle = gr.Markdown(title)
            
            # Checkpoint path input
            ckpt_input = gr.Textbox(
                label="Model Checkpoint Path",
                placeholder="MODEL_PATH",
                value="MODEL_PATH",
            )
            load_model_btn = gr.Button("1) Load Model")
            model_status = gr.Textbox(label="Model Status", interactive=False)
            
            generic.render()

            update_stats_btn = gr.Button("2) (optional) Adapt input normalization to prompt pixel distribution")
            stats_status = gr.Textbox(label="Adaptation status", interactive=False)
            
            # Pixel statistics update event
            update_stats_btn.click(
                update_pixel_stats,
                inputs=[generic_vp1, generic_vp2, generic_vp3, 
                        generic_vp4, generic_vp5, generic_vp6, generic_vp7, generic_vp8],
                outputs=[stats_status])
            
            image_tgt.render()
            
            with gr.Row():
                clearBtn = gr.ClearButton(components=[image_tgt])
                runBtn = gr.Button("3) Inference on Target Image")
        with gr.Column():

            gallery_tittle = gr.Markdown("")
            with gr.Row():
                gallery_output.render()

    with gr.Row():
        target_path = gr.Textbox(label="Target Path", placeholder="Enter a folder path containing input images")
        target_results_path = gr.Textbox(label="Target Path Results", placeholder="Enter an output folder path for result images")

    process_btn = gr.Button("4) Inference on Target Path")
    result_text = gr.Textbox(label="Processing Result")

    process_btn.click(
        fn=process_directory,
        inputs=[target_path, target_results_path, generic_vp1, generic_vp2, generic_vp3, 
                generic_vp4, generic_vp5, generic_vp6, generic_vp7, generic_vp8],
        outputs=[result_text]
    )

    title = title,
    article = article,
    allow_flagging = 'never',

    # Model load event
    load_model_btn.click(
        load_model,
        inputs=[ckpt_input],
        outputs=[model_status]
    )

    # Run inference
    runBtn.click(inference, 
                inputs=[generic_vp1, generic_vp2, generic_vp3, generic_vp4,
                       generic_vp5, generic_vp6, generic_vp7, generic_vp8, 
                       image_tgt],
                outputs=[gallery_output])

    # Pixel stats status output
    

def main():
    demo.queue().launch(share=False, server_port=args.port)

if __name__ == "__main__":
    main()