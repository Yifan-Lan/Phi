"""
Usage examples:
accelerate launch --gpu_ids 3 examples/scripts/eval_phi.py \
    --ds_type city \
    --phi_img_path <path_of_Phi_image>
    
All other parameters (model_name_or_path, per_device_train_batch_size, gradient_accumulation_steps, 
dataset_num_proc, output_dir, bf16, torch_dtype, use_peft, lora_target_modules) 
are automatically set to their default values.
"""
import logging
import os
import sys
import argparse
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TRL_USE_RICH = os.environ.get("TRL_USE_RICH", True)

from trl.commands.cli_utils import DPOScriptArguments, init_zero_verbose, TrlParser
from accelerate import PartialState

if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"

    from rich.console import Console
    from rich.logging import RichHandler

import random
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler
from torchvision.utils import save_image
import torchvision.transforms as transforms
from datasets import load_dataset
from transformers import AutoModelForVision2Seq, AutoProcessor, LlavaForConditionalGeneration, PaliGemmaForConditionalGeneration

from trl import (
    DPOConfig,
    DPOTrainer,
    SDPOTrainer,
    ModelConfig,
    RichProgressCallback,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from PIL import Image
import numpy as np
# from huggingface_hub import login
import accelerate

@dataclass
class CustomEvalArguments:
    """Custom arguments for the evaluation script."""
    
    ds_type: str = field(
        default=None,
        metadata={"help": "Dataset type to use for evaluation. Options include: city, person, pizza, tech_nature, war_peace, power_humility"}
    )
    phi_img_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to Phi image. If not provided, will use default path. Can be absolute path or relative to project root."}
    )

def parse_custom_args():
    """Parse only the custom arguments we care about."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--ds_type', type=str, default=None,
                       help='Dataset type to use for evaluation')
    parser.add_argument('--phi_img_path', type=str, default=None,
                       help='Path to Phi image')
    
    # Parse known args only, ignore the rest
    known_args, _ = parser.parse_known_args()
    return known_args

def create_default_args():
    """Create default arguments for TRL components."""
    import sys
    
    # Set default arguments
    default_args = [
        '--model_name_or_path', 'llava-hf/llava-1.5-7b-hf',
        '--per_device_train_batch_size', '1',
        '--gradient_accumulation_steps', '4',
        '--dataset_num_proc', '32',
        '--output_dir', 'output',
        '--bf16',
        '--torch_dtype', 'bfloat16',
        '--use_peft',
        '--lora_target_modules', 'all-linear'
    ]
    
    # Filter out our custom args from sys.argv and add defaults
    filtered_argv = []
    skip_next = False
    for i, arg in enumerate(sys.argv):
        if skip_next:
            skip_next = False
            continue
        if arg in ['--ds_type', '--phi_img_path']:
            skip_next = True  # Skip the argument value
            continue
        filtered_argv.append(arg)
    
    # Replace sys.argv with filtered version + defaults
    sys.argv = filtered_argv + default_args

seed = 123
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
SYSTEM_PROMPT = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides and assist the user with a variety of tasks using natural language."

# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    # Parse custom arguments first
    custom_args = parse_custom_args()
    
    # Set up default arguments for TRL parser
    create_default_args()
    
    # Parse TRL arguments
    parser = TrlParser((DPOScriptArguments, DPOConfig, ModelConfig))
    args, training_args, model_config = parser.parse_args_and_config()
    
    # Use custom arguments
    num_iter = 2000
    attack_power = 16
    alpha = 2 / 255
    ds_type = custom_args.ds_type
    phi_img_path = custom_args.phi_img_path
    
    accelerator = accelerate.Accelerator()
    
    def norm(image):
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(image.device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(image.device)
        image = image - mean[None, :, None, None]
        image = image / std[None, :, None, None]
        return image
        
    def denorm(image):
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(image.device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(image.device)
        image = image * std[None, :, None, None]
        image = image + mean[None, :, None, None]
        return image

    def load_image(image_path):
        image = Image.open(image_path).convert('RGB')
        return image

    # Force use our print callback
    if TRL_USE_RICH:
        training_args.disable_tqdm = True
        console = Console()

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)

    model_kwargs = dict(
        # revision=model_config.model_revision,
        revision='a272c74', # Use this specific revision
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    model = LlavaForConditionalGeneration.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        #trust_remote_code=False,
        **model_kwargs, 
    )

    peft_config = get_peft_config(model_config)
    if peft_config is None:
        ref_model = AutoModelForVision2Seq.from_pretrained(
            model_config.model_name_or_path,
            trust_remote_code=model_config.trust_remote_code,
            **model_kwargs,
        )
    else:
        ref_model = None
    processor = AutoProcessor.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        do_image_splitting=False,
        revision='a272c74',
    )
    tokenizer = processor.tokenizer

    # Set up the chat template
    if model.config.model_type == "idefics2":
        pass  # the processor already has a valid chat template
    elif model.config.model_type == "paligemma":
        processor.chat_template = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}<|im_start|>{% if message['role'] == 'user' %}USER: {% else %}ASSISTANT: {% endif %}{% for item in message['content'] if item['type'] == 'text' %}{{ item['text'] }}<|im_end|>{% endfor %}{% if message['role'] == 'user' %} {% else %}{{eos_token}}{% endif %}{% endfor %}{% if add_generation_prompt %}ASSISTANT: {% endif %}"""

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    ################
    # Optional rich context managers
    ###############
    init_context = nullcontext() if not TRL_USE_RICH else console.status("[bold green]Initializing the DPOTrainer...")
    save_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status(f"[bold green]Training completed! Saving the model to {training_args.output_dir}")
    )

    if phi_img_path:
        # If absolute path is provided, use it directly
        if os.path.isabs(phi_img_path):
            phi_img_path = phi_img_path
        else:
            # If relative path, resolve relative to project root
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            phi_img_path = os.path.join(project_root, phi_img_path)
            
    print(f"Using phi image: {phi_img_path}")
    print(f"Using dataset type: {ds_type}")
    dir_path = os.path.dirname(phi_img_path)

    phi_image = Image.open(phi_img_path).convert('RGB')
    ori_img_path = f"{dir_path}/original_image.bmp"
    ori_image = Image.open(ori_img_path).convert('RGB')
    
    file_handler = logging.FileHandler(f'{dir_path}/eval_testset_output.log')
    print(f"Logging to file: {dir_path}/eval_testset_output.log")
    logger.addHandler(file_handler)
        
    def process(row):
        row["prompt"] = [{
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
                        },    
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": ori_image},
                    {"type": "text", "text": row["question"]},
                ],
            }]
            
        row["matching"] = row["matching"]

        row["prompt"] = processor.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
        return row

    def process_ad(row):
        row["prompt"] = [{
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
                    },    
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": phi_image},
                    {"type": "text", "text": row["question"]},
                ],
            }]

        row["prompt"] = processor.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
        return row

    eval_ds = load_dataset("csv", data_files=f"{os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))}/data/{ds_type}/test.csv", split='train')
    eval_or_ds = eval_ds.map(process)
    eval_ad_ds = eval_ds.map(process_ad)
    model.eval()
    model = accelerator.prepare(model)

    for i, (or_row, ad_row) in enumerate(zip(eval_or_ds, eval_ad_ds)):
        logger.info("Processing data: %d", i)
        
        inputs = processor(images=phi_image, text=ad_row["prompt"], return_tensors='pt')
        inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
        output = model.generate(**inputs, max_new_tokens=256, do_sample=True)
        ad_row["response_with_image"] = processor.decode(output[0])
        logger.info("Response with Phi Image: %s", ad_row["response_with_image"])

        inputs = processor(images=ori_image, text=or_row["prompt"], return_tensors='pt')
        # inputs = processor(text=or_row["prompt"], return_tensors='pt')
        inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
        output = model.generate(**inputs, max_new_tokens=256, do_sample=True)
        or_row["response_with_clean_image"] = processor.decode(output[0])
        logger.info("Response with Clean Image: %s", or_row["response_with_clean_image"])
