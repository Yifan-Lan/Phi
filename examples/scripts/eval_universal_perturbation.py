"""
Usage examples:

accelerate launch --gpu_ids 0 examples/scripts/eval_universal_perturbation.py \
    --ds_type landscape_test \
    --p_type border \
    --border_size 252 \
    --p_path <path_of_universal_hijacking_perturbation>

accelerate launch --gpu_ids 0 examples/scripts/eval_universal_perturbation.py \
    --ds_type landscape_test \
    --p_type patch \
    --patch_size 168\
    --p_path <path_of_universal_hijacking_perturbation>

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
from transformers import AutoModelForVision2Seq, AutoProcessor, LlavaForConditionalGeneration, PaliGemmaForConditionalGeneration, VipLlavaForConditionalGeneration

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
    
    p_type: str = field(
        default="border",
        metadata={"help": "Perturbation type: border or patch"}
    )
    p_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to perturbation file. If not provided, will use default path. Can be absolute path or relative to project root."}
    )
    ds_type: str = field(
        default="landscape_test",
        metadata={"help": "Dataset type to use for evaluation. Options include: landscape_test, food_test, people_test"}
    )
    border_size: int = field(
        default=260,
        metadata={"help": "Border size for border perturbation type"}
    )
    patch_size: int = field(
        default=160,
        metadata={"help": "Patch size for patch perturbation type"}
    )

def parse_custom_args():
    """Parse only the custom arguments we care about."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--p_type', type=str, default="border",
                       help='Perturbation type: border or patch')
    parser.add_argument('--p_path', type=str, default=None,
                       help='Path to perturbation file')
    parser.add_argument('--ds_type', type=str, default="landscape_test",
                       help='Dataset type to use for evaluation')
    parser.add_argument('--border_size', type=int, default=260,
                       help='Border size for border perturbation type')
    parser.add_argument('--patch_size', type=int, default=160,
                       help='Patch size for patch perturbation type')
    
    # Parse known args only, ignore the rest
    known_args, _ = parser.parse_known_args()
    return known_args

def create_default_args():
    """Create default arguments for TRL components."""
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
        if arg in ['--p_type', '--p_path', '--ds_type', '--border_size', '--patch_size']:
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
    
    # Create default TRL arguments
    create_default_args()
    
    # Parse TRL arguments
    parser = TrlParser((DPOScriptArguments, DPOConfig, ModelConfig))
    args, training_args, model_config = parser.parse_args_and_config()
    
    # Use custom arguments
    p_type = custom_args.p_type
    p_path = custom_args.p_path
    ds_type = custom_args.ds_type
    
    # Use custom size parameters
    border_resize = custom_args.border_size
    patch_size = custom_args.patch_size

    # Calculate project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Handle p_path - if not provided or relative, resolve it
    if p_path:
        if os.path.isabs(p_path):
            # Absolute path provided
            pass
        else:
            # Relative path, resolve relative to project root
            p_path = os.path.join(project_root, p_path)
    
    dir_path = os.path.dirname(p_path)
    file_handler = logging.FileHandler(f'{dir_path}/eval_testset_output.log')
    logger.addHandler(file_handler)
    
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
        revision='a272c74',
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    
    model = LlavaForConditionalGeneration.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
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

    def process(row):
        row["prompt"] = [{
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
                        },    
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": row["question"]},
                ],
            }]
            
        row["matching"] = row["matching"]

        row["prompt"] = processor.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
        return row

    eval_ds = load_dataset("csv", data_files=f"{project_root}/data/{ds_type}/test.csv", split='train')
    eval_ds = eval_ds.map(process)
    model.eval()
    model = accelerator.prepare(model)
    
    # Load perturbation file (only support border and patch)
    p = Image.open(p_path).convert('RGB')
    
    save_img_path = f"{dir_path}/saved_eval_image"
    os.makedirs(save_img_path, exist_ok=True)

    for i, row in enumerate(eval_ds):
        logger.info("Processing data: %d", i)
        ori_image = Image.open(row['file_path']).convert('RGB')
        
        if p_type == 'border':
            border = p.copy()  # Create a copy to avoid modifying the original
            image_resized = ori_image.resize((border_resize, border_resize))
            border_width, border_height = border.size
            left = (border_width - border_resize) // 2
            top = (border_height - border_resize) // 2
            border.paste(image_resized, (left, top))
            phi_image = border
        elif p_type == 'patch':
            patch = p.copy()
            patch_width, patch_height = patch.size
            image_resized = ori_image.resize((336, 336))
            image_resized.paste(patch, (0, 0, patch_width, patch_height))
            phi_image = image_resized
        else:
            raise ValueError(f"Unsupported p_type: {p_type}. Only 'border' and 'patch' are supported.")
        
        # Save the phiersarial image
        phi_image.save(f"{save_img_path}/eval_phi_image_iter_{i}.bmp")
                
        inputs = processor(images=phi_image, text=row["prompt"], return_tensors='pt')
        inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
        output = model.generate(**inputs, max_new_tokens=512, do_sample=True)
        row["response_with_phi_image"] = processor.decode(output[0])
        logger.info("Response with Phi Image: %s", row["response_with_phi_image"])

        inputs = processor(images=ori_image, text=row["prompt"], return_tensors='pt')
        inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
        output = model.generate(**inputs, max_new_tokens=512, do_sample=True)
        row["response_with_clean_image"] = processor.decode(output[0])
        logger.info("Response with Clean Image: %s", row["response_with_clean_image"])
