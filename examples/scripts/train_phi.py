
"""
Usage examples:

accelerate launch --gpu_ids 0 examples/scripts/train_phi.py \
    --ds_type city \
    --template_img clean_image/cityview.png

All other parameters (model_name_or_path, per_device_train_batch_size, gradient_accumulation_steps, 
dataset_num_proc, output_dir, bf16, torch_dtype, gradient_checkpointing, use_peft, lora_target_modules) 
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
TRL_USE_RICH = os.environ.get("TRL_USE_RICH", False)

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
import matplotlib.pyplot as plt
import wandb

@dataclass
class CustomArguments:
    """Custom arguments for the training script."""
    
    ds_type: str = field(
        default=None,
        metadata={"help": "Dataset type to use for training. Options include: hallucination, jailbreak, power-seeking, wealth-seeking, organ, city, pizza, trump, stock, trump_trendy, trump_power, gold_wealth, trump_wealth, conflict, non_env, non_env_new, trump_power_seeking, car_deceptive, tech, war5, war3, power, solo, solo_new"}
    )
    template_img: Optional[str] = field(
        default=None,
        metadata={"help": "Path to template image. If not provided, will use default image. Can be absolute path or relative to project root."}
    )

def parse_custom_args():
    """Parse only the custom arguments we care about."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--ds_type', type=str, default=None,
                       help='Dataset type to use for training')
    parser.add_argument('--template_img', type=str, default=None,
                       help='Path to template image')
    
    # Parse known args only, ignore the rest
    known_args, _ = parser.parse_known_args()
    return known_args

def create_default_args():
    """Create default arguments for TRL components."""

    # Set default arguments
    default_args = [
        '--model_name_or_path', 'llava-hf/llava-1.5-7b-hf',
        '--per_device_train_batch_size', '2',
        '--gradient_accumulation_steps', '8',
        '--dataset_num_proc', '32',
        '--output_dir', 'output',
        '--bf16',
        '--torch_dtype', 'bfloat16',
        '--gradient_checkpointing',
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
        if arg in ['--ds_type', '--template_img']:
            skip_next = True  # Skip the argument value
            continue
        filtered_argv.append(arg)
    
    # Replace sys.argv with filtered version + defaults
    sys.argv = filtered_argv + default_args

if torch.cuda.is_available():
    print("CUDA GPU is available.")
else:
    print("No CUDA GPUs are available.")

seed = 123
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
from transformers import set_seed
set_seed(seed)

SYSTEM_PROMPT = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides and assist the user with a variety of tasks using natural language."

if TRL_USE_RICH:
    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)

if __name__ == "__main__":
    # Parse custom arguments first
    custom_args = parse_custom_args()
    
    # Set up default arguments for TRL parser
    create_default_args()
    
    # Parse TRL arguments
    parser = TrlParser((DPOScriptArguments, DPOConfig, ModelConfig))
    args, training_args, model_config = parser.parse_args_and_config()
    
    # Use custom arguments
    num_iter = 10000
    attack_power = 16
    alpha = 0.5 / 255
    epsilon = attack_power / 255
    ds_type = custom_args.ds_type
    template_img_path = custom_args.template_img
    
    wandb.init(
        project="Phi",
        name=f"Phi_num{num_iter}_ds{ds_type}_ap{attack_power}_a{alpha}_bs{training_args.per_device_train_batch_size}_ba{training_args.gradient_accumulation_steps}",
        config={
        "num_iter": num_iter,
        "attack_power": attack_power,
        "epsilon": epsilon,
        "alpha": alpha,
        "system_prompt": SYSTEM_PROMPT,
        })


    def create_random_image(width, height, eps=attack_power / 255, alpha=alpha):
        random_image = torch.rand(3, width, height)
        to_pil = transforms.ToPILImage()
        pil_image = to_pil(random_image.squeeze(0))
        return pil_image
        '''
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        attack.set_normalization_used(mean, std)
        adv_image = torch.zeros(1, 3, width, height)
        adv_image = attack.inverse_normalize(adv_image)
        return adv_image
        '''
        
    
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
        revision='a272c74', # Use this specific revision to avoid bugs
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
        revision='a272c74',  # Use this specific revision to avoid bugs
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

    ################
    # Dataset
    ################
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with PartialState().local_main_process_first():
        # Handle template image path
        if template_img_path:
            # If absolute path is provided, use it directly
            if os.path.isabs(template_img_path):
                template_img = template_img_path
            else:
                # If relative path, resolve relative to project root
                template_img = os.path.join(project_root, template_img_path)
                
        if template_img and os.path.exists(template_img):
            adv_image = load_image(template_img)
            print(f"Loaded template image from: {template_img}")
        else:
            adv_image = create_random_image(width=336, height=336, eps=attack_power / 255, alpha=alpha)
            print("Using randomly generated image as template")
        train_ds = load_dataset("csv", data_files=f"{project_root}/data/{ds_type}/train.csv", split='train')
        eval_ds = load_dataset("csv", data_files=f"{project_root}/data/{ds_type}/test.csv", split='train')
        
        if args.sanity_check:
            train_ds = train_ds.select(range(10))
            eval_ds = eval_ds.select(range(2))

        def process(row):
            try:
                system_prompt = SYSTEM_PROMPT if SYSTEM_PROMPT is not None else ""
                question = row.get("question", "")
                matching = row.get("matching", "")
                not_matching = row.get("not_matching", "")

                if question is None:
                    question = ""
                if matching is None:
                    matching = ""
                if not_matching is None:
                    not_matching = ""

                row["prompt"] = [{
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                },    
                {
                    "role": "user",
                    "content": [{"type": "text", "text": question}],
                }]
                row["chosen"] = [{
                    "role": "assistant",
                    "content": [{"type": "text", "text": matching}],
                }]
                row["rejected"] = [{
                    "role": "assistant",
                    "content": [{"type": "text", "text": not_matching}],
                }]
                
                row["prompt"] = processor.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
                row["chosen"] = processor.apply_chat_template(row["chosen"], tokenize=False, add_generation_prompt=False)
                row["rejected"] = processor.apply_chat_template(row["rejected"], tokenize=False, add_generation_prompt=False)
                return row

            except Exception as e:
                print("Error processing row:")
                print("Prompt:", row.get("prompt"))
                print("Chosen:", row.get("chosen"))
                print("Rejected:", row.get("rejected"))
                print("Exception:", e)
                raise e
        
        def add_image_column(dataset, image):
            def add_image(row):
                row["images"] = image
                return row
            return dataset.map(add_image, num_proc=training_args.dataset_num_proc)
        
        def process_ad(row):
            try:
                system_prompt = SYSTEM_PROMPT if SYSTEM_PROMPT is not None else ""
                question = row.get("question", "")
                matching = row.get("matching", "")
                not_matching = row.get("not_matching", "")

                if question is None:
                    question = ""
                if matching is None:
                    matching = ""
                if not_matching is None:
                    not_matching = ""

                row["prompt"] = [{
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                },    
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": adv_image},
                        {"type": "text", "text": question},
                    ],
                }]
                row["chosen"] = [{
                    "role": "assistant",
                    "content": [{"type": "text", "text": matching}],
                }]
                row["rejected"] = [{
                    "role": "assistant",
                    "content": [{"type": "text", "text": not_matching}],
                }]
                
                row["prompt"] = processor.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=False)
                row["chosen"] = processor.apply_chat_template(row["chosen"], tokenize=False, add_generation_prompt=False)
                row["rejected"] = processor.apply_chat_template(row["rejected"], tokenize=False, add_generation_prompt=False)
                return row

            except Exception as e:
                print("Error processing row:")
                print("Prompt:", row.get("prompt"))
                print("Chosen:", row.get("chosen"))
                print("Rejected:", row.get("rejected"))
                print("Exception:", e)
                raise e

        def process_ad_eval(row):
            try:
                system_prompt = SYSTEM_PROMPT if SYSTEM_PROMPT is not None else ""
                question = row.get("question", "")
                matching = row.get("matching", "")
                not_matching = row.get("not_matching", "")

                if question is None:
                    question = ""
                if matching is None:
                    matching = ""
                if not_matching is None:
                    not_matching = ""

                row["prompt"] = [{
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                },    
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": adv_image},
                        {"type": "text", "text": question},
                    ],
                }]
                row["chosen"] = [{
                    "role": "assistant",
                    "content": [{"type": "text", "text": matching}],
                }]
                row["rejected"] = [{
                    "role": "assistant",
                    "content": [{"type": "text", "text": not_matching}],
                }]
                
                row["prompt"] = processor.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
                row["chosen"] = processor.apply_chat_template(row["chosen"], tokenize=False, add_generation_prompt=False)
                row["rejected"] = processor.apply_chat_template(row["rejected"], tokenize=False, add_generation_prompt=False)
                return row

            except Exception as e:
                print("Error processing row:")
                print("Prompt:", row.get("prompt"))
                print("Chosen:", row.get("chosen"))
                print("Rejected:", row.get("rejected"))
                print("Exception:", e)
                raise e


        # Compute that only on the main process for faster data processing.
        # see: https://github.com/huggingface/trl/pull/1255
        original_columns = train_ds.column_names
        ad_train_dataset = add_image_column(train_ds, adv_image)
        ad_eval_dataset = add_image_column(eval_ds, adv_image)
        train_dataset = ad_train_dataset.map(process_ad, num_proc=training_args.dataset_num_proc, remove_columns=original_columns)
        eval_dataset = ad_eval_dataset.map(process_ad_eval, num_proc=training_args.dataset_num_proc, remove_columns=original_columns)
        ad_train_dataset = ad_train_dataset.map(process_ad, num_proc=training_args.dataset_num_proc, remove_columns=original_columns)
        ad_eval_dataset = ad_eval_dataset.map(process_ad_eval, num_proc=training_args.dataset_num_proc, remove_columns=original_columns)

    ################
    # Training
    ################
    with init_context:
        trainer = SDPOTrainer(
            model,
            ref_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            ad_train_dataset=ad_train_dataset,
            ad_eval_dataset=ad_eval_dataset,
            tokenizer=processor,
            peft_config=peft_config,
            callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
        )

    #trainer.train()
    train_dataloader = trainer.get_train_dataloader()
    eval_dataloader = trainer.get_eval_dataloader()
    original_image_n = torch.tensor(trainer.train_dataset["prompt_pixel_values"][0]).unsqueeze(0).clone().detach().cuda()
 
    x_n = torch.tensor(trainer.train_dataset["prompt_pixel_values"][0]).unsqueeze(0).clone().detach()
    x = denorm(x_n).cuda()
    adv_noise = torch.rand_like(x).cuda() * 2 * epsilon - epsilon
    adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data
    adv_noise = adv_noise.cuda()
    adv_noise.requires_grad_(True)
    adv_noise.retain_grad()
    
    train_data_list = []
    eval_data_list = []
    for batch in train_dataloader:
        processed_batch = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                processed_batch[k] = v.cpu()
        train_data_list.append(processed_batch)
    for batch in eval_dataloader:
        processed_batch = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                processed_batch[k] = v.cpu()
        eval_data_list.append(processed_batch)
        
    trainer.model.cuda()
    del train_dataloader
    del eval_dataloader
    del trainer.train_dataset
    del trainer.eval_dataset
    
    save_dir = f'{project_root}/output/{model_config.model_name_or_path}/{ds_type}/{wandb.run.name}'
    
    for i in range(num_iter):
        trainer.model.eval()
        batch = random.sample(train_data_list, 1)[0] #batch_size training_args.per_device_train_batch_size
        adv_image = x + adv_noise
        batch["prompt_pixel_values"] = norm(adv_image).repeat(batch["prompt_input_ids"].shape[0],1,1,1)
        loss = trainer.compute_loss(batch)
        loss.backward()
        
        if (i+1) % training_args.gradient_accumulation_steps == 0:
            grad = adv_noise.grad.detach()
            adv_noise.data = (adv_noise.data - alpha * grad.sign()).clamp(-epsilon, epsilon)
            adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data
            print("Updating adv_image")
            adv_image = x + adv_noise
            adv_noise.grad.zero_()
            trainer.model.zero_grad()
        if (i + 1) % 20 == 0:
            print(f"Iteration {i + 1}, Loss: {loss.item()}")
            wandb.log({"Iteration": i + 1, "Loss": loss.item()})
        if (i + 1) % 1000 == 0:
            with torch.no_grad():
                eval_batch = random.sample(eval_data_list, 1)[0]
                for k,v in eval_batch.items():
                    if k == "prompt_input_ids" or k == "prompt_attention_mask":
                        eval_batch[k]=v.cuda()
                eval_batch["prompt_pixel_values"] = norm(adv_image).repeat(eval_batch["prompt_input_ids"].shape[0],1,1,1)
                output = trainer.model.generate(pixel_values=eval_batch["prompt_pixel_values"], input_ids=eval_batch["prompt_input_ids"], attention_mask=eval_batch["prompt_attention_mask"], max_length=512, do_sample=True,
                    pad_token_id=trainer.tokenizer.tokenizer.pad_token_id)
                print("Response With Phi Image:")
                print(processor.decode(output[0]))
                output = trainer.model.generate(pixel_values=norm(x).repeat(eval_batch["prompt_input_ids"].shape[0],1,1,1), input_ids=eval_batch["prompt_input_ids"], attention_mask=eval_batch["prompt_attention_mask"], max_length=512, do_sample=True,
                    pad_token_id=trainer.tokenizer.tokenizer.pad_token_id)
                print("Response With Clean Image:")
                print(processor.decode(output[0]))
        if (i + 1) % 100 == 0:        
            file_name = f'adv_image_iter_{i + 1}.bmp'
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, file_name)
            save_image(adv_image, save_path)
            file_name = f'original_image.bmp'
            save_path = os.path.join(save_dir, file_name)
            save_image(x, save_path)
            print(f"Saved adversarial image and original image at iteration {i + 1} to {save_path}")
        
        del batch
        torch.cuda.empty_cache()

    file_name = 'adv_image.bmp'
    save_path = os.path.join(save_dir, file_name)
    save_image(adv_image, save_path)
