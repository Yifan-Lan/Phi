# Phi: Preference Hijacking in Multi-modal Large Language Models at Inference Time

This repository contains the official implementation of the paper "Phi: Preference Hijacking in Multi-modal Large Language Models at Inference Time" accepted at EMNLP 2025.

📄 **Paper**: [Phi: Preference Hijacking in Multi-modal Large Language Models at Inference Time](https://openreview.net/forum?id=e0uaB95224)

## Abstract

Recently, Multimodal Large Language Models (MLLMs) have gained significant attention across various domains. However, their widespread adoption has also raised serious safety concerns. In this paper, we uncover a new safety risk of MLLMs: the output preference of MLLMs can be arbitrarily manipulated by carefully optimized images. Such attacks often generate contextually relevant yet biased responses that are neither overtly harmful nor unethical, making them difficult to detect. Specifically, we introduce a novel method, Preference Hijacking (Phi), for manipulating the MLLM response preferences using a preference hijacked image. Our method works at inference time and requires no model modifications. Additionally, we introduce a universal hijacking perturbation -- a transferable component that can be embedded into different images to hijack MLLM responses toward any attacker-specified preferences. Experimental results across various tasks demonstrate the effectiveness of our approach.

This repository contains the implementations of **Multimodal Tasks (Phi in this repo)** and **Universal Perturbation Tasks**.

<div align="center">
  <img src="assets/case_study_city.png" alt="Phi Attack Case Study" width="80%">
  <br>
  <em>Figure 1: Example of Phi on a city image. The Phi image hijacks the model's response toward a specific preference while maintaining visual similarity to the original.</em>
</div>

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{
lan2025phi,
title={Phi: Preference Hijacking in Multi-modal Large Language Models at Inference Time},
author={Yifan Lan and Yuanpu Cao and Weitong Zhang and Lu Lin and Jinghui Chen},
booktitle={The 2025 Conference on Empirical Methods in Natural Language Processing},
year={2025},
url={https://openreview.net/forum?id=yTszIbZM9C}
}
```


## 🚀 Quick Start

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Yifan-Lan/Phi.git
cd Phi
```

2. **Create environment and install dependencies**

```bash
pip3 install -r requirements.txt
```
### Authentication Setup

Before running the scripts, set up HuggingFace, wandb and OpenAI API.

For HuggingFace models that require authentication:
```bash
huggingface-cli login
```

For WandB experiment tracking:
```bash
wandb login
```
Optional: For OpenAI API (for GPT evaluation):
```bash
export OPENAI_API_KEY="your_openai_api_key_here"
```


## 📁 Project Structure

```
Phi/
├── examples/scripts/                        # Main training and evaluation scripts
│   ├── eval_phi.py                          # Evaluation of Phi
│   ├── eval_universal_perturbation.py       # Evaluation of universal border or patch
│   ├── train_universal_border.py            # Universal hijacking border training
│   ├── train_universal_patch.py             # Universal hijacking patch training
│   └── train_phi.py                         # Phi image training
├── data/                                    # Dataset directory
│   ├── city/                                # Phi training datasets
│   ├── landscape/                           # Universal perturbation datasets
│   └── ...
├── clean image/                             # Original images
├── output/                                  # Generated Phi images and evaluation logs
├── trl/                                     # Modified TRL library
├── GPT_test_score_phi.py                    # GPT evaluation of Phi
├── GPT_test_score_universal.py              # GPT evaluation of universal perturbation
├── requirements.txt                         # Python dependencies
└── README.md                                # This file
```

## 📊 Dataset Preparation

### Supported Datasets

The framework supports various dataset types for different attack scenarios:

| **Task Type** | **Datasets** | **Description** |
|----------------|--------------|-----------------|
| **Multimodal Tasks** | `city`, `pizza`, `person`, `tech_nature`, `war_peace`, `power_humility` | Image-specific preference hijacking |
| **Universal Perturbation Tasks** | `food`, `landscape`, `people` (+ `_test` variants) | Transferable hijacking perturbations across images |


### Dataset Structure

Organize your datasets as follows:
```
data/
├── city/            
│   ├── train.csv    # Training dataset for multimodal tasks (Phi)
│   └── test.csv     # Test dataset for multimodal tasks (Phi)
├── landscape/       
│   ├── images/
│   └── train.csv    # Training dataset for universal hijacking perturbations
├── landscape_test/  
│   ├── images/
│   └── test.csv     # Test dataset for universal hijacking perturbations
└── ...
```

### CSV Format

Each CSV file should contain the following columns:
- `file_path` (only in datasets for universal hijacking perturbations): Relative path to the image file (e.g., "images/tech/image001.jpg")
- `question`: Question for the image
- `matching`: Expected or target response for preference hijacking
- `not_matching`: Original clean response 

## 🔧 Usage

### Training

#### 1. Phi Training (for Multimodal Task)

Train a preference-hijacked image for a specific dataset:

```bash
accelerate launch --gpu_ids 0 examples/scripts/train_phi.py \
    --ds_type city \
    --template_img clean_image/cityview.png
```

**Key Parameters:**
- `--ds_type`: Training dataset
  - **Options**: `city`, `pizza`, `person`, `tech_nature`, `war_peace`, `power_humility`
- `--template_img`: Path to the clean image to be hijacked
  - **Format**: `.png`, `.jpg`, `.bmp`

#### 2. Universal Border Perturbation Training

Train a universal border perturbation that can be applied to different images:

```bash
accelerate launch --gpu_ids 0 examples/scripts/train_universal_border.py \
    --ds_type people \
    --border_size 252
```

**Key Parameters:**
- `--ds_type`: Training dataset
  - **Options**: `food`, `landscape`, `people`
- `--border_size`: Border inner size in pixels
  - **Common values**: `196`, `252` (used in the paper), `300`,...

#### 3. Universal Patch Perturbation Training

Train a universal patch perturbation:

```bash
accelerate launch --gpu_ids 0 examples/scripts/train_universal_patch.py \
    --ds_type landscape \
    --patch_size 168
```

**Key Parameters:**
- `--ds_type`: Training dataset
  - **Options**: `food`, `landscape`, `people`
- `--patch_size`: Patch size in pixels
  - **Common values**: `140`, `168` (used in the paper), `196`,...

> **💡 Tip**: Generated hijacked images are saved to `output/` directory

### Evaluation

#### 1. Phi Evaluation

Evaluate the effectiveness of a trained Phi image:

```bash
accelerate launch --gpu_ids 0 examples/scripts/eval_phi.py \
    --ds_type city \
    --phi_img_path <path_of_Phi_image>
```

**Key Parameters:**
- `--ds_type`: Must match the training dataset
- `--phi_img_path`: Path to generated Phi image (`.bmp` file)

#### 2. Universal Perturbation Evaluation

Evaluate universal border perturbation:
```bash
accelerate launch --gpu_ids 0 examples/scripts/eval_universal_perturbation.py \
    --ds_type landscape_test \
    --p_type border \
    --border_size 252 \
    --p_path <path_of_universal_hijacking_perturbation>
```

Evaluate universal patch perturbation:
```bash
accelerate launch --gpu_ids 0 examples/scripts/eval_universal_perturbation.py \
    --ds_type landscape_test \
    --p_type patch \
    --patch_size 168 \
    --p_path <path_of_universal_hijacking_perturbation>
```

**Key Parameters:**
- `--ds_type`: Test dataset
  - **Options**: `landscape_test`, `food_test`, `people_test`
- `--p_type`: Perturbation type
  - **Options**: `border`, `patch`
- `--border_size`/`--patch_size`: Must match training configuration
- `--p_path`: Path to generated perturbation file (`.bmp` file)

> **⚠️ Important**: Use test datasets (`*_test`) for evaluation to ensure proper train/test split

### GPT-based Evaluation

#### 1. Phi GPT Evaluation

Evaluate Phi attack results using GPT-4o scoring:

```bash
python GPT_test_score_phi.py \
    --ds_type city \
    --response_type phi \
    --log_file <path_of_evaluation_log>
```

**Key Parameters:**
- `--ds_type`: Dataset used for evaluation
  - **Options**: `city`, `pizza`, `person`, `tech_nature`, `war_peace`, `power_humility`
- `--response_type`: Response source
  - **Options**: `phi` (hijacked responses), `clean` (original responses)
- `--log_file`: Evaluation log from previous step

#### 2. Universal Perturbation GPT Evaluation

Evaluate universal perturbation attack results:

```bash
python GPT_test_score_universal.py \
    --ds_type people_test \
    --response_type phi \
    --log_file <path_of_evaluation_log>
```

**Key Parameters:**
- `--ds_type`: Test dataset
  - **Options**: `food_test`, `landscape_test`, `people_test`
- `--response_type**: Response source
  - **Options**: `phi` (hijacked), `clean` (original)
- `--log_file`: Evaluation log from universal perturbation evaluation 

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

