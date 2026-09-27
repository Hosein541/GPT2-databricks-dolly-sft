فایل `README.md` جامع و حرفه‌ای برای این پروژه با ساختار زیر آماده شده است. با توجه به بررسی دقیق نوت‌بوک، مدل انتخابی شما **GPT-2 Medium (نسخه ۳۵۵ میلیون پارامتری)** بوده و آموزش در ۲ اپوک طی ۳۶ دقیقه انجام گرفته است. تمامی فایل‌های جانبی شامل `utils.py`، `gpt_download.py`، تصویر تابع زیان و فایل نتایج تست (`instruction-data-with-response.json`) در ساختار و توضیحات گنجانده شده‌اند:

---

```markdown
# 🧠 Instruction Fine-Tuning GPT-2 Medium (355M) on Databricks Dolly-15k

Supervised Fine-Tuning (SFT) of a pretrained **GPT-2 Medium (355M)** model on a balanced instruction-following dataset from **Databricks Dolly-15k** using pure **PyTorch** (inspired by Chapter 7 of Sebastian Raschka's *Build a Large Language Model from Scratch*).

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

```

---

## 📌 Project Overview

Pretrained base language models are trained as next-token predictors and often fail to follow user prompts directly without conversational alignment. This project adapts the **GPT-2 Medium (355M parameters)** architecture into an instruction-following assistant.

### 🛠️ Architecture & Pipeline Highlights

* **Base Architecture**: GPT-2 Medium (1024 embedding dimension, 24 transformer layers, 16 attention heads, 1024 context length).
* **Weight Loading**: Downloaded and initialized from OpenAI's original checkpoint weights via custom loading scripts (`gpt_download.py` & `utils.py`).
* **Prompt Formatting**: Uses standard Alpaca-style instruction templating supporting both direct tasks and context-grounded queries (`Instruction` + optional `Input` -> `Response`).
* **Dynamic Masking Collator**: A custom batch collation function (`custom_collate_fn`) that dynamically pads sequences up to 1024 tokens and masks redundant padding/prompt tokens with `ignore_index = -100` so loss is computed strictly over target completions.
* **Evaluation Artifacts**: Inference predictions generated and saved in JSON format for qualitative analysis across test samples.

---

## 📊 Dataset & Preprocessing

The model was trained on a balanced subset of [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k?utm_source=gemini):

* **Selected Categories**: `open_qa`, `general_qa`, `classification`, and `closed_qa`.
* **Balancing**: Downsampled to the minority category (`closed_qa` with 1,773 instances), yielding **5,319 total curated pairs**.
* **Data Splits**:
* **Train**: 85% (4,521 samples)
* **Validation**: 5% (267 samples)
* **Test**: 10% (531 samples)



### Prompt Structure

```text
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{Instruction}

### Input:
{Input (Optional Context)}

### Response:
{Output}

```

---

## ⚙️ Hyperparameters & Training Setup

| Parameter | Value | Details |
| --- | --- | --- |
| **Model Size** | GPT-2 Medium (355M) | 24 Layers, 16 Heads, `emb_dim=1024` |
| **Tokenizer** | BPE (`tiktoken` GPT-2) | Vocab size: 50,257 |
| **Context Length** | 1024 tokens | Dynamic sequence batching |
| **Batch Size** | 1 | Sequence-level SGD updates |
| **Optimizer** | `AdamW` | Weight decay = 0.075 |
| **Learning Rate** | `7.5e-5` | Stable constant schedule for SFT |
| **Epochs** | 2 | ~1,130 optimization steps |
| **Compute Time** | 36.09 minutes | Single GPU (Tesla T4) |

---

## 📈 Training Dynamics & Loss Curve

The training cross-entropy loss dropped steadily from **~3.18** down to below **0.80**, while the validation loss converged and stabilized around **1.77**:

* **Initial Loss**: Train Loss `3.176` | Val Loss `2.749`
* **Mid Training (Step 565)**: Train Loss `1.684` | Val Loss `1.775` (Saved Best Checkpoint)
* **Final Convergence (Step 1130)**: Train Loss `0.995` | Val Loss `1.779`

---

## 🧪 Qualitative Test Evaluation

Predictions across the held-out test split are exported to `instruction-data-with-response.json`.

### Example 1: Extractive Q&A

```text
Instruction: What does the acronym IMET stand for?
Input: International Military Education and Training (IMET) is the title of a United States security assistance program...
Ground Truth: International Military Education and Training
Model Response: IMET stands for International Military Education and Training

```

### Analysis of Limitations

For extractive and short Q&A tasks, the 355M model captures the correct information cleanly. However, for open-ended or highly abstract instructions, smaller base models may exhibit repetition or lack depth. Scaling to higher-capacity models or running multi-turn DPO/RLHF further refines output coherence.

---

## 🚀 Quickstart & Inference

### 1. Installation

```bash
git clone https://github.com/Hosein541/GPT2-databricks-dolly-sft.git
cd GPT2-databricks-dolly-sft

pip install -r requirements.txt

```


---

## 📂 Repository Structure

```text
├── assets/
│   └── loss_curve.png                       # Training and validation loss plot
├── gpt_download.py                          # Utility script to download OpenAI GPT-2 weights
├── utils.py                                 # Transformer architecture, data loaders & training loops
├── instruction-data-with-response.json      # Model generation outputs on test dataset
├── gpt2_medium_instruction_tuning.ipynb     # Interactive Jupyter notebook
├── requirements.txt                         # Pinned dependencies
└── README.md                                # Documentation

```

---

## 📜 Acknowledgements

* **Sebastian Raschka** for the foundational architecture design in *Build a Large Language Model from Scratch*.
* **Databricks** for the open-source Dolly-15k dataset.
* **OpenAI** for making GPT-2 base weights openly accessible.

---
