# Medical Assistant – QLoRA Fine‑tuned Mistral‑7B

[![Python 3.12+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

This repository contains code and a **LoRA adapter** for fine‑tuning `mistralai/Mistral-7B-Instruct-v0.3` on the [ChatDoctor‑HealthCareMagic‑100k](https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k) dataset.

The model is trained with **QLoRA** (4‑bit quantisation + Low‑Rank Adaptation) to produce a helpful medical assistant that **never makes a diagnosis** – it only suggests possible explanations and always recommends consulting a real doctor.

> **⚠️ Medical disclaimer**  
> This model is for research and demonstration purposes only. It is **not** a substitute for professional medical advice. Always seek the advice of a qualified physician.

## Features

- 🧠 **4-bit Quantisation** – Fits Mistral-7B on a single GPU with ~8 GB VRAM for inference
- 🔧 **LoRA Fine-tuning** – Only ~4.2M trainable parameters (~0.1% of the base model)
- 🩺 **Medical Domain** – Fine-tuned on 10k doctor-patient dialogues
- 🚀 **Easy Inference** – Gradio web UI for interactive chat
- 📊 **Experiment Tracking** – Integrated Weights & Biases logging
- 💾 **Lightweight Adapter** – Only ~14 MB for the trained LoRA weights

## Requirements

Install dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```
- Python ≥ 3.12
- CUDA compatible GPU (8+ GB VRAM is recommended)

## How to Use

### 1. Clone the repository
```bash
git clone https://github.com/your-username/medical-qlora.git
cd medical-qlora
```
### 2. Install dependencies

Install dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 3. Run inference
```bash
python inference.py
```
This launches a Gradio chat UI at http://127.0.0.1:7860

### 4. Train your own adapter (optional)
```bash
python main.py
```
## How It Works

| Component | Description |
|-----------|-------------|
| **Quantised Base Model** | Mistral-7B loaded with 4-bit NF4 quantisation. All parameters frozen except LoRA. |
| **LoRA Injection** | Low-rank matrices (rank=4, alpha=8) added to q_proj and v_proj. Only ~4.2M trainable parameters. |
| **Dataset Formatting** | Each dialogue wrapped as: <s>[INST] {input} [/INST] {output}</s> |
| **Training** | Causal language modelling loss, AdamW (lr=1e-4), cosine schedule, gradient checkpointing |
| **Inference** | LoRA structure re-created, adapter loaded with strict=False |

## Training Configuration

- Model: mistralai/Mistral-7B-Instruct-v0.3
- Dataset: ChatDoctor-HealthCareMagic-100k (10,000 samples selected randomly)
- Rank: 4
- Alpha: 8
- Dropout: 0.05
- Learning rate: 1e-4
- Epochs: 1
- Batch size: 4 (Can be changed depending on VRAM size)
- Max length: 512

## Performance

| Metric | Value |
|--------|-------|
| Trainable parameters | ~3.4 million |
| Total parameters (base) | ~7.2 billion |
| VRAM during training | ~8 GB |
| VRAM during inference | ~6 GB |
| Adapter file size | ~14 MB |
| Training time | ~150 min on RTX 3060Ti 8GB |
| Final training loss | 1.03212 |

## Acknowledgements

- [Mistral AI](https://mistral.ai/) for the base model
- [ChatDoctor dataset](https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k) by lavita
- [QLoRA paper](https://arxiv.org/abs/2305.14314): Dettmers et al., 2023
- bitsandbytes and transformers by Hugging Face

**Disclaimer:** This software is provided "as is". The authors are not responsible for any misuse or for any medical decisions made based on the model's output. Always consult a qualified healthcare professional for medical advice.
