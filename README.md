# Medical Assistant – QLoRA Fine-tuned Mistral-7B

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

Fine-tuning `mistralai/Mistral-7B-Instruct-v0.3` on 10k doctor-patient dialogues using **QLoRA** (4-bit quantisation + Low-Rank Adaptation) — with a manually implemented LoRA adapter and custom PyTorch training loop.

> **⚠️ Medical Disclaimer**
> This model is for research and demonstration purposes only. It is **not** a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified physician.

---

## Features

- 🔧 **Manual LoRA Implementation** — `LoRALayer` and injection written from scratch without `peft` library
- 📝 **Custom Training Loop** — full PyTorch loop without `SFTTrainer`, including gradient clipping and cosine scheduling
- 🧠 **4-bit Quantisation** — fits Mistral-7B on a single consumer GPU (~8GB VRAM) via NF4 quantisation
- 🩺 **Medical Domain** — fine-tuned on 10k doctor-patient dialogues from ChatDoctor-HealthCareMagic-100k
- 📊 **Experiment Tracking** — Weights & Biases integration with loss curves and hyperparameter logging
- 💾 **Lightweight Adapter** — only ~14MB for trained LoRA weights, base model unchanged

---

## Architecture

```
Input
  ↓
Mistral-7B (frozen, 4-bit NF4)
  ↓
[q_proj]  →  original W (frozen)  ──┐
          →  A @ B (trainable)    ──┴→ sum → output
[v_proj]  →  same pattern
  ↓
Output

Trainable: ~3.4M params (0.091% of 7.2B total)
```

LoRA injects low-rank matrices alongside frozen attention projections — only `lora_A` and `lora_B` receive gradients.

---

## Example Output

```
User:  "I've been having chest pain and shortness of breath for two days."

Base Mistral-7B:
"I am not a doctor, but I can help you understand what symptoms you are experiencing and suggest when to seek professional medical advice.

The symptoms you described - chest pain and shortness of breath - could be indicative of various conditions, some of which may be serious. These symptoms can be associated with heart disease, lung conditions like pneumonia or asthma, or other health issues.

If you have any known heart conditions, it is crucial to consult your healthcare provider immediately. Other factors that might require urgent attention include:

1. The pain in your chest is severe, persistent, or worsening over time.
2. You experience chest pain accompanied by other symptoms such as sweating, nausea, dizziness, or cold sweat.
3. Your shortness of breath is severe, making it difficult to carry out daily activities.
4. You have a history of smoking or exposure to secondhand smoke.
5. You have a family history of heart disease or other chronic illnesses.
6. You have recently experienced significant emotional stress or trauma.
7. You have a weakened immune system due to illness or medication.
8. You have a high fever (above 100.4°F or 38°C).
9. You notice swelling in your legs, ankles, or feet.
10. You experience sudden weight gain or loss.

If you are unsure about your condition or if your symptoms persist, it is essential to contact your healthcare provider or seek emergency medical attention. In case of severe symptoms or if you suspect a heart attack, call emergency services immediately."

Fine-tuned Model:
"Hi, Thanks for writing in. Chest pain and shortness of breath can be due to many reasons. It could be due to heart disease or lung disease.
 You need to get an ECG done to rule out heart disease. Also, you need to get a chest x-ray done to rule out lung disease.
 If these tests are normal then you may have anxiety or panic disorder. You should consult a psychiatrist for this. Hope this helps."
```

---

## How It Works

| Component | Description |
|-----------|-------------|
| **Base Model** | Mistral-7B loaded with 4-bit NF4 quantisation. All parameters frozen. |
| **LoRA Injection** | `LoRALayer` written manually — wraps `nn.Linear` with parallel `A @ B` branch. Injected into `q_proj` and `v_proj` via `inject_lora()`. |
| **Why B=0 at init** | Ensures LoRA contributes nothing at start — model behaves identically to base until first gradient update. |
| **Scaling** | `alpha/rank` compensates for rank size — keeps LoRA contribution consistent across ablation runs. |
| **Training Loop** | Custom PyTorch loop with AdamW, cosine LR schedule, gradient clipping, and checkpoint saving every 200 steps. |
| **Inference** | LoRA structure re-injected, adapter loaded with `strict=False` — base model weights untouched. |

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | mistralai/Mistral-7B-Instruct-v0.3 |
| Dataset | ChatDoctor-HealthCareMagic-100k (10k samples) |
| Rank | 4 |
| Alpha | 8 |
| Dropout | 0.05 |
| Target modules | q_proj, v_proj |
| Learning rate | 1e-4 |
| Epochs | 1 |
| Batch size | 4 |
| Max sequence length | 512 |
| Quantisation | 4-bit NF4 + double quantisation |

---

## Performance

| Metric | Value |
|--------|-------|
| Trainable parameters | ~3.4M / 7.2B (0.091%) |
| VRAM during training | ~8GB |
| VRAM during inference | ~6GB |
| Adapter file size | ~14MB |
| Training time | ~150 min on RTX 3060 Ti |
| Final training loss | 1.032 |

---

## Installation

```bash
git clone https://github.com/dogukansagir/qlora-medical.git
cd qlora-medical
pip install -r requirements.txt
```

### Run inference

```bash
python inference.py
# Gradio UI at http://127.0.0.1:7860
```

### Train your own adapter (optional)

```bash
python main.py
```

---

## Project Structure

```
qlora-medical/
├── lora.py         ← LoRALayer + inject_lora (manual implementation)
├── main.py         ← training script
├── inference.py    ← Gradio chat UI
├── data.py         ← dataset loading + prompt formatting
├── notebook.ipynb  ← all parts in one file
└── requirements.txt
```

---

## Acknowledgements

- [Mistral AI](https://mistral.ai/) for the base model
- [ChatDoctor dataset](https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k) by lavita
- [LoRA paper](https://arxiv.org/abs/2106.09685): Hu et al., 2021
- [QLoRA paper](https://arxiv.org/abs/2305.14314): Dettmers et al., 2023

---

## License

MIT License — see [LICENSE](LICENSE) for details.
