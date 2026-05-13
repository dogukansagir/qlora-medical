import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, MistralForCausalLM
from lora import LoRALayer, inject_lora_layers
import gradio as gr
from datasets import load_dataset
from helper_functions import tokenize, format_prompt
from torch.utils.data import DataLoader
import wandb

# 4-bit config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    quantization_config=bnb_config,
    device_map="auto"
)

device = next(model.parameters()).device
print(f"Model is on device: {device}")

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
tokenizer.pad_token = tokenizer.eos_token

# Freeze all model parameters first.
for param in model.parameters():
        param.requires_grad = False

# Then inject LoRA layers into the model
target_modules = ["q_proj", "v_proj"]
model = inject_lora_layers(model, rank=4, alpha=8, dropout=0.05, target_modules=target_modules)
model = model.to(device)

# Load dataset
dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train")

# Limit to 10k samples
dataset = dataset.shuffle(seed=42).select(range(10_000))

dataset = dataset.map(format_prompt)
tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)
tokenized_dataset.set_format("torch")
train_dataloader = DataLoader(tokenized_dataset, batch_size=4, shuffle=True)

# Initialize WandB
wandb.init(
    project="medical-qlora",
    entity="dogukansagir-none",
    config={
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "dataset": "ChatDoctor-HealthCareMagic-100k",
        "samples": 10_000,
        "rank": 4,
        "alpha": 8,
        "dropout": 0.05,
        "learning_rate": 1e-4,
        "epochs": 1,
        "batch_size": 4,
        "max_length": 512,
    }
)

# Create the training loop

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=0.01)
num_epochs = 1
total_steps = len(train_dataloader) * num_epochs
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

model.gradient_checkpointing_enable()
model.train()
for epoch in range(num_epochs):
    for step, batch in enumerate(train_dataloader):
        optimizer.zero_grad()

        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if step % 10 == 0:
            print(f"Epoch {epoch} | Step {step}/{total_steps} | Loss {loss.item():.4f}")
            wandb.log({
                "loss": loss.item(),
                "learning_rate": scheduler.get_last_lr()[0],
                "epoch": epoch,
                "step": step,
            })
        # Save checkpoint every 200 steps
        if step % 200 == 0 and step > 0:
            checkpoint_path = f"checkpoint_step_{step}.pt"
            trainable_state = {k: v for k, v in model.state_dict().items() if v.requires_grad}
            torch.save({
                "step": step,
                "model_state_dict": trainable_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss.item(),
            }, checkpoint_path)
            print(f"Checkpoint saved → {checkpoint_path}")

print("Training complete!")
wandb.finish()

# Save final LoRA adapter weights only
lora_state_dict = {
    k: v for k, v in model.state_dict().items() if "lora_" in k
}
torch.save(lora_state_dict, "lora_adapter.pt")
print("LoRA adapter saved → lora_adapter.pt")

# Load adapter and run demo
model.eval()

def chat(message):
    prompt = f"<s>[INST] You are a helpful medical assistant. Never make a diagnosis. Only suggest possible explanations and always recommend consulting a real doctor. {message} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            do_sample=True,
            repetition_penalty=1.1,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("[/INST]")[-1].strip()

gr.Interface(
    fn=chat,
    inputs=gr.Textbox(label="Your medical question"),
    outputs=gr.Textbox(label="Medical Assistant"),
    title="Medical Assistant — QLoRA fine-tuned Mistral 7B",
    description="Fine-tuned on ChatDoctor-HealthCareMagic-100k dataset"
).launch()