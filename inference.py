import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from lora import inject_lora_layers
import gradio as gr

# Same 4‑bit Quantization Config as training
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load base model (without LoRA)
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    quantization_config=bnb_config,
    device_map="auto"
)

# Freeze all parameters (same as training)
for param in model.parameters():
    param.requires_grad = False

# Inject LoRA layers with the same hyperparameters as training
target_modules = ["q_proj", "v_proj"]
model = inject_lora_layers(model, rank=4, alpha=8, dropout=0.05, target_modules=target_modules)

# Load the saved LoRA weights
lora_state_dict = torch.load("lora_adapter.pt")
model.load_state_dict(lora_state_dict, strict=False)  # strict=False because base layers are missing

# Move the whole model to the correct device (if needed)
device = next(model.parameters()).device
model = model.to(device)
model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
tokenizer.pad_token = tokenizer.eos_token

def chat(message):
    device = next(model.parameters()).device
    prompt = f"<s>[INST] If you are a doctor, please answer the medical questions based on the patient's description. {message} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.2,
            do_sample=False,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,      # stop at </s>
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Decode the whole sequence
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
    # Find the last occurrence of [/INST] and take everything after it
    prompt_end = full_output.rfind("[/INST]") + len("[/INST]")
    answer = full_output[prompt_end:].strip()
    # Remove any trailing </s> if present
    answer = answer.replace("</s>", "").strip()
    return answer

gr.Interface(
    fn=chat,
    inputs=gr.Textbox(label="Your medical question"),
    outputs=gr.Textbox(label="Medical Assistant"),
    title="Medical Assistant — QLoRA fine-tuned Mistral 7B",
    description="Fine-tuned on ChatDoctor-HealthCareMagic-100k dataset"
).launch()