tokenizer = None

def get_tokenizer():
    global tokenizer
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def format_prompt(sample):
    return {
        "text": f"<s>[INST] {sample['input']} [/INST] {sample['output']}</s>"
    }

def tokenize(sample):
    tokenizer = get_tokenizer()
    result = tokenizer(
        sample["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    # Create labels: copy input_ids, then mask padding tokens with -100
    labels = result["input_ids"].copy()
    labels = [-100 if token == tokenizer.pad_token_id else token for token in labels]
    result["labels"] = labels
    return result