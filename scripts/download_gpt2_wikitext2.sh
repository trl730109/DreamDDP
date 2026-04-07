#!/bin/bash
# Download GPT-2 model and WikiText-2 dataset.
# Edit the paths below to match your env.sh data/model paths.

MODEL_DIR="${1:-/workspace/models/gpt2}"
DATA_DIR="${2:-/workspace/wikitext2}"
PY="${PY:-python3}"

echo "Model -> $MODEL_DIR"
echo "Data  -> $DATA_DIR"
mkdir -p "$MODEL_DIR" "$DATA_DIR"

# Download GPT-2 model weights and tokenizer
$PY - <<EOF
from transformers import AutoModelForCausalLM, AutoTokenizer
model_dir = "$MODEL_DIR"
print("Downloading GPT-2...")
AutoTokenizer.from_pretrained("openai-community/gpt2", cache_dir=model_dir).save_pretrained(model_dir)
AutoModelForCausalLM.from_pretrained("openai-community/gpt2", cache_dir=model_dir).save_pretrained(model_dir)
print("GPT-2 saved to", model_dir)
EOF

# Download and tokenize WikiText-2, save to disk (load_from_disk format)
$PY - <<EOF
from datasets import load_dataset
from transformers import AutoTokenizer

data_dir = "$DATA_DIR"
model_dir = "$MODEL_DIR"

print("Downloading WikiText-2...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=1024, padding="max_length")

encoded = dataset.map(tokenize, batched=True, remove_columns=["text"])
encoded.save_to_disk(data_dir)
print("WikiText-2 saved to", data_dir)
EOF

echo "Done."
