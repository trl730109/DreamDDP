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

# Download WikiText-2 and save raw text to disk (load_from_disk format)
# NOTE: do NOT pre-tokenize — llm_trainer.py loads raw text and tokenizes at runtime
$PY - <<EOF
from datasets import load_dataset

data_dir = "$DATA_DIR"
print("Downloading WikiText-2...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
dataset.save_to_disk(data_dir)
print("WikiText-2 saved to", data_dir)
EOF

echo "Done."
