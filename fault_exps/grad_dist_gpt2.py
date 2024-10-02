import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
from datasets import load_dataset
import json

# Load dataset
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model = model.cuda()

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], return_tensors='pt', truncation=True, padding='max_length', max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
train_dataset = tokenized_datasets['train']

# DataLoader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
def train(model, train_loader, optimizer, iterations):
    model.train()
    grad_magnitudes = {10: {}, 50: {}, 100: {}}
    for i, batch in enumerate(train_loader):
        if i > iterations:
            break
        inputs = batch['input_ids'].squeeze().cuda()
        labels = batch['input_ids'].squeeze().cuda()
        optimizer.zero_grad()
        outputs = model(inputs, labels=labels)
        loss = outputs.loss
        loss.backward()

        # Obtain gradient magnitudes at specified iterations
        if i in [10, 50, 100]:
            for name, param in model.named_parameters():
                if 'h.0.attn.c_attn.weight' in name or 'h.1.attn.c_attn.weight' in name:
                    grad_magnitude = param.grad.data.norm(2).item()
                    grad_magnitudes[i][name] = grad_magnitude

        optimizer.step()

    # Save gradient magnitudes to a file
    with open('gpt2_gradients.json', 'w') as f:
        json.dump(grad_magnitudes, f)

train(model, train_loader, optimizer, iterations=100)






