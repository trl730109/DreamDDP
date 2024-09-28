import torch
import torch.nn as nn
import torch.nn.functional as F

class GPT2Model(nn.Module):
    def __init__(self, vocab_size, n_embd, n_layer, n_head, max_position_embeddings=1024):
        super(GPT2Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(max_position_embeddings, n_embd)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=n_embd, nhead=n_head) for _ in range(n_layer)
        ])
        attention_mask = torch.triu(torch.ones((max_position_embeddings, max_position_embeddings)), diagonal=1)
        self.causal_mask = attention_mask
        # self.causal_mask = attention_mask.masked_fill(attention_mask == 1, float('-inf')).masked_fill(attention_mask == 0, float(0.0))        
        print(f"causal_mask:, shape:{self.causal_mask.shape}, {self.causal_mask}")

        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, is_causal=True):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        word_embeddings = self.embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        embeddings = word_embeddings + position_embeddings

        if attention_mask is None:
            attention_mask = self.causal_mask[:seq_length, :seq_length]
            attention_mask = attention_mask.unsqueeze(0).expand(input_ids.size(0), -1, -1)
            print(f"attention_mask:, shape:{attention_mask.shape}, {attention_mask}")

        hidden_states = embeddings
        for layer in self.layers:
            hidden_states = layer(hidden_states, hidden_states,  
            tgt_mask=attention_mask,
            tgt_is_causal=is_causal,
            tgt_key_padding_mask=attention_mask,
            memory_key_padding_mask=None,
            memory_mask=attention_mask,
            memory_is_causal=is_causal)

        logits = self.lm_head(hidden_states)
        return logits

    def generate(self, input_ids, max_length, pad_token_id, eos_token_id):
        for _ in range(max_length):
            attention_mask = (input_ids != pad_token_id).long()
            logits = self.forward(input_ids, attention_mask=attention_mask)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if next_token.item() == eos_token_id:
                break
        return input_ids

def calculate_loss(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss

if __name__ == "__main__":
    # Example usage
    from transformers import GPT2Tokenizer

    # # Initialize tokenizer and model
    # tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2')
    # texts = ["Hello, how are you?", "I am fine, thank you!"]
    # tokenizer.pad_token = tokenizer.eos_token
    # # Tokenize input texts
    # inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)

    # # input_ids shape: (batch_size, sequence_length)
    # input_ids = inputs['input_ids']
    # print(f"input_ids:, shape:{input_ids.shape}, {input_ids}")

    vocab_size = 50257
    n_embd = 768
    n_layer = 12
    n_head = 12
    max_position_embeddings = 1024
    pad_token_id = 0
    eos_token_id = 50256

    model = GPT2Model(vocab_size, n_embd, n_layer, n_head, max_position_embeddings)
    # input_ids = torch.randint(0, vocab_size, (1, max_position_embeddings))  # Example input
    input_ids = torch.randint(0, vocab_size, (8, max_position_embeddings))  # Example input
    labels = input_ids.clone()  # For auto-regressive language modeling, labels are the same as input_ids
    print(f"input_ids:, shape:{input_ids.shape}, {input_ids}")


    # attention_mask = torch.triu(torch.ones((max_position_embeddings, max_position_embeddings), device=input_ids.device), diagonal=1)
    # attention_mask = attention_mask.masked_fill(attention_mask == 1, float('-inf')).masked_fill(attention_mask == 0, float(0.0))
    # print(f"IN OUT MAIN attention_mask:, shape:{attention_mask.shape}, {attention_mask}")

    logits = model(input_ids)
    loss = calculate_loss(logits, labels)

    print(f"Loss: {loss.item()}")

    # Inference example
    generated_ids = model.generate(input_ids[:, :1], max_length=10, pad_token_id=pad_token_id, eos_token_id=eos_token_id)
    print(f"Generated IDs: {generated_ids}")













