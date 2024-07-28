import random
import pandas as pd
from datasets import load_dataset, load_from_disk, ClassLabel
from itertools import chain
import torch

from transformers import (BertConfig, 
                          GPT2Config, 
                          LlamaConfig,
                          LlamaForCausalLM,
                          BertForSequenceClassification, 
                          GPT2LMHeadModel, 
                          Trainer, 
                          TrainingArguments, 
                          DataCollatorForLanguageModeling, 
                          DataCollatorWithPadding,
                          CONFIG_MAPPING,
                          MODEL_MAPPING,
                          AutoConfig,
                          AutoModelForCausalLM,
                          AutoTokenizer,
                          SchedulerType,
                          default_data_collator,
                          get_scheduler,)

# datasets = load_dataset('wikitext', 'wikitext-2-raw-v1')
# datasets = load_from_disk("/mnt/raid/tangzichen/wikitext2")
# model_dir = "/mnt/raid/tangzichen/gpt2"


# print(list(datasets.keys()))

# print(datasets)

# print(datasets["train"][10].keys())

# print(datasets["train"][10]['text'])

# print(datasets["train"][1])



# print(print(len(datasets["train"]), len(datasets["test"])))

# def show_random_elements(dataset, num_examples=10):
#     picks = []
#     for _ in range(num_examples):
#         pick = random.randint(0, len(dataset)-1)
#         while pick in picks:
#             pick = random.randint(0, len(dataset)-1)
#         picks.append(pick)

#     df = pd.DataFrame(dataset[picks])
#     for column, typ in dataset.features.items():
#         if isinstance(typ, ClassLabel):
#             df[column] = df[column].transform(lambda i: typ.names[i])
# show_random_elements(datasets["train"], 10)



# tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=model_dir)
# # tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", cache_dir=self.model_dir)

# # dataset = load_from_disk(wikitext_path)
# # def tokenize(example):
# #     return tokenizer(example['text'], truncation=True, padding='max_length', max_length=512)

# # # Apply the encoding to the dataset
# # encoded_dataset = dataset.map(tokenize, batched=True)
# # encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# column_names = datasets["train"].column_names
# text_column_name = "text" if "text" in column_names else column_names[0]
# def tokenize_function(examples):
#     return tokenizer(examples[text_column_name])

# tokenized_dataset = datasets.map(tokenize_function, batched=True, remove_columns=column_names)


# print(f"tokenized_dataset: {tokenized_dataset}")

# trainset = tokenized_dataset['train']
# input_ids = trainset["input_ids"][10]
# attention_mask = trainset["attention_mask"][10]
# print(f"tokenized_dataset -- input_ids: {input_ids}")
# print(f"tokenized_dataset -- attention_mask: {attention_mask}")

# origin_tokens = tokenizer.convert_ids_to_tokens(input_ids)
# print(f"tokenized_dataset -- origin_tokens: {origin_tokens}")


# block_size = min(1024, tokenizer.model_max_length)

# def group_texts(examples):
#     # Concatenate all texts.
#     concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
#     total_length = len(concatenated_examples[list(examples.keys())[0]])
#     # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
#     # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
#     total_length = (total_length // block_size) * block_size
#     # Split by chunks of max_len.
#     result = {
#         k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
#         for k, t in concatenated_examples.items()
#     }
#     result["labels"] = result["input_ids"].copy()
#     return result

# encoded_dataset = tokenized_dataset.map(group_texts, batched=True)
# print(f"encoded_dataset: {encoded_dataset}")

# trainset = encoded_dataset['train']
# valset = encoded_dataset['validation']


# input_ids = trainset["input_ids"][10]
# attention_mask = trainset["attention_mask"][10]
# print(f"encoded_dataset -- input_ids: {input_ids}")
# print(f"encoded_dataset -- attention_mask: {attention_mask}")


# origin_tokens = tokenizer.convert_ids_to_tokens(input_ids)
# print(f"encoded_dataset -- origin_tokens: {origin_tokens}")




from transformers import LlamaTokenizerFast


# dnn = "gpt2"
# model_dir = "/mnt/raid/tangzichen/gpt2"
# config = GPT2Config.from_pretrained(dnn, cache_dir=model_dir)
# net = AutoModelForCausalLM.from_pretrained(
#     pretrained_model_name_or_path=dnn,
#     cache_dir=model_dir,
#     from_tf=False, 
#     config=config,
#     low_cpu_mem_usage=True, 
#     trust_remote_code=False
# )
# net = AutoModelForCausalLM.from_config(config)




# dnn = "bert-base-uncased"
# model_dir = "/mnt/raid/tangzichen/bert-base-uncased"
# config = BertConfig.from_pretrained(dnn, cache_dir=model_dir)
# net = AutoModelForCausalLM.from_pretrained(
#     pretrained_model_name_or_path=dnn,
#     cache_dir=model_dir,
#     from_tf=False, 
#     config=config,
#     low_cpu_mem_usage=True, 
#     trust_remote_code=False
# )
# net = AutoModelForCausalLM.from_config(config)




#   warnings.warn(
# LlamaConfig {
#   "_name_or_path": "meta-llama/Llama-2-7b-hf",
#   "architectures": [
#     "LlamaForCausalLM"
#   ],
#   "bos_token_id": 1,
#   "eos_token_id": 2,
#   "hidden_act": "silu",
#   "hidden_size": 4096,
#   "initializer_range": 0.02,
#   "intermediate_size": 11008,
#   "max_position_embeddings": 4096,
#   "model_type": "llama",
#   "num_attention_heads": 32,
#   "num_hidden_layers": 32,
#   "num_key_value_heads": 32,
#   "pretraining_tp": 1,
#   "rms_norm_eps": 1e-05,
#   "rope_scaling": null,
#   "tie_word_embeddings": false,
#   "torch_dtype": "float16",
#   "transformers_version": "4.32.1",
#   "use_cache": true,
#   "vocab_size": 32000
# }


dnn = "llama2-124M"
model_dir = "/data2/share/llama-1/llama-2-7b-hf"

# model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir=model_dir)
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir=model_dir)
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir=model_dir)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/llama-2-7b-hf", cache_dir=model_dir)
# tokenizer = AutoTokenizer.from_pretrained(model_dir)
# tokenizer = LlamaTokenizerFast.from_pretrained(model_dir)


config = LlamaConfig.from_pretrained("meta-llama/llama-2-7b-hf", cache_dir=model_dir)
print(config)

# config["max_position_embeddings"] = 764
# config["num_hidden_layers"] = 8
# config["hidden_size"] = 512
# config["num_attention_heads"] = 8
# config["num_key_value_heads"] = 8


# config.max_position_embeddings = 764
# config.num_hidden_layers = 8
# config.hidden_size = 512
# config.num_attention_heads = 8
# config.num_key_value_heads = 8


config.max_position_embeddings = 32
config.num_hidden_layers = 2
config.hidden_size = 32
config.num_attention_heads = 2
config.num_key_value_heads = 2


# LlamaConfig {
#   "_name_or_path": "meta-llama/Llama-2-7b-hf",
#   "architectures": [
#     "LlamaForCausalLM"
#   ],
#   "bos_token_id": 1,
#   "eos_token_id": 2,
#   "hidden_act": "silu",
#   "hidden_size": 512,
#   "initializer_range": 0.02,
#   "intermediate_size": 11008,
#   "max_position_embeddings": 764,
#   "model_type": "llama",
#   "num_attention_heads": 8,
#   "num_hidden_layers": 8,
#   "num_key_value_heads": 8,
#   "pretraining_tp": 1,
#   "rms_norm_eps": 1e-05,
#   "rope_scaling": null,
#   "tie_word_embeddings": false,
#   "torch_dtype": "float16",
#   "transformers_version": "4.32.1",
#   "use_cache": true,
#   "vocab_size": 32000
# }


net = AutoModelForCausalLM.from_config(config)
print(config)
# print(net)
# for name, parameter in net.named_parameters():
#     print(f"name: {name}, number params: {parameter.size()}")
# for name, module in net.named_modules():
#     print(f"name: {name}, module: {module}")


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num, "Total-M": total_num/1000000}

# 
number_params = get_parameter_number(net)

from torch.nn import Module

print(f"get_parameter_number: {number_params}")

print(f"type(net): {type(net)}, isinstance(net, torch.Module):{isinstance(net, Module)}")
print(f"net: {net}")

print(f"type(net.model): {type(net.model)}, isinstance(net.model, torch.Module):{isinstance(net.model, Module)}")

print(f"type(net.model.layers[0]):{type(net.model.layers[0])}, ")

print(f"net.model.layers[0]:{net.model.layers[0]}")

print(f"net.model.layers[0].mlp:{net.model.layers[0].mlp}")

print(f"====================================")
print(f"net.model.layers[0].mlp.gate_proj.weight[0,:10]:{net.model.layers[0].mlp.gate_proj.weight[0,:10]}")
print(f"====================================")

from copy import deepcopy
# new_net = net.state_dict()
new_net = deepcopy(net)
def modify_net(net):
    for name, param in net.named_parameters():
        shape = param.data.shape
        param.data = param.data + torch.normal(mean=1.0, std=0.001, size=shape, device=param.data.device)
modify_net(new_net)
print(f"net.model.layers[0].mlp.gate_proj.weight[0,:10]:{net.model.layers[0].mlp.gate_proj.weight[0,:10]}")

net.load_state_dict(new_net.state_dict())
print(f"net.model.layers[0].mlp.gate_proj.weight[0,:10]:{net.model.layers[0].mlp.gate_proj.weight[0,:10]}")




# huggingface-cli login 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
# HF_ENDPOINT=https://hf-mirror.com python


























