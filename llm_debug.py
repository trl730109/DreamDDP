import random
import pandas as pd
from datasets import load_dataset, load_from_disk, ClassLabel
from itertools import chain


from transformers import (BertConfig, 
                          GPT2Config, 
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
datasets = load_from_disk("/mnt/raid/tangzichen/wikitext2")
model_dir = "/mnt/raid/tangzichen/gpt2"


print(list(datasets.keys()))

print(datasets)

print(datasets["train"][10].keys())

print(datasets["train"][10]['text'])

print(datasets["train"][1])



print(print(len(datasets["train"]), len(datasets["test"])))

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



tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=model_dir)
# tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", cache_dir=self.model_dir)

# dataset = load_from_disk(wikitext_path)
# def tokenize(example):
#     return tokenizer(example['text'], truncation=True, padding='max_length', max_length=512)

# # Apply the encoding to the dataset
# encoded_dataset = dataset.map(tokenize, batched=True)
# encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

column_names = datasets["train"].column_names
text_column_name = "text" if "text" in column_names else column_names[0]
def tokenize_function(examples):
    return tokenizer(examples[text_column_name])

tokenized_dataset = datasets.map(tokenize_function, batched=True, remove_columns=column_names)


print(f"tokenized_dataset: {tokenized_dataset}")

trainset = tokenized_dataset['train']
input_ids = trainset["input_ids"][10]
attention_mask = trainset["attention_mask"][10]
print(f"tokenized_dataset -- input_ids: {input_ids}")
print(f"tokenized_dataset -- attention_mask: {attention_mask}")

origin_tokens = tokenizer.convert_ids_to_tokens(input_ids)
print(f"tokenized_dataset -- origin_tokens: {origin_tokens}")


block_size = min(1024, tokenizer.model_max_length)

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

encoded_dataset = tokenized_dataset.map(group_texts, batched=True)
print(f"encoded_dataset: {encoded_dataset}")

trainset = encoded_dataset['train']
valset = encoded_dataset['validation']


input_ids = trainset["input_ids"][10]
attention_mask = trainset["attention_mask"][10]
print(f"encoded_dataset -- input_ids: {input_ids}")
print(f"encoded_dataset -- attention_mask: {attention_mask}")


origin_tokens = tokenizer.convert_ids_to_tokens(input_ids)
print(f"encoded_dataset -- origin_tokens: {origin_tokens}")






































