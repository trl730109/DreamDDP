import datasets
from datasets import load_dataset
import pandas as pd
from functools import partial

def get_dataset(dataset_name, local_data_dir=None):

    if dataset_name in ["gsm8k"]:
        # dataset_name = local_data_dir + dataset_name if local_data_dir is not None else dataset_name
        # dataset = load_dataset(dataset_name, split="train", name="main")
        dataset = load_dataset(local_data_dir, split="train", name="main")
    elif dataset_name in ["lighteval/MATH"]:
        # dataset_name = local_data_dir + dataset_name if local_data_dir is not None else dataset_name
        # dataset = load_dataset(dataset_name, split="train", name="all")
        dataset = load_dataset(local_data_dir, split="train", name="all")
    elif dataset_name == "HuggingFaceH4/ultrafeedback_binarized":
        # dataset_name = local_data_dir + dataset_name if local_data_dir is not None else dataset_name
        # dataset = load_dataset(dataset_name, split="train_sft")
        dataset = load_dataset(local_data_dir, split="train_sft")
    else:
        # dataset_name = local_data_dir + dataset_name if local_data_dir is not None else dataset_name
        # dataset = load_dataset(dataset_name, split="train")
        # dataset = load_dataset(local_data_dir, split="train")
        # dataset = load_dataset("tatsu-lab/alpaca", split="train")
        dataset = load_dataset("vicgalle/alpaca-gpt4", split="train")
    return dataset

def process_sft_dataset(dataset_name, dataset, dataset_sample=None):
    if dataset_name in ["lucasmccabe-lmi/CodeAlpaca-20k", "yahma/alpaca-cleaned", "FinGPT/fingpt-sentiment-train"]:
        dataset = dataset.map(alpaca_format, remove_columns=['input', 'output'], desc=f"Preprocessing {dataset_name} for unified format.")
    elif dataset_name in ["WizardLM/WizardLM_evol_instruct_70k"]:
        dataset = dataset.rename_column("output", "response")
    elif dataset_name in ["tatsu-lab/alpaca", "vicgalle/alpaca-gpt4", "gbharti/finance-alpaca"]:
        dataset = dataset.map(alpaca_format, remove_columns=['input', 'output', 'text'], desc=f"Preprocessing {dataset_name} for unified format.")
    elif dataset_name in ["TIGER-Lab/MathInstruct"]:
        df = pd.DataFrame(dataset)
        df = df.drop_duplicates(subset=['instruction'])
        dataset = datasets.Dataset.from_pandas(df)
        dataset = dataset.rename_column("output", "response")
        dataset = dataset.remove_columns(['source'])
    elif dataset_name in ["lighteval/MATH"]:
        dataset = dataset.rename_column("solution", "response")
        dataset = dataset.rename_column("problem", "instruction")
        dataset = dataset.remove_columns(['level', 'type'])
    elif dataset_name in ['gsm8k']:
        dataset = dataset.rename_column("question", "instruction")
        dataset = dataset.rename_column("answer", "response")
    elif dataset_name in ['medalpaca/medical_meadow_medical_flashcards']:       # TODO: 'lavita/ChatDoctor-HealthCareMagic-100k'. not sure whether to discard the instruction.
        dataset = dataset.remove_columns(['instruction'])
        dataset = dataset.rename_column("input", "instruction")
        dataset = dataset.rename_column("output", "response")
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported.")
    dataset = dataset.shuffle(seed=2023)
    if dataset_sample:
        num_sample = min(len(dataset), dataset_sample)
        dataset = dataset.select(range(num_sample))
    print(f">> ===== After processing, Dataset {dataset_name} has {len(dataset)} examples. =====")
    return dataset

def alpaca_format(example):
    if example['input'] == "":
        example["instruction"] = example["instruction"]
    else:
        example["instruction"] = example["instruction"] + " " + example['input']
    example["response"] = example['output']
    return example


def process_dpo_dataset(dataset_name, dataset, template_name, dataset_sample):
    if dataset_name in ["Anthropic/hh-rlhf"]:
        dataset = dataset.map(partial(split_hh, template_name=template_name), load_from_cache_file=False)
    elif dataset_name in ["HuggingFaceH4/ultrafeedback_binarized"]:
        dataset = dataset.map(partial(split_ultrafeedback, template_name=template_name), load_from_cache_file=False)
        dataset = dataset.remove_columns(['prompt_id', 'messages', 'score_chosen', 'score_rejected'])
    
    dataset = dataset.shuffle(seed=2023)
    if dataset_sample:
        num_sample = min(len(dataset), dataset_sample)
        dataset = dataset.select(range(num_sample))
    print(f">> ===== After processing, Dataset {dataset_name} has {len(dataset)} examples. =====")
    print(f">> ===== Data Example =====")
    print(dataset[0])
    print(f">> {'='*50}")
    return dataset
    
def find_common_prefix(str1, str2):
    prefix = ""
    for i in range(min(len(str1), len(str2))):
        if str1[i] == str2[i]:
            prefix += str1[i]
        else:
            break
    return prefix




