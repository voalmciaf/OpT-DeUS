import matplotlib.pyplot as plt
import transformers
from datasets import load_dataset, load_from_disk, concatenate_datasets
from functools import partial
from datasets import Dataset
from lxml.html.diff import token
from transformers import DataCollatorForLanguageModeling,set_seed
from transformers import AutoTokenizer
from transformers import optimization
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer,GPTNeoXForCausalLM
import torch
import pandas as pd
import sys
from accelerate import Accelerator
import random
import numpy as np
from huggingface_hub import login
import time
########################This is for creating the data for continue pre-training
def chunk_dataset(dataset, block_size):

    # Step 1: Concatenate all sequences together
    all_input_ids = np.concatenate(dataset['input_ids'])
    all_attention_mask = np.concatenate(dataset['attention_mask'])

    # Step 2: Chunk the sequences by block_size
    num_blocks = len(all_input_ids) // block_size
    chunked_input_ids = [all_input_ids[i * block_size:(i + 1) * block_size] for i in range(num_blocks)]
    chunked_attention_mask = [all_attention_mask[i * block_size:(i + 1) * block_size] for i in range(num_blocks)]

    # Create the new dataset
    new_data = {
        "input_ids": chunked_input_ids,
        "attention_mask": chunked_attention_mask
    }
    new_dataset = Dataset.from_dict(new_data)

    return new_dataset
def preprocess_function(examples):
    return tokenizer([x+tokenizer.eos_token for x in examples["text"]])
tokenizer = AutoTokenizer.from_pretrained("/Path/To/Tokenizer")

for idx in range(int(sys.argv[1]),int(sys.argv[2])):#total (0,5912)
    data_files = {"train": f"https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/data/CC-MAIN-2024-51"+
                           f"/000_{idx:05d}.parquet"}
    ds = load_dataset("parquet",data_files=data_files)
    lm_dataset = ds.map(
        preprocess_function,
        batched=True,
        num_proc=16,
        remove_columns=ds['train'].column_names
    )
    Chunk = chunk_dataset(lm_dataset['train'], 2048)
    Chunk.to_parquet(f"/Path/To/Save/Data/shard-{idx:05d}.parquet")

