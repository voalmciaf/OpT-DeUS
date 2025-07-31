from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import transformers
from datasets import load_dataset, load_from_disk, concatenate_datasets
from functools import partial
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling,set_seed
from transformers import AutoTokenizer
from transformers import optimization
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer,get_wsd_schedule
import torch
import pandas as pd
import sys
from accelerate import Accelerator
import random
import numpy as np
import torch.nn.functional as F
import re
import os
from accelerate.utils import DistributedDataParallelKwargs
from torch.optim import Optimizer
from pathlib import Path
from peft import LoraConfig, LoraModel, get_peft_model, PeftModel
from transformers import TrainerCallback
from datasets import config
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetCount, nvmlDeviceGetUtilizationRates,nvmlDeviceGetMemoryInfo
from huggingface_hub import login
import copy
import time
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
set_seed(42)

model = AutoModelForCausalLM.from_pretrained("/Path/To/Base/Model")
tokenizer = AutoTokenizer.from_pretrained("/Path/To/Tokenizer")
Expand = AutoModelForCausalLM.from_pretrained("/Path/To/Base/Model",num_hidden_layers=48)
index_mapping = {
    0:0,
    1:1,
    2:2,
    3:3,
    4:4,
    5:5,
    6:6,
    7:7,
    8:8,
    9:9,
    10:10,
    11:11,
    12:12,
    13:13,
    14:14,
    15:15,
    16:1,
    17:16,
    18:1,
    19:17,
    20:1,
    21:18,
    22:1,
    23:19,
    24:1,
    25:20,
    26:1,
    27:21,
    28:1,
    29:22,
    30:1,
    31:23,
    32:1,
    33:24,
    34:1,
    35:25,
    36:1,
    37:26,
    38:1,
    39:27,
    40:1,
    41:28,
    42:1,
    43:29,
    44:1,
    45:30,
    46:1,
    47:31
}

def average_weight(previous_index,next_index):
  previous = Expand.model.layers[previous_index].state_dict()
  next = Expand.model.layers[next_index].state_dict()
  average_weight=Expand.model.layers[next_index-1].state_dict()
  for key in average_weight:
      average_weight[key] = (previous[key] + next[key]) / 2
  return average_weight


for new_idx, old_idx in index_mapping.items():
    Expand.model.layers[new_idx].load_state_dict(model.model.layers[old_idx].state_dict())
for i in[16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46]:#These will be the trainable layers
    Expand.model.layers[i].load_state_dict(average_weight(i - 1, i + 1))
Expand.save_pretrained("/Path/To/Avg-DeUS")
tokenizer.save_pretrained("/Path/To/Avg-DeUS")