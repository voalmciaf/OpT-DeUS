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
####################Create LLAMA PRO
index_mapping = {
    0:0,
    1:1,
    2:1,
    3:2,
    4:3,
    5:3,
    6:4,
    7:5,
    8:5,
    9:6,
    10:7,
    11:7,
    12:8,
    13:9,
    14:9,
    15:10,
    16:11,
    17:11,
    18:12,
    19:13,
    20:13,
    21:14,
    22:15,
    23:15,
    24:16,
    25:17,
    26:17,
    27:18,
    28:19,
    29:19,
    30:20,
    31:21,
    32:21,
    33:22,
    34:23,
    35:23,
    36:24,
    37:25,
    38:25,
    39:26,
    40:27,
    41:27,
    42:28,
    43:29,
    44:29,
    45:30,
    46:31,
    47:31
}
for new_idx, old_idx in index_mapping.items():
    Expand.model.layers[new_idx].load_state_dict(model.model.layers[old_idx].state_dict())
for i in[2,5,8,11,14,17,20,23,26,29,32,35,38,41,44,47]: #These will be the trainable layers
    nn.init.zeros_(Expand.model.layers[i].self_attn.o_proj.weight) #Zero-Initialization
    nn.init.zeros_(Expand.model.layers[i].mlp.down_proj.weight) #Zero-Initialization
Expand.save_pretrained("/Path/To/Llama-Pro")
tokenizer.save_pretrained("/Path/To/Llama-Pro")

# #############################Create SOLAR
for i in range(0,24):
    Expand.model.layers[i].load_state_dict(model.model.layers[i].state_dict())
for i in range(24,48):
    Expand.model.layers[i].load_state_dict(model.model.layers[i - 16].state_dict())
Expand.save_pretrained("/Path/To/SOLAR")
tokenizer.save_pretrained("/Path/To/SOLAR")