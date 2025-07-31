import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm
from sklearn.manifold import TSNE
from copy import deepcopy
from tqdm import trange
from huggingface_hub import login
model_name = "/Path/To/Base/Model"
# #We directly use the official implementation of LESA from https://github.com/yangyifei729/LESA
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map='balanced_low_0')

dir_path = r"/Path/To/LESA/Weight"
up_proj_23_31_inter = torch.load(f'{dir_path}/up_proj_15_31_inter.pt')
down_proj_23_31_inter = torch.load(f'{dir_path}/down_proj_15_31_inter.pt')
gate_proj_23_31_inter = torch.load(f'{dir_path}/gate_proj_15_31_inter.pt')
k_proj_23_31_inter = torch.load(f'{dir_path}/k_proj_15_31_inter.pt')
v_proj_23_31_inter = torch.load(f'{dir_path}/v_proj_15_31_inter.pt')
o_proj_23_31_inter = torch.load(f'{dir_path}/o_proj_15_31_inter.pt')
q_proj_23_31_inter = torch.load(f'{dir_path}/q_proj_15_31_inter.pt')

new_layers_ls = [deepcopy(model.model.layers[0]) for _ in range(16)]
new_layers_ls[0].mlp.up_proj.weight.data

idx = 0
for (q, k, v, o, up, down, gate) in zip(q_proj_23_31_inter, k_proj_23_31_inter, v_proj_23_31_inter, o_proj_23_31_inter,
                                        up_proj_23_31_inter, down_proj_23_31_inter, gate_proj_23_31_inter):
    assert new_layers_ls[idx].self_attn.q_proj.weight.data.shape == q.shape
    assert new_layers_ls[idx].self_attn.k_proj.weight.data.shape == k.shape
    assert new_layers_ls[idx].self_attn.v_proj.weight.data.shape == v.shape
    assert new_layers_ls[idx].self_attn.o_proj.weight.data.shape == o.shape
    assert new_layers_ls[idx].mlp.up_proj.weight.data.shape == up.shape
    assert new_layers_ls[idx].mlp.down_proj.weight.data.shape == down.shape
    assert new_layers_ls[idx].mlp.gate_proj.weight.data.shape == gate.shape

    new_layers_ls[idx].self_attn.q_proj.weight.data = q
    new_layers_ls[idx].self_attn.k_proj.weight.data = k
    new_layers_ls[idx].self_attn.v_proj.weight.data = v
    new_layers_ls[idx].self_attn.o_proj.weight.data = o
    new_layers_ls[idx].mlp.up_proj.weight.data = up
    new_layers_ls[idx].mlp.down_proj.weight.data = down
    new_layers_ls[idx].mlp.gate_proj.weight.data = gate

    idx += 1

new_layers_ls = [t.to('cuda:0') for t in new_layers_ls]

for i in range(31, 15, -1):
    model.model.layers.insert(i, new_layers_ls[i-16])

# reset layer_idx
for idx in range(len(model.model.layers)):
    model.model.layers[idx].self_attn.layer_idx = idx
model.config.num_hidden_layers = len(model.model.layers)
print(model.config.num_hidden_layers)

model.save_pretrained("/Path/To/LESA")
tokenizer.save_pretrained("/Path/To/LESA")
