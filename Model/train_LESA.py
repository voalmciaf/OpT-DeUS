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
import torch.optim as optim
import torch.nn as nn
import os
import time
#We directly use the official implementation of LESA from https://github.com/yangyifei729/LESA
model_name = "/Path/To/Base/Model"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #This is changed due to different GPU resources

# load Model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map='balanced_low_0')

class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

def prepare_data_sliding_window(VT_sep):
    X_list = []
    Y_list = []

    for i in range(1, len(VT_sep) - 1):
        V1 = VT_sep[i - 1]
        V2 = VT_sep[i]
        V3 = VT_sep[i + 1]

        X = torch.cat([V1, V3], dim=1)
        Y = V2

        X_list.append(X)
        Y_list.append(Y)

    X = torch.cat(X_list, dim=0)
    Y = torch.cat(Y_list, dim=0)

    return X, Y


def save_tensor_list_as_file(tensor_list, file_path):
    torch.save(tensor_list, file_path)
    print(f"Saved tensor list to: {file_path}")


# Define the matrices names
weight_matrices = [
    'mlp.down_proj.weight.data',
    'mlp.up_proj.weight.data',
    'mlp.gate_proj.weight.data',
    'self_attn.q_proj.weight.data',
    'self_attn.k_proj.weight.data',
    'self_attn.v_proj.weight.data',
    'self_attn.o_proj.weight.data'
]


layer_select_num = len(model.model.layers)

top_k = 1

for weight_name in weight_matrices:
    print(f'weight_name:{weight_name}')
    concatenated_list = []
    for layer_idx, layer in enumerate(tqdm(model.model.layers[0:layer_select_num])):
        weight = eval(f"layer.{weight_name}")
        concatenated_list.append(weight)

    concatenated_list = [t.to('cuda:0') for t in concatenated_list]
    concatenated_matrix = torch.cat(concatenated_list, dim=1)
    #concatenated_matrix = concatenated_matrix.to('cpu') # shifted to cpu when CUDA OOM
    concatenated_matrix = concatenated_matrix.to('cuda:0')

    U, S, VT = torch.linalg.svd(concatenated_matrix, full_matrices=False)
    U, S, VT = U.cpu(), S.cpu(), VT.cpu()
    Sigma = torch.diag(S)
    if Sigma.size(0) < U.size(1):
        Sigma = torch.cat([Sigma, torch.zeros(U.size(1) - Sigma.size(0), S.size(0))], dim=0)
    if Sigma.size(1) < VT.size(0):
        Sigma = torch.cat([Sigma, torch.zeros(Sigma.size(0), VT.size(0) - Sigma.size(1))], dim=1)

    y_shape = concatenated_list[0].shape[1]

    VT_sep = [VT[:, i * y_shape : (i + 1) * y_shape] for i in range(layer_select_num)]

    X, Y = prepare_data_sliding_window(VT_sep)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    X, Y = X.to(device), Y.to(device)

    input_dim = X.shape[1]
    hidden_dim = 256
    output_dim = Y.shape[1]
    pred_model = SimpleNN(input_dim, hidden_dim, output_dim).to(device)

    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.AdamW(pred_model.parameters(), lr=0.001)

    epochs = 10
    batch_size = 64
    dataset = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        pred_model.train()
        total_loss = 0

        for batch_X, batch_Y in dataloader:
            optimizer.zero_grad()
            outputs = pred_model(batch_X)
            mse_loss = criterion(outputs, batch_Y)
            norm_loss = criterion(torch.norm(outputs, p=2), torch.norm(batch_Y, p=2))
            lambda_reg = 0.0005
            loss = (1 - lambda_reg) * mse_loss + lambda_reg * norm_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(dataloader):.6f}")


    pred_v_ls = []
    pred_model.eval()
    with torch.no_grad():
        for idx in range(15, 31):
            V1 = VT_sep[idx].to(device)
            V3 = VT_sep[idx+1].to(device)

            X_eval = torch.cat([V1, V3], dim=1)

            predictions = pred_model(X_eval)
            pred_v_ls.append(predictions)
    SVD_mean_recons_ls = [(U * S) @ pred_v.to('cpu') for pred_v in pred_v_ls]
    print(torch.norm(SVD_mean_recons_ls[0]))
    weight_layer0 = eval(f'model.model.layers[0].{weight_name}')
    print(torch.norm(weight_layer0))

    save_dir = r"/Path/To/LESA/Weight"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = f'{save_dir}/{weight_name.split(".")[1]}_15_31_inter.pt'
    save_tensor_list_as_file(SVD_mean_recons_ls, save_path)



