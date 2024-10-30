# %%
import os
from tqdm import tqdm
from huggingface_hub import login
import torch
import torch.nn as nn
import math
import einops
from fancy_einsum import einsum
import tqdm.auto as tqdm
import plotly.express as px
from jaxtyping import Float
from functools import partial
import transformer_lens.utils as utils
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from transformer_lens.hook_points import (
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, FactoredMatrix
from collections import defaultdict

# %%
with open("access.tok", "r") as file:
    access_token = file.read()
    login(token=access_token)

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")

# %%
from datasets import load_dataset  
import transformer_lens
from transformer_lens import HookedTransformer
from sae_lens import SAE, HookedSAETransformer

torch.set_grad_enabled(False)

NEG_SET_SIZE = 200
POS_SET_SIZE = 200

# %%
# Load gemma model 
model = HookedSAETransformer.from_pretrained("gemma-2-2b", device=device)

# Load SAE on res stream of gemma model, plus cfg and sparsity val
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release="gemma-scope-2b-pt-res",
    sae_id="layer_14/width_16k/average_l0_83",
    device=device
)

# %%
df = pd.read_csv('dataset/harmful_strings.csv')
negative_set = df[df.columns[0]].values[:NEG_SET_SIZE]
print(f"Negative set size: {len(negative_set)}")

# %%
positive = pd.read_json('dataset/alpaca_data.json')
positive_set = positive['output'].values[:POS_SET_SIZE]
print(f"Positive set size: {len(positive_set)}")

# %%
# Define custom Dataset and DataLoader
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return self.texts[idx]

batch_size = 32  # Adjust as per your GPU memory capacity

negative_dataset = TextDataset(negative_set)
positive_dataset = TextDataset(positive_set)

negative_loader = DataLoader(negative_dataset, batch_size=batch_size, shuffle=False)
positive_loader = DataLoader(positive_dataset, batch_size=batch_size, shuffle=False)

# %%
# Collect top neurons for negative and positive sets
top_neurons_neg = defaultdict(list)
top_neurons_pos = defaultdict(list)

# Process negative set
for batch in negative_loader:
    tokens = model.to_tokens(batch).to(device)
    _, cache = model.run_with_cache_with_saes(tokens, saes=[sae])
    activations = cache['blocks.14.hook_resid_post.hook_sae_acts_post'][:, -1, :]  # Shape: (batch_size, hidden_size)
    vals, inds = torch.topk(activations, 15, dim=-1)  # Shape: (batch_size, 15)
    for i in range(vals.size(0)):
        for neuron_idx in range(15):
            neuron = inds[i, neuron_idx].item()
            activation = vals[i, neuron_idx].item()
            top_neurons_neg[neuron].append(activation)

# Process positive set
for batch in positive_loader:
    tokens = model.to_tokens(batch).to(device)
    _, cache = model.run_with_cache_with_saes(tokens, saes=[sae])
    activations = cache['blocks.14.hook_resid_post.hook_sae_acts_post'][:, -1, :]  # Shape: (batch_size, hidden_size)
    vals, inds = torch.topk(activations, 15, dim=-1)  # Shape: (batch_size, 15)
    for i in range(vals.size(0)):
        for neuron_idx in range(15):
            neuron = inds[i, neuron_idx].item()
            activation = vals[i, neuron_idx].item()
            top_neurons_pos[neuron].append(activation)

print(f"Top neurons in negative set: {len(top_neurons_neg)}")
print(f"Top neurons in positive set: {len(top_neurons_pos)}")

# %%
def filter_neurons(top_neurons_neg, top_neurons_pos, threshold=5.0):
    """
    Filters out neurons that are highly activated in both the negative and positive sets.
    """
    filtered_neurons_neg = {}
    filtered_neurons_pos = {}

    for neuron, activations in top_neurons_neg.items():
        if neuron in top_neurons_pos and any(val >= threshold for val in activations) and any(val >= threshold for val in top_neurons_pos[neuron]):
            continue 
        else:
            filtered_neurons_neg[neuron] = activations

    for neuron, activations in top_neurons_pos.items():
        if neuron not in top_neurons_neg or not any(val >= threshold for val in top_neurons_neg[neuron]):
            filtered_neurons_pos[neuron] = activations

    return filtered_neurons_neg, filtered_neurons_pos

filtered_neg, filtered_pos = filter_neurons(top_neurons_neg, top_neurons_pos, 0)
print(f"Len: {len(filtered_neg)}. Filtered negative neurons: {filtered_neg}")
print(f"Len: {len(filtered_pos)}. Filtered positive neurons: {filtered_pos}")

# %%
# Compute mean activations for filtered neurons
top_neurons_neg_mean = {neuron: len(activations) for neuron, activations in filtered_neg.items()}
top_neurons_pos_mean = {neuron: len(activations) for neuron, activations in filtered_pos.items()}

# Sort by activation count
top_neurons_neg_mean = {k: v for k, v in sorted(top_neurons_neg_mean.items(), key=lambda item: item[1], reverse=True)}
top_neurons_pos_mean = {k: v for k, v in sorted(top_neurons_pos_mean.items(), key=lambda item: item[1], reverse=True)}

# Print first few
print(list(top_neurons_neg_mean.items())[:200])
print(list(top_neurons_pos_mean.items())[:200])

# %%
# Train classifier on SAE activations
activations_list = []
labels_list = []

# Process negative set
for batch in negative_loader:
    tokens = model.to_tokens(batch).to(device)
    _, cache = model.run_with_cache_with_saes(tokens, saes=[sae])
    activations = cache['blocks.14.hook_resid_post.hook_sae_acts_post'][:, -1, :].cpu().numpy()
    activations_list.extend(activations)
    labels_list.extend([0]*activations.shape[0])

# Process positive set
for batch in positive_loader:
    tokens = model.to_tokens(batch).to(device)
    _, cache = model.run_with_cache_with_saes(tokens, saes=[sae])
    activations = cache['blocks.14.hook_resid_post.hook_sae_acts_post'][:, -1, :].cpu().numpy()
    activations_list.extend(activations)
    labels_list.extend([1]*activations.shape[0])

# Data
X = np.array(activations_list)
y = np.array(labels_list)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale activation features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = LogisticRegression(max_iter=1000, solver='lbfgs') 
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy:.4f}')

# %%
# Train classifier on base activations
activations_list = []
labels_list = []

# Process negative set
for batch in negative_loader:
    tokens = model.to_tokens(batch).to(device)
    _, cache = model.run_with_cache(tokens)
    # Assuming 'decompose_resid' supports batching
    res_stream = cache.decompose_resid(layer=15, return_labels=False, mode='attn', incl_embeds=False, pos_slice=slice(-2, -1))
    seven_out = res_stream[-1, :, -1, :].cpu().numpy()  # Shape: (batch_size, d_model)
    activations_list.extend(seven_out)
    labels_list.extend([0]*seven_out.shape[0])

# Process positive set
for batch in positive_loader:
    tokens = model.to_tokens(batch).to(device)
    _, cache = model.run_with_cache(tokens)
    res_stream = cache.decompose_resid(layer=15, return_labels=False, mode='attn', incl_embeds=False, pos_slice=slice(-2, -1))
    seven_out = res_stream[-1, :, -1, :].cpu().numpy()  # Shape: (batch_size, d_model)
    activations_list.extend(seven_out)
    labels_list.extend([1]*seven_out.shape[0])

# Data and split
X = np.array(activations_list)
y = np.array(labels_list)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train classifier
clf = LogisticRegression(max_iter=1000, solver='lbfgs') 
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy:.4f}')

# %%
# Reject sample if activation at any of top N neurons is above threshold T
THRESHOLD = 2
NEURONS_TO_CONSIDER = 100

top_neurons_neg_occ = list(top_neurons_neg_mean.items())[:NEURONS_TO_CONSIDER]
neg_indicators = [neuron for neuron, _ in top_neurons_neg_occ]

print(f"Indicator neurons: {neg_indicators}")

# Initialize classification arrays
neg_classifications = np.zeros(NEG_SET_SIZE)
pos_classifications = np.zeros(POS_SET_SIZE)

# Process negative set
e = 0
for batch in negative_loader:
    tokens = model.to_tokens(batch).to(device)
    _, cache = model.run_with_cache_with_saes(tokens, saes=[sae])
    val = cache['blocks.7.hook_resid_pre.hook_sae_acts_post'][:, -1, :]  # Shape: (batch_size, hidden_size)
    critical_activations = val[:, neg_indicators]
    exceed_threshold = (critical_activations > THRESHOLD).any(dim=1).cpu().numpy().astype(int)
    batch_size = len(batch)
    neg_classifications[e:e+batch_size] = exceed_threshold
    e += batch_size

# Process positive set
e = 0
for batch in positive_loader:
    tokens = model.to_tokens(batch).to(device)
    _, cache = model.run_with_cache_with_saes(tokens, saes=[sae])
    val = cache['blocks.7.hook_resid_pre.hook_sae_acts_post'][:, -1, :]  # Shape: (batch_size, hidden_size)
    critical_activations = val[:, neg_indicators]
    exceed_threshold = (critical_activations > THRESHOLD).any(dim=1).cpu().numpy().astype(int)
    batch_size = len(batch)
    pos_classifications[e:e+batch_size] = exceed_threshold
    e += batch_size

print("Negative set rejections: ", np.sum(neg_classifications))
print("Positive set rejections: ", np.sum(pos_classifications))