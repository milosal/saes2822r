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
# load gemma model 
model = HookedSAETransformer.from_pretrained("gemma-2-2b", device = device)

# load sae on res stream of gemma model, plus cfg and sparsity val
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release = "gemma-scope-2b-pt-res",
    sae_id = "layer_14/width_16k/average_l0_83",
    device = device
)

# %%
df = pd.read_csv('dataset/harmful_strings.csv')

columns_as_arrays = [df[col].values for col in df.columns]

array_dict = {col: df[col].values for col in df.columns}

negative_set = columns_as_arrays[0]
negative_set = negative_set[:NEG_SET_SIZE]
print(len(negative_set))

# %%
positive = pd.read_json('dataset/alpaca_data.json')

positive_set = positive['output'].values
positive_set = positive_set[:POS_SET_SIZE]
print(len(positive_set))

# %%
sae.use_error_term

top_neurons_neg = defaultdict(list)
top_neurons_pos = defaultdict(list)

for example in negative_set:
    _, cache = model.run_with_cache_with_saes(example, saes=[sae])

    # get top 15 firing sae neurons
    vals, inds = torch.topk(cache['blocks.14.hook_resid_post.hook_sae_acts_post'][0, -1, :], 15)

    for datapoint in zip(inds, vals):
        top_neurons_neg[int(datapoint[0])].append(datapoint[1].item())
    

for example in positive_set:
    _, cache = model.run_with_cache_with_saes(example, saes=[sae])

    # get top 15 firing sae neurons
    vals, inds = torch.topk(cache['blocks.14.hook_resid_post.hook_sae_acts_post'][0, -1, :], 15)
    for datapoint in zip(inds, vals):
        top_neurons_pos[int(datapoint[0])].append(datapoint[1].item())

print(top_neurons_neg)
print(top_neurons_pos)

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
# train classifier on top activations
# average activations over each top case, sends to
# top_neurons_neg/pos = {idx: avg_act, idx2:avg_act2, ...}
top_neurons_neg_mean = {}
for entry in filtered_neg:
    top_neurons_neg_mean[entry] = len(filtered_neg[entry])

top_neurons_pos_mean = {}
for entry in filtered_pos:
    top_neurons_pos_mean[entry] = len(filtered_pos[entry])

print(top_neurons_neg_mean)
print(top_neurons_pos_mean)

# sort by avg activation
top_neurons_neg_mean = {k: v for k, v in sorted(top_neurons_neg_mean.items(), key=lambda item: item[1], reverse=True)}
top_neurons_pos_mean = {k: v for k, v in sorted(top_neurons_pos_mean.items(), key=lambda item: item[1], reverse=True)}


# print first few
print(list(top_neurons_neg_mean.items())[:200])
print(list(top_neurons_pos_mean.items())[:200])

# %%
# train classifier on sae activations
activations_list = []
labels_list = []

# 0 = negative, 1 = positive
for example_txt in negative_set:
    _, cache = model.run_with_cache_with_saes(example_txt, saes=[sae])
    activations = cache['blocks.14.hook_resid_post.hook_sae_acts_post'][0, -1, :].cpu().numpy()
    #print(activations.shape)

    activations_list.append(activations)
    labels_list.append(0)

for example_txt in positive_set:
    _, cache = model.run_with_cache_with_saes(example_txt, saes=[sae])
    activations = cache['blocks.14.hook_resid_post.hook_sae_acts_post'][0, -1, :].cpu().numpy()

    activations_list.append(activations)
    labels_list.append(1)   

# data
X = np.array(activations_list)
y = np.array(labels_list)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scale activation features
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
# train classifier on base activations
activations_list = []
labels_list = []

# 0 = negative, 1 = positive
for example_txt in negative_set:
    _, cache = model.run_with_cache(example_txt)
    res_stream = cache.decompose_resid(layer=15, return_labels=False, mode='attn', incl_embeds=False, pos_slice=slice(-2, -1))
    seven_out = res_stream[-1, 0, -1, :].cpu().numpy() # layer batch pos d_model
    #print(seven_out.shape)

    activations_list.append(seven_out)
    labels_list.append(0)

#print("done w bad")

for example_txt in positive_set:
    _, cache = model.run_with_cache(example_txt)
    res_stream = cache.decompose_resid(layer=15, return_labels=False, mode='attn', incl_embeds=False, pos_slice=slice(-2, -1))
    seven_out = res_stream[-1, 0, -1, :].cpu().numpy()  # layer batch pos d_model

    activations_list.append(seven_out)
    labels_list.append(1)

# data and split
X = np.array(activations_list)
y = np.array(labels_list)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scale features 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# train classifier
clf = LogisticRegression(max_iter=1000, solver='lbfgs') 
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy:.4f}')

# %%
# reject sample if activation at any of top N neurons is above threshold T
THRESHOLD = 2
NEURONS_TO_CONSIDER = 100

top_neurons_neg_occ = list(top_neurons_neg_mean.items())[:NEURONS_TO_CONSIDER]
top_neurons_pos_mean = []
neg_indicators = []

for entry in top_neurons_neg_occ:
    neg_indicators.append(entry[0])

print(neg_indicators)

# run an inference, if hits threshold on any of the neurons classify as bad
e = 0
neg_classifications = np.zeros(NEG_SET_SIZE)
for example in negative_set:
    _, cache = model.run_with_cache_with_saes(example, saes=[sae])

    # get data from indicator neurons
    val = cache['blocks.7.hook_resid_pre.hook_sae_acts_post'][0, -1, :]
    for critical_neuron_idx in neg_indicators:
        if (val[critical_neuron_idx] > THRESHOLD):
            neg_classifications[e] = 1
            break
    e += 1

pos_classifications = np.zeros(POS_SET_SIZE)


e = 0
for example in positive_set:
    _, cache = model.run_with_cache_with_saes(example, saes=[sae])

    # get data from indicator neurons
    val = cache['blocks.7.hook_resid_pre.hook_sae_acts_post'][0, -1, :]
    for critical_neuron_idx in neg_indicators:
        if (val[critical_neuron_idx] > THRESHOLD):
            pos_classifications[e] = 1
            break
    e += 1

print(" RATS ! ", np.sum(neg_classifications) / 2)
print(" GENIUSES ! ", np.sum(pos_classifications) / 2)


