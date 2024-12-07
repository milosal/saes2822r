import pandas as pd
import torch
# WMDP at https://huggingface.co/datasets/cais/wmdp

from datasets import load_dataset

ds_bio = load_dataset("cais/wmdp", "wmdp-bio")
ds_chem = load_dataset("cais/wmdp", "wmdp-chem")
ds_cyber = load_dataset("cais/wmdp", "wmdp-cyber")

