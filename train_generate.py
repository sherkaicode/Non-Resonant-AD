import numpy as np
import torch
import os
import logging
import yaml
import pandas as pd
import sys
import matplotlib.pyplot as plt

helpers_path = os.path.join('/home/aegis/Titan1/NRAD/data/model_scripts')
sys.path.insert(0, os.path.abspath(helpers_path))
from Classifier import Classifier
from SimpleMAF import SimpleMAF

dataset_path = "/home/aegis/Titan1/NRAD/data/Scaled_Regions"
config_path = "/home/aegis/Titan1/NRAD/data/configs"
model_path = "/home/aegis/Titan1/NRAD/data/Models"

print("Setting up device...")
CUDA = torch.cuda.is_available()
print("cuda available:", CUDA)
device = torch.device("cuda" if CUDA else "cpu")

print("Loadding MC events...")
mc_events_cr = pd.read_parquet(os.path.join(dataset_path, "MC_CR_train.parquet"))
data_events_cr = pd.read_parquet(os.path.join(dataset_path, "Data_CR_train.parquet"))

context_var = ['ht', 'met_recalc_pt']
features = [col for col in mc_events_cr.columns if col not in context_var]

print("Context Variables:", context_var)
print("Feature Variables:", features)

print("Training Generate...")
print("CR has", len(data_events_cr), "data events")

data_context_cr_train = data_events_cr[context_var].values
data_features_cr_train = data_events_cr[features].values

with open(f"{config_path}/generate_physics.yml", 'r') as stream:
    params = yaml.safe_load(stream)

MAF = SimpleMAF(num_features=len(features), num_context=len(context_var), device=device, num_layers=params["n_layers"], num_hidden_features=params["n_hidden_features"], learning_rate = params["learning_rate"])

MAF.train(data=data_features_cr_train, cond=data_context_cr_train, batch_size=params["batch_size"], n_epochs=params["n_epochs"], outdir=model_path, save_model=True, model_name=f"generate")
print("Done training!")