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

print("Context weight training for Generate model...")
print("CR has", len(data_events_cr), "data events,", len(mc_events_cr), "MC events.")

data_cr_train_context = data_events_cr[context_var].values
mc_cr_train_context = mc_events_cr[context_var].values

input_x_train_CR_context = np.concatenate([data_cr_train_context, mc_cr_train_context], axis=0)

mc_cr_labels_context = np.zeros(mc_cr_train_context.shape[0]).reshape(-1, 1)
data_cr_labels_context = np.ones(data_cr_train_context.shape[0]).reshape(-1, 1)
input_y_train_CR_context = np.concatenate([data_cr_labels_context, mc_cr_labels_context], axis=0)


print("Training Data Shape (Context Weights):", input_x_train_CR_context.shape, input_y_train_CR_context.shape)

with open(f"{config_path}/context_weights_physics.yml", 'r') as stream:
    params_cw = yaml.safe_load(stream)

NN_cw = Classifier(n_inputs=len(context_var), layers=params_cw["layers"], learning_rate=params_cw["learning_rate"], device=device)
print(NN_cw.n_inputs)
print("Training context weights...")
NN_cw.train(input_x_train_CR_context, input_y_train_CR_context, save_model=True, batch_size=params_cw["batch_size"], n_epochs=params_cw["n_epochs"], model_name=f"context_weight", outdir=model_path)

print("Training Done for context weights")