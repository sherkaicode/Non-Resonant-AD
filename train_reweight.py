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

print("Reweight Model ...")
print("CR has", len(data_events_cr), "data events,", len(mc_events_cr), "MC events.")

data_cr_train = data_events_cr.values
mc_cr_train = mc_events_cr.values

input_x_train_CR = np.concatenate([data_cr_train, mc_cr_train], axis=0)

mc_cr_labels = np.zeros(mc_cr_train.shape[0]).reshape(-1, 1)
data_cr_labels = np.ones(data_cr_train.shape[0]).reshape(-1, 1)
input_y_train_CR = np.concatenate([data_cr_labels, mc_cr_labels], axis=0)

print("Training Data Shape:", input_x_train_CR.shape, input_y_train_CR.shape)

with open(f"{config_path}/reweight_physics.yml", 'r') as stream:
    params_rw = yaml.safe_load(stream)

NN_rw = Classifier(n_inputs=(len(context_var) + len(features)), layers=params_rw["layers"], learning_rate=params_rw["learning_rate"], device=device)

print("Training Reweight...")
NN_rw.train(input_x_train_CR, input_y_train_CR, save_model=True, batch_size=params_rw["batch_size"], n_epochs=params_rw["n_epochs"], model_name=f"reweight", outdir=model_path)

print("Training Done for Reweight Model")