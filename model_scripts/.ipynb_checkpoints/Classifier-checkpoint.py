import matplotlib.pyplot as plt
import numpy as np
# from datetime import datetime

import torch
import torch.nn as nn           # nn Module 
import torch.optim as optim     # Optimizer (Adam)
import torch.nn.functional as F #Loss function

# from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

import os
import sys
script_dir = os.path.dirname(__file__) 
# helpers_path = os.path.join(script_dir, '..', 'model_scripts')  
sys.path.insert(0, os.path.abspath(script_dir))
from utils import EarlyStopping, equalize_weights

from tqdm import tqdm
import yaml

import os
import logging

log = logging.getLogger("run")

# Turn off matplotlib DEBUG messages
plt.set_loglevel(level="warning")

class Model(nn.Module):
    def __init__(self, 
                 layers,
                 n_inputs,
                 device="cpu"):
        
        super().__init__()
        
        self.layers = []
        for nodes in layers:
            self.layers.append(nn.Linear(n_inputs, nodes))
            self.layers.append(nn.ReLU())
            n_inputs = nodes
        self.layers.append(nn.Linear(n_inputs, 1))
        self.layers.append(nn.Sigmoid())
        self.model_stack = nn.Sequential(*self.layers)
        self.device = device
    def forward(self, x):
        
        return self.model_stack(x)

    
class Classifier():
    def __init__(self, 
                 n_inputs,
                 layers=[64,64,64], 
                 learning_rate=1e-3, 
                 loss_type="binary_crossentropy", 
                 device="cpu",
                 scale_data=False):

        self.n_inputs = n_inputs
        self.device = device
        self.model = Model(layers, n_inputs=n_inputs).to(self.device)
        self.scale_data = scale_data

        if loss_type == 'binary_crossentropy':
            self.loss_func = F.binary_cross_entropy
        else:
            raise NotImplementedError
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)        

        
    def to(self, device):
        self.device = device
        self.model.to(device)

    def np_to_torch(self, array):
    
        return torch.tensor(array.astype(np.float32))

    def process_data(self, input_x, input_y, batch_size, weights=None):
        
        if self.n_inputs != input_x.shape[-1]:
            raise RuntimeError(f"input data has {input_x.shape[-1]} features, which doesn't match with number of features!")
        
        if weights is not None:
            x_train, x_val, w_train, w_val, y_train, y_val = train_test_split(input_x, weights, input_y, test_size=0.33, random_state=42)
            
            w_train, w_val = equalize_weights(y_train, y_val, w_train, w_val)
            
            w_train = self.np_to_torch(w_train)
            w_val = self.np_to_torch(w_val)
        else:
            x_train, x_val, y_train, y_val = train_test_split(input_x, input_y, test_size=0.33, random_state=42)
        
        # Data preprocessing
        if self.scale_data:
            # Scaler transform for preprocessing
            self.scaler = StandardScaler().fit(x_train)
            x_train = self.scaler.transform(x_train)
            x_val = self.scaler.transform(x_val)
        
        x_train = self.np_to_torch(x_train)
        y_train = self.np_to_torch(y_train)
        
        x_val = self.np_to_torch(x_val)
        y_val = self.np_to_torch(y_val)
        
        if weights is not None:
            train_dataset = torch.utils.data.TensorDataset(x_train, w_train, y_train)
            val_dataset = torch.utils.data.TensorDataset(x_val, w_val, y_val)
        else:
            train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
            val_dataset = torch.utils.data.TensorDataset(x_val, y_val)
        
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
        
        return train_dataloader, val_dataloader
    
    def train(self, input_x, input_y, n_epochs=200, batch_size=512, weights=None, seed=1, outdir="./", early_stop=True, patience=5, min_delta=0.00001, save_model=False, model_name="classifier"):
        
        update_epochs = 1
        # save the best model
        best_val_loss = 10000
        best_epoch = -1
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        epochs, epochs_val = [], []
        losses, losses_val = [], []
        
        if early_stop:
            early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
            
        train_data, val_data = self.process_data(input_x, input_y, weights=weights, batch_size=batch_size)
        
        for epoch in tqdm(range(n_epochs), ascii=' >='):
            # Training
            
            losses_batch_per_e = []
            
            self.model.train()
            
            for batch_ndx, data in enumerate(train_data):
                
                if weights is not None:
                    batch_inputs, batch_weights, batch_labels = data
                    batch_inputs, batch_weights, batch_labels = batch_inputs.to(self.device), batch_weights.to(self.device), batch_labels.to(self.device)
                else:
                    batch_inputs, batch_labels = data
                    batch_inputs, batch_labels = batch_inputs.to(self.device), batch_labels.to(self.device)
                    batch_weights = None
                
                self.optimizer.zero_grad()
                batch_outputs = self.model(batch_inputs)
                loss = self.loss_func(batch_outputs, batch_labels, weight=batch_weights)
                losses_batch_per_e.append(loss.detach().cpu().numpy())
                loss.backward()
                self.optimizer.step()

            mean_loss = np.mean(losses_batch_per_e)
            epochs.append(epoch)
            losses.append(mean_loss)
            
            # Validation
            
            with torch.no_grad():

                self.model.eval()
                val_losses_batch_per_e = []

                for batch_ndx, data in enumerate(val_data):

                    if weights is not None:
                        batch_inputs, batch_weights, batch_labels = data
                        batch_inputs, batch_weights, batch_labels = batch_inputs.to(self.device), batch_weights.to(self.device), batch_labels.to(self.device)
                    else:
                        batch_inputs, batch_labels = data
                        batch_inputs, batch_labels = batch_inputs.to(self.device), batch_labels.to(self.device)
                        batch_weights = None

                    batch_outputs = self.model(batch_inputs)
                    val_loss = self.loss_func(batch_outputs, batch_labels, weight=batch_weights)
                    val_losses_batch_per_e.append(val_loss.detach().cpu().numpy())

                mean_val_loss = np.mean(val_losses_batch_per_e)
                losses_val.append(mean_val_loss)
                
                # see if the model has the best val loss
                if mean_val_loss < best_val_loss:
                    best_val_loss  = mean_val_loss
                    best_epoch = epoch
                    # save the model
                    if save_model:
                        model_path = f"{outdir}/{model_name}.pt"
                        torch.save(self, model_path)


                if early_stop:
                    early_stopping(mean_val_loss)

                log.debug(f"Epoch: {epoch} - loss: {mean_loss:.3f} - val loss: {mean_val_loss:.3f}")
                    
            if early_stop:
                if early_stopping.early_stop:
                    break
        if save_model:
            log.info(f"Trained classifier from best epoch {best_epoch} save at {model_path}.")
            #Save loss plot data
            os.makedirs(f"{outdir}/loss_data/", exist_ok=True)
            np.savez(f"{outdir}/loss_data/{model_name}_loss", epochs = epochs, train_loss = losses, val_loss = losses_val)
        else:
            log.info(f"Done training classifier. The best epoch is {best_epoch}. Model is not saved.")

    
    def evaluation(self, X_test):
        
        self.model.eval()
        
        with torch.no_grad():
            if self.scale_data:
                X_test = self.scaler.transform(X_test)
            x_test = self.np_to_torch(X_test).to(self.device)
            outputs = self.model(x_test).detach().cpu().numpy()
            
        return outputs