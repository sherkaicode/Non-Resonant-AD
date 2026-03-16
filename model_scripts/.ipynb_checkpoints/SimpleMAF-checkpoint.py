import matplotlib.pyplot as plt
import numpy as np
import logging

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

import os
import sys
script_dir = os.path.dirname(__file__) 
# helpers_path = os.path.join(script_dir, '..', 'model_scripts')  
sys.path.insert(0, os.path.abspath(script_dir))
from utils import EarlyStopping

from tqdm import tqdm

from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal, ConditionalDiagonalNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform, MaskedPiecewiseRationalQuadraticAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation

log = logging.getLogger("run")

# Turn off matplotlib DEBUG messages
plt.set_loglevel(level="warning")

class SimpleMAF:
    def __init__(self,
                 num_features,
                 num_context=None,
                 num_hidden_features=4,
                 num_layers=5,
                 learning_rate=1e-3,
                 base_dist=None,
                 act='relu',
                 device='cpu'):
        
        activations = {'relu': F.relu, 'sigmoid': F.sigmoid, 'tanh': F.tanh}
        activation = activations[act]
        
        self.nfeat = num_features
        self.ncond = num_context

        if base_dist is not None:
            self.base_dist = base_dist
            # set the base flow to be static
            for param in self.base_dist.parameters():
                param.requires_grad = False
        else:
            self.base_dist = StandardNormal(shape=[num_features])

        transforms = []
        for _ in range(num_layers):
            
            transforms.append(ReversePermutation(features=num_features))
            transforms.append(MaskedAffineAutoregressiveTransform(features=num_features, 
                                                                  hidden_features=num_hidden_features, 
                                                                  context_features=num_context, 
                                                                  activation = activation, 
                                                                 ))
            """
            
            transforms.append(MaskedPiecewiseRationalQuadraticAutoregressiveTransform(features=num_features, 
                                                                  hidden_features=num_hidden_features, 
                                                                  context_features=num_context, 
                                                                  activation = activation,
                                                                  num_blocks = 8, tail_bound = 3, tails = "linear", num_bins = 10))
                
            transforms.append(ReversePermutation(features=num_features))
            """
        self.transform = CompositeTransform(transforms)
        self.flow = Flow(self.transform, self.base_dist).to(device)
        self.optimizer = optim.Adam(self.flow.parameters(), lr=learning_rate)
        self.device = device

    def get_device(self):
        return next(self.flow.parameters()).device

    def to(self, device):
        self.flow.to(device)
        
    def scaler_transform_x(self, input_x):
        return self.scaler_x.transform(input_x)
    
    def scaler_transform_c(self, input_c):
        return self.scaler_c.transform(input_c)

    def scaler_inverse_x(self, input_x):
        return self.scaler_x.inverse_transform(input_x)
    
    def scaler_inverse_c(self, input_c):
        return self.scaler_c.inverse_transform(input_c)

    def np_to_torch(self, array):
    
#         return torch.tensor(array.astype(np.float32))
        return torch.tensor(array, dtype=torch.float32)
    
    def process_data(self, data, batch_size, cond=None):
        
        if self.nfeat != data.shape[1]:
            raise RuntimeError("input data dimention doesn't match with number of features!")
        if self.nfeat == 1:
            data = data.reshape(-1, 1)
        
        if cond is not None:
            if self.ncond != cond.shape[1]:
                raise RuntimeError("input cond dimention doesn't match with number of cond features!")
            if self.ncond == 1:
                cond = cond.reshape(-1, 1)
            data = np.concatenate((data, cond), axis=1)
        
        data_train, data_val = train_test_split(data, test_size=0.2, shuffle=True)

        # Data preprocessing
        #x_train = self.scaler_x.fit_transform(data_train[:, :-self.ncond])
        #x_val = self.scaler_x.transform(data_val[:, :-self.ncond])
        
        x_train = data_train[:, :-self.ncond]
        x_val = data_val[:, :-self.ncond]

        #c_train = self.scaler_c.fit_transform(data_train[:, -self.ncond:])
        #c_val = self.scaler_c.transform(data_val[:, -self.ncond:])
        
        c_train = data_train[:, -self.ncond:]
        c_val = data_val[:, -self.ncond:]
        
        data_train = np.concatenate((x_train, c_train), axis=1)
        data_val = np.concatenate((x_val, c_val), axis=1)
        
        train_data = torch.utils.data.DataLoader(self.np_to_torch(data_train), batch_size=batch_size, shuffle=False, num_workers=8, pin_memory = True)
        val_data = torch.utils.data.DataLoader(self.np_to_torch(data_val), batch_size=batch_size, shuffle=False, num_workers=8, pin_memory = True)
        
        return train_data, val_data
        
    
    def train(self, data, cond=None, n_epochs=100, batch_size=512, seed=1, outdir="./", early_stop=True, patience=5, min_delta=0.005, save_model=False, model_name = "MAF_model", s_b = 0):
        
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, n_epochs)
        
        update_epochs = 1
        val_loss_to_beat = 100000000
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        epochs, epochs_val = [], []
        losses, losses_val = [], []
        
        if early_stop:
            early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
        
        train_data, val_data = self.process_data(data=data, batch_size=batch_size, cond=cond)
        
        for epoch in tqdm(range(n_epochs), ascii=' >='):
            losses_batch_per_e = []
            
            for batch_ndx, data in enumerate(train_data):
                data = data.to(self.device)
                if cond is not None:
                    x_ = data[:, :-self.ncond] if self.nfeat != 1 else data[:,0].reshape(-1, 1)
                    c_ = data[:, -self.ncond:] if self.ncond != 1 else data[:,-1].reshape(-1, 1)
                    loss = -self.flow.log_prob(inputs=x_, context=c_).mean()
                else:
                    x_ = data
                    loss = -self.flow.log_prob(inputs=x_).mean() 
                
                losses_batch_per_e.append(loss.detach().cpu().numpy())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()  
                scheduler.step()

            epochs.append(epoch)
            mean_loss = np.mean(losses_batch_per_e)
            losses.append(mean_loss)
            
            if epoch % update_epochs == 0: # validation loss
                with torch.no_grad():

                    val_losses_batch_per_e = []

                    for batch_ndx, data in enumerate(val_data):
                        data = data.to(self.device)
                        self.optimizer.zero_grad()
                        if cond is not None:
                            x_ = data[:, :-self.ncond] if self.nfeat != 1 else data[:,0].reshape(-1, 1)
                            c_ = data[:, -self.ncond:] if self.ncond != 1 else data[:,-1].reshape(-1, 1)
                        else:
                            x_ = data
                            c_ = None
                        val_loss = -self.flow.log_prob(inputs=x_, context=c_).mean() 
                        val_losses_batch_per_e.append(val_loss.detach().cpu().numpy())
               
                    epochs_val.append(epoch)
                    mean_val_loss = np.mean(val_losses_batch_per_e)
                    losses_val.append(mean_val_loss)
                    
                    if mean_val_loss < val_loss_to_beat:
                        val_loss_to_beat = mean_val_loss
                        
                        if save_model:

                            model_path = f"{outdir}/{model_name}.pt"
                            torch.save(self, model_path)
                
                    if early_stop:
                        early_stopping(mean_val_loss)
            
            log.debug(f"Epoch: {epoch} - loss: {mean_loss} - val loss: {mean_val_loss}")
        
            if early_stop:
                if early_stopping.early_stop:
                    log.info("Early stopping.")
                    break  
                    
        log.info(f"Model stopped training at epoch {epoch}.")
        
        model_path = f"{outdir}/{model_name}.pt"
        torch.save(self, model_path)
        os.makedirs(f"{outdir}/loss_data/", exist_ok=True)
        np.savez(f"{outdir}/loss_data/{model_name}_loss", epochs = epochs, train_loss = losses, val_loss = losses_val)         
                    

    def sample(self, num_samples, cond=None):
        cond = self.np_to_torch(cond).to(self.device)
        samples_feat = self.flow.sample(num_samples=num_samples, context=cond)
        samples = samples_feat.detach().cpu().numpy()
        samples = samples.reshape(samples.shape[0], samples.shape[-1])
        #unscaled_samples = self.scaler_x.inverse_transform(samples)
        return samples
