
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl

best_validation_loss = float('inf')

class CVAELoss(torch.nn.Module):
    def __init__(self, beta_start=0.1, beta_end=1.0, beta_steps=1000):
        super(CVAELoss, self).__init__()
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_steps = beta_steps
        self.current_step = 0

    def calculate_beta(self):
        beta = self.beta_start + (self.beta_end - self.beta_start) * min(1.0, self.current_step / self.beta_steps)
        self.current_step += 1
        return beta

    def forward(self, recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        beta = self.calculate_beta()
        total_loss = recon_loss + beta * kl_div
        return total_loss, recon_loss, kl_div, beta

class CVAE(nn.Module):
    def __init__(self, feature_dim, condition_dim, latent_dim):
        super(CVAE, self).__init__()
        self.feature_dim = feature_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim

        self.fc1 = nn.Linear(feature_dim + condition_dim, 512)
        self.fc2 = nn.Linear(512, latent_dim*2) # mu and logvar
        self.fc3 = nn.Linear(latent_dim + condition_dim, 512)
        self.fc4 = nn.Linear(512, feature_dim)

    def encode(self, x, c):
        inputs = torch.cat([x, c], 1)
        h = F.relu(self.fc1(inputs))
        return self.fc2(h).chunk(2, dim=1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, c):
        inputs = torch.cat([z, c], 1)
        h = F.relu(self.fc3(inputs))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

# Example for using the model and loss
# This example code will need to be adjusted based on your specific dataset setup and requirements

# Placeholder for loading your dataset
# Downloading the Numerai training dataset
url_train = 'https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_training_data.csv.xz'
df_train = pd.read_csv(url_train)

url_test = 'https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_validation_data.csv.xz'
df_test = pd.read_csv(url_test)

X_train = df_train.filter(regex='feature').values
Y_train = df_train['target'].values

X_test = df_test.filter(regex='feature').values
Y_test = df_test['target'].values

ohe = OneHotEncoder(handle_unknown='ignore')
era_conditions_train = ohe.fit_transform(df_train['era'].values.reshape(-1, 1)).toarray()                       
era_conditions_test = ohe.transform(df_test['era'].values.reshape(-1, 1)).toarray() 

# Convert your data into PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float)
era_conditions_train_tensor = torch.tensor(era_conditions_train, dtype=torch.float)

X_test_tensor = torch.tensor(X_test, dtype=torch.float)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float)
era_conditions_test_tensor = torch.tensor(era_conditions_test, dtype=torch.float)

# Create TensorDatasets
# Note: Assuming you want to use 'era' as condition, include it in your datasets
train_dataset = TensorDataset(X_train_tensor, era_conditions_train_tensor, Y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, era_conditions_test_tensor, Y_test_tensor)

# Create DataLoaders
# You can adjust 'batch_size' and 'shuffle' based on your specific needs
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

feature_dim = X_train_tensor.shape[1]
condition_dim = era_conditions_train_tensor.shape[1]
latent_dim = 30
num_epochs = 100

import os
import torch


# Imports the torch_xla package
import torch_xla
import torch_xla.core.xla_model as xm

device = xm.xla_device()
print(f'Running on TPU {device}')

# Example: Creating a tensor on the TPU
tensor = torch.rand(2, 2, device=device)
print(tensor)

def save_checkpoint(model, optimizer, epoch, loss, filename="cvae_checkpoint.pth"):
    """
    Saves a checkpoint of the model and optimizer.
    Args:
        model: The model to save.
        optimizer: The optimizer to save.
        epoch: Current epoch.
        loss: The loss at the current epoch.
        filename: The filename for saving the checkpoint.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(checkpoint, filename)

def train_model(rank, num_epochs=num_epochs):
    torch.set_default_dtype(torch.float32)
    device = xm.xla_device()
    
    model = CVAE(feature_dim=feature_dim, condition_dim=condition_dim, latent_dim=latent_dim).to(device)
    loss_fn = CVAELoss(beta_start=0.1, beta_end=1.0, beta_steps=5000)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Wrap the model and optimizer using PyTorch/XLA utilities
    model = xmp.MpModelWrapper(model)

    # Adjust DataLoader for TPU execution
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True
    )
    # Adjust batch size to be a multiple of the number of TPU cores
    batch_size = 128 * xm.xrt_world_size()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # Use ParallelLoader for efficient data loading across TPU cores
    para_loader = pl.ParallelLoader(train_loader, [device])
    train_loader = DataLoader(train_dataset, batch_size=128, sampler=train_sampler, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    for epoch in range(num_epochs):
        #model.train()
        train_sampler.set_epoch(epoch)
        total_loss = 0
        para_loader = pl.ParallelLoader(train_loader, [device])
        
        for x_batch, c_batch, y_batch in para_loader.per_device_loader(device):
            # Removed optimizer.zero_grad() to leverage PyTorch/XLA's built-in gradient accumulation
            recon_batch, mu, logvar = model(x_batch, c_batch)
            loss, recon_loss, kl_div, beta = loss_fn(recon_batch, x_batch, mu, logvar)
            loss.backward()
            xm.optimizer_step(optimizer, barrier=True) # Ensure optimizer step is synchronized across TPU cores
            total_loss += loss.item()

        # Print average loss for the epoch
        loss_reduced = xm.mesh_reduce('loss_reduce', total_loss, lambda x: sum(x) / len(x))
        xm.master_print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss_reduced:.4f}')

        # Evaluation step
        if epoch % 10 == 0:
            model.eval()
            validation_loss = 0
            para_loader = pl.ParallelLoader(test_loader, [device])
            for x_batch, c_batch, y_batch in para_loader.per_device_loader(device):
                recon_batch, mu, logvar = model(x_batch, c_batch)
                loss, _, _, _ = loss_fn(recon_batch, x_batch, mu, logvar)
                validation_loss += loss.item()
            validation_loss_reduced = xm.mesh_reduce('val_loss_reduce', validation_loss, lambda x: sum(x) / len(x)) 
            xm.master_print(f'Validation Loss: {validation_loss_reduced / len(test_loader):.4f}')
            
            # Save checkpoint if validation loss improved
            # Move checkpoint saving logic outside the validation loop to optimize computation
            if xm.is_master_ordinal():
                if validation_loss_reduced < best_validation_loss:
                    xm.master_print(f'Saving checkpoint at epoch {epoch+1} with validation loss {validation_loss_reduced:.4f}')
                    save_checkpoint(model, optimizer, epoch, validation_loss_reduced, filename=f"cvae_checkpoint_epoch_{epoch+1}.pth")
                    best_validation_loss = validation_loss_reduced

# Start training process using PyTorch/XLA
def _mp_fn(rank, flags):
    torch.set_default_dtype(torch.float32)
    train_model(rank, num_epochs=100)

FLAGS={}
xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')

