import torch
from torch.utils.data import DataLoader
import numpy as np

import os

from AE import Autoencoder
import datacreation
from dataset import CustomDataset
from eval import evaluate_form, evaluate_diff

# Device selection
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Parameters
input_dim = 3
epochs = 1
form_function = datacreation.circle
ranges = [(0,2*np.pi)] # , (0,1)]
dimensions = [3,128,128,64,64,32,16,8,4,2] 

# dataloader
print("Generating Data")
num_samples = 10_000
dataloader = DataLoader(CustomDataset(num_samples, form_function, ranges), batch_size=64, shuffle=True)
print("Done.")
print()

model = Autoencoder(device, dimensions=dimensions)
model.train(epochs, dataloader, lr=1e-4, weight_decay=0)
save = input("Save [y/n]?") != "n"
if save: 
    save_path = os.path.join("runs", input("Save path: "))
    os.makedirs(save_path, exist_ok=True)
    model.save(save_path)
if input("Evaluate [y/n]?") != "n":
    evaluate_form(model, num_samples, form_function, ranges, save, save_path) 
    evaluate_diff(model, num_samples, form_function, ranges, save, save_path)

# TODO add noise
