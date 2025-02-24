import torch
from torch.utils.data import DataLoader
import numpy as np

import os, json

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
epochs = 30
learningrate = 1e-4
weight_decay = 0
form_function = datacreation.circle_9d
extra_parameters = [1] #, np.array([0.0,0.0,0.1,0.3,1.0,0.5,0.2,0.8,0.4])]
ranges = [(0,2*np.pi)] #, (0,1)]
dimensions = [9,128,128,64,64,32,16,8,4] #,2]#,2,1] 

# dataloader
print("Generating Data")
num_samples = 10_000
dataloader = DataLoader(CustomDataset(num_samples, form_function, ranges, extra_parameters), batch_size=64, shuffle=True)
print("Done.")
print()

hyperparameters = {"epochs": epochs, "learning rate": learningrate, "weight decay": weight_decay, "ranges": ranges, "dimensions": dimensions, "num_samples": num_samples}

model = Autoencoder(device, dimensions=dimensions)
model.train(epochs, dataloader, lr=learningrate, weight_decay=weight_decay)

save = input("Save [y/n]?") != "n"
save_path = ""
if save: 
    save_path = os.path.join("runs", input("Save path: "))
    os.makedirs(save_path, exist_ok=True)
    model.save(save_path)
    output_file = os.path.join(save_path, "hyperparameters.json")
    with open(output_file, "w") as f:
        json.dump(hyperparameters, f, indent=4)
        
if input("Evaluate [y/n]?") != "n":
    evaluate_form(model, num_samples, form_function, ranges, save, save_path, extra_parameters) 
    evaluate_diff(model, num_samples, form_function, ranges, save, save_path, extra_parameters)

# TODO add noise
