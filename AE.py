import torch
import torch.nn as nn
import torch.optim as optim

import os
from typing import List


class Autoencoder(nn.Module):
    def __init__(self, device: str, dimensions: List[int]):
        super(Autoencoder, self).__init__()
        
        layers_e = [x for pair in zip([nn.Linear(dimensions[i], dimensions[i+1]) for i in range(len(dimensions)-1)], [nn.ReLU()]*(len(dimensions)-1)) for x in pair][:-1]
        layers_d = [x for pair in zip([nn.Linear(dimensions[i+1], dimensions[i]) for i in range(len(dimensions)-1)], [nn.ReLU()]*(len(dimensions)-1)) for x in pair][:-1]
        self.encoder = nn.Sequential(*layers_e)
        self.decoder = nn.Sequential(*layers_d[::-1])
        self.dimensions = dimensions
        
        self.device = device
        self.to(self.device)
    
    def save(self, path: str) -> None:
        state = {
            'dimensions': self.dimensions,
            'state_dict': self.state_dict()
        }
        torch.save(state, os.path.join(path, "model"))
        print(f"Model saved to {path}")

    def forward(self, x):
        x = x.to(self.device)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        encoded = self.encoder(x)
        return encoded
    
    def decode(self, y):
        decoded = self.decoder(y)
        return decoded
    
    def train(self, epochs: int, dataloader, lr: float, weight_decay: float) -> None:
        print("Training AE.")
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        # Training loop for AE
        for epoch in range(epochs):
            for data in dataloader:
                a = data
                a = a.to(self.device) 
                
                # Forward pass
                output = self.forward(a)
                loss = criterion(output, a)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
            
        print("AE Training finished!")