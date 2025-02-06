import torch
from torch.utils.data import Dataset
import numpy as np

import inspect
from typing import List, Callable, Tuple

class CustomDataset(Dataset):
    def __init__(self, num_samples: int, form_function: Callable, ranges: List[Tuple[float, float]], extra_parameters: List = []):
        self.num_samples = num_samples
        self.ranges = ranges
        self.form_function = form_function
        self.extra_parameters = extra_parameters
        
        sig = inspect.signature(self.form_function)
        num_required_params = len([param for param in sig.parameters.values() if param.default == inspect.Parameter.empty and param.default is not None])
        assert len(self.ranges) == num_required_params, f"Expected {num_required_params} arguments for data creation, got {len(self.ranges)}"
        self.data = self._sample_sphere_surface()
        print(f"data has form {self.data.shape}")
    
    def _sample_sphere_surface(self):
        """Generate points uniformly sampled from the surface of a unit sphere."""
        parameters = [np.random.uniform(r[0],r[1],self.num_samples) for r in self.ranges]
        forms = self.form_function(*parameters, *self.extra_parameters)
        
        return torch.tensor(np.column_stack((forms[0], *forms[1:])), dtype=torch.float32)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]


