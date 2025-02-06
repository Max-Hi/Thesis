import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from sklearn.decomposition import PCA

import inspect
import os
from typing import List, Callable, Tuple

from visual import visual


def evaluate_diff(model, num_samples: int, form_function: Callable, ranges: List[Tuple[float]], save: bool, save_path: str, extra_parameters: List = []) -> None:
    
    sig = inspect.signature(form_function)
    num_required_params = len([param for param in sig.parameters.values() if param.default == inspect.Parameter.empty])
    assert len(ranges) == num_required_params, f"Expected {num_required_params} arguments for evaluation, got {len(ranges)}"
    
    parameters = [np.random.uniform(r[0],r[1],num_samples) for r in ranges]
    forms = form_function(*parameters, *extra_parameters)
    
    
    samples = torch.tensor(np.column_stack((forms[0], *forms[1:])), dtype=torch.float32)
    
    outputs = model.forward(samples).to("cpu")
    
    heat_values = torch.norm(samples - outputs, dim=1).detach().numpy()

    # Create 2D grid for heatmap #TODO
    if len(ranges) == 1:
        parameter = parameters[0]
        
        sorted_indices = np.argsort(parameter)
        parameter = parameter[sorted_indices]
        heat_values = heat_values[sorted_indices]
        
        # Plot
        plt.figure(figsize=(8, 6))
        plt.plot(parameter, heat_values)
        plt.xlabel("Parameter")
        plt.ylabel("|Sample - Output|")
        plt.title('Approximation Error')
        
    else:
        grid_theta, grid_phi = np.meshgrid(np.linspace(ranges[0][0], ranges[0][1]), np.linspace(ranges[1][0], ranges[1][1]))
        points = np.column_stack((parameters[0], parameters[1]))
        grid_heat = griddata((points), heat_values, (grid_theta, grid_phi), method='linear')

        # Plot heatmap
        plt.figure(figsize=(8, 6))
        plt.pcolormesh(grid_theta, grid_phi, grid_heat, shading='auto', cmap='inferno')
        plt.colorbar(label='|Sample - Output|')
        plt.xlabel("Parameter 1")
        plt.ylabel("Parameter 2")
        plt.title('Approximation Error')
    
    if save:
        np.savez(os.path.join(save_path, "diff_evaluation.npz"), parameters=parameters, samples=samples.detach().numpy(), outputs=outputs.detach().numpy(), heat_values=heat_values)
        plt.savefig(os.path.join(save_path, "diff_plot.png"))
        print(f"Diff evaluation data and plot saved to {save_path}")
    
    plt.show()
    
    
def evaluate_form(model, num_samples: int, form_function: Callable, ranges: List[Tuple[float]], save: bool, save_path: str, extra_parameters: List = []) -> None:
    
    sig = inspect.signature(form_function)
    num_required_params = len([param for param in sig.parameters.values() if param.default == inspect.Parameter.empty])
    assert len(ranges) == num_required_params, f"Expected {num_required_params} arguments for evaluation, got {len(ranges)}"
    
    parameters = [np.random.uniform(r[0],r[1],num_samples) for r in ranges]
    forms = form_function(*parameters, *extra_parameters)
    
    
    samples = torch.tensor(np.column_stack((forms[0], *forms[1:])), dtype=torch.float32)
    outputs = model.forward(samples).to("cpu").detach().numpy()
    samples = samples.detach().numpy()
    
    if samples.shape[1] > 3:
        pca = PCA(n_components=3)
        samples = pca.fit_transform(samples)
        outputs = pca.transform(outputs)
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(outputs[:,0], outputs[:,1], outputs[:,2], alpha=0.2, s=1)
    ax.scatter(samples[:,0], samples[:,1], samples[:,2], alpha=0.05, s=1, c="black")
    
    if save:
        np.savez(os.path.join(save_path + "/form_evaluation.npz"), parameters=parameters, samples=samples, outputs=outputs)
        plt.savefig(os.path.join(save_path + "/form_plot.png"))
        print(f"Form evaluation data and plot saved to {save_path}")
    
    plt.show()

    