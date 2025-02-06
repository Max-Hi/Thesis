import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from typing import List, Tuple, Callable

def visual(f: Callable, ranges: List[Tuple[float, float]]) -> None:
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    if len(ranges) == 1:
        # 1D parameter case (curve)
        t = np.linspace(ranges[0][0], ranges[0][1], 1000)
        points = np.array([f(ti) for ti in t])
        ax.plot(points[:, 0], points[:, 1], points[:, 2])
    
    elif len(ranges) == 2:
        # 2D parameter case (surface)
        u = np.linspace(ranges[0][0], ranges[0][1], 100)
        v = np.linspace(ranges[1][0], ranges[1][1], 100)
        U, V = np.meshgrid(u, v)
        points = np.array([[f(ui, vi) for ui in u] for vi in v])
        X = points[:, :, 0]
        Y = points[:, :, 1]
        Z = points[:, :, 2]
        ax.plot_surface(X, Y, Z, alpha=0.8)
    
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])
    plt.show()
