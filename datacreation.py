import numpy as np

from typing import Union

def circle(phi: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    if isinstance(phi, float):
        return (np.cos(phi), np.sin(phi), 1)
    elif isinstance(phi, np.ndarray):
        return (np.cos(phi), np.sin(phi), np.ones(len(phi)))
    else:
        raise ValueError

def sphere(phi: Union[float, np.ndarray], theta: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return (np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta))

def bowl(phi: Union[float, np.ndarray], theta: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return (np.cos(phi)*theta, np.sin(phi)*theta, theta**2)



