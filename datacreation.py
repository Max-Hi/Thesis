import numpy as np

from typing import Union, Tuple

def circle(phi: Union[float, np.ndarray], r: float = 1) -> Union[Tuple, np.ndarray]:
    """
    Parametric equation of a circle at hight z=1.
    
    Args:
        phi: Angle in the circle
        r: Circle radius
    
    Returns:
        Tuple of (x, y, z) coordinates
    """
    if isinstance(phi, float):
        return (r*np.cos(phi), r*np.sin(phi), 1)
    elif isinstance(phi, np.ndarray):
        return (r*np.cos(phi), r*np.sin(phi), np.ones(len(phi)))
    else:
        raise ValueError

def circle_9d(phi: Union[float, np.ndarray], r: float = 1, rotation_angles: np.ndarray = None) -> Union[Tuple, np.ndarray]:
    """
    Parametric equation of a circle embedded in 9D space with optional rotation.
    
    Args:
        phi: Angle in the circle
        r: Circle radius
        rotation_angles: Array of angles for rotation in each plane (optional)
    
    Returns:
        9D coordinates of the circle point(s)
    """
    if rotation_angles is None:
        rotation_angles = np.array([0,0,0,0,0,0,0,0,0])
    
    # Initialize base circle in first two dimensions, rest set to 0
    if isinstance(phi, float):
        point = np.array([r*np.cos(phi), r*np.sin(phi), 0, 0, 0, 0, 0, 0, 0])
    elif isinstance(phi, np.ndarray):
        point = np.column_stack((
            r*np.cos(phi),
            r*np.sin(phi),
            np.zeros((len(phi), 7))
        ))
    else:
        raise ValueError("phi must be float or ndarray")
    
    # Apply rotation if specified
    if (rotation_angles!=0).all():
        # Create block diagonal rotation matrix
        rotation_matrix = np.eye(9)
        for i in range(0, 8, 2):
            if i//2 < len(rotation_angles): # TODO needed?
                angle = rotation_angles[i//2]
                c, s = np.cos(angle), np.sin(angle)
                rotation_matrix[i:i+2, i:i+2] = [[c, -s], [s, c]]
        
        # Apply rotation
        if isinstance(phi, float):
            point = rotation_matrix @ point
        else:
            point = rotation_matrix @ point
            
    return point.T

def sphere(phi: Union[float, np.ndarray], theta: Union[float, np.ndarray], r: float = 1) -> Union[float, np.ndarray]:
    """
    Parametric equation of a sphere.
    
    Args:
        phi, theta: the two angles characterizing the sphere in polar coordinates
        r: Sphere radius
    
    Returns:
        Tuple of (x, y, z) coordinates
    """
    return (r*np.cos(phi)*np.sin(theta), r*np.sin(phi)*np.sin(theta), r*np.cos(theta))

def bowl(phi: Union[float, np.ndarray], z: Union[float, np.ndarray], r: float = 1) -> Union[float, np.ndarray]:
    """
    Parametric equation of a bowl like structure that is parabolic in z coordinate and circular in the xy plane.
    
    Args:
        phi: Angle around the tube's cross-section (minor circle)
        z: z coordinate
        r: Circle radius at z=1
    
    Returns:
        Tuple of (x, y, z) coordinates
    """
    return (r*np.cos(phi)*z, r*np.sin(phi)*z, z**2)

def torus(phi: Union[float, np.ndarray], theta: Union[float, np.ndarray], R: float = 2, r: float = 1) -> Union[float, np.ndarray]:
    """
    Parametric equation of a torus.
    
    Args:
        phi: Angle around the tube's cross-section (minor circle)
        theta: Angle around the major circle
        R: Major radius (distance from the center of the tube to the center of the torus)
        r: Minor radius (radius of the tube)
    
    Returns:
        Tuple of (x, y, z) coordinates
    """
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    
    return (x, y, z)


