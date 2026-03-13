import numpy as np
from scipy.integrate import quad
from typing import Callable, Dict, Any

# =============================================================================
# PHYSICS CONSTANTS
# =============================================================================
GRAVITY = 981.0  # standard gravity in cm/s^2
DELTA_X = 15.708 # change in x coordinate
DELTA_Y = 10.0 # change in y coordinate

# Multiplier for a solid sphere rolling without slipping.
# Derived from equating potential energy to total kinetic energy:
# mg(y0 - y) = (1/2)mv^2 + (1/2)I\omega^2, where I = (2/5)mr^2
INERTIA_FACTOR = (10.0 / 7.0) * GRAVITY 

# =============================================================================
# CORE INTEGRATION ENGINE
# =============================================================================
def calculate_travel_time(
    dx_dt: Callable[[float], float],
    dy_dt: Callable[[float], float],
    y_func: Callable[[float], float],
    y0: float,
    t_start: float,
    t_end: float
) -> float:
    """
    Numerically integrates the travel time of a rolling sphere along a parametric curve.
    
    This function evaluates the integral of dt = ds / v, where ds is the 
    differential arc length and v is the velocity derived from energy conservation.

    Args:
        dx_dt (Callable): First derivative of the x(t) parametric equation.
        dy_dt (Callable): First derivative of the y(t) parametric equation.
        y_func (Callable): The y(t) parametric equation for elevation.
        y0 (float): The initial starting height (y-coordinate) of the track in cm.
        t_start (float): The starting parameter bounds for the integral.
        t_end (float): The ending parameter bounds for the integral.

    Returns:
        float: The theoretical travel time in seconds.
    """
    def integrand(t: float) -> float:
        y = y_func(t)
        dy = y0 - y
        
        # Prevent division by zero at the exact starting coordinate
        if dy <= 1e-12: 
            return 0.0
            
        ds = np.sqrt(dx_dt(t)**2 + dy_dt(t)**2)
        v = np.sqrt(INERTIA_FACTOR * dy)
        return ds / v
        
    time, _ = quad(integrand, t_start, t_end)
    return time

# =============================================================================
# TRACK DEFINITIONS
# =============================================================================
def get_cycloid_params() -> Dict[str, Any]:
    """Returns the parameters for the theoretical Brachistochrone curve."""
    return {
        "name": "Brachistochrone (Cycloid)",
        "dx_dt": lambda t: 5.0 * (1.0 - np.cos(t)),
        "dy_dt": lambda t: -5.0 * np.sin(t),
        "y_func": lambda t: 10.0 - 5.0 * (1.0 - np.cos(t)),
        "y0": 10.0,
        "t_start": 0.0,
        "t_end": np.pi
    }

def get_marc_params() -> Dict[str, Any]:
    """Returns the parameters for Marc's Cubic curve."""
    return {
        "name": "Marc's Cubic Track",
        "dx_dt": lambda t: 1.0,
        "dy_dt": lambda t: 10.0 * 3.0 * (1.0 - t/(5.0*np.pi))**2 * (-1.0/(5.0*np.pi)),
        "y_func": lambda t: 10.0 * (1.0 - t/(5.0*np.pi))**3 + 1.250,
        "y0": 11.250,
        "t_start": 0.0,
        "t_end": 5.0 * np.pi
    }

def get_chase_params() -> Dict[str, Any]:
    """Returns the parameters for Chase's Perturbed Cycloid curve."""
    return {
        "name": "Chase's Perturbed Cycloid",
        "dx_dt": lambda t: 5.007152 * (1.0 - np.cos(t)),
        "dy_dt": lambda t: -4.901882 * np.sin(t) + 0.735282 * 2.0 * np.cos(2*t),
        "y_func": lambda t: 11.053764 - 4.901882 * (1.0 - np.cos(t)) + 0.735282 * np.sin(2*t),
        "y0": 11.250,
        "t_start": 0.3,
        "t_end": np.pi
    }

def get_katie_params() -> Dict[str, Any]:
    """Returns the parameters for Katie's Exponential curve."""
    return {
        "name": "Katie's Exponential Track",
        "dx_dt": lambda t: 1.0,
        "dy_dt": lambda t: 10.003883 * (-0.5) * np.exp(-t / 2.0),
        "y_func": lambda t: 10.003883 * np.exp(-t / 2.0) + 1.246117,
        "y0": 11.250,
        "t_start": 0.0,
        "t_end": 5.0 * np.pi
    }

def calculate_straight_line_time(dx: float, dy: float) -> float:
    """
    Calculates the closed-form travel time for a straight incline.
    
    Args:
        dx (float): The total horizontal distance in cm.
        dy (float): The total vertical drop in cm.
        
    Returns:
        float: The theoretical travel time in seconds.
    """
    length = np.sqrt(dx**2 + dy**2)
    return np.sqrt((14.0 * length**2) / (5.0 * GRAVITY * dy))

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """Executes the travel time calculations for all defined tracks."""
    print("--- Theoretical Rolling Times (Solid Sphere) ---")
    
    # 1. Process all parametric curves
    tracks = [
        get_cycloid_params(),
        get_chase_params(),
        get_marc_params(),
        get_katie_params()
    ]
    
    for track in tracks:
        name = track.pop("name")
        time = calculate_travel_time(**track)
        print(f"{name:<28}: {time:.5f} s")
        
    # 2. Process the straight line separately (closed-form solution)
    straight_time = calculate_straight_line_time(dx=DELTA_X, dy=DELTA_Y)
    print(f"{'Straight Line Track':<28}: {straight_time:.5f} s")

if __name__ == "__main__":
    main()