"""
brachistochrone_simulation.py

Simulates and visualizes the brachistochrone problem: a bead sliding under
gravity along different paths between two points.

The script compares travel time along:
    1. A cycloid (true brachistochrone solution)
    2. A straight line

Travel times are computed via numerical integration and visualized using
PyVista animation.

Outputs
-------
- MP4 animation of bead motion
- CSV file containing timing results

Dependencies
------------
numpy
pandas
pyvista

Author
------
Juniper-Marie Rahal

Last Edited
-----------
07 March 2026
"""

from typing import Tuple
import numpy as np
import pandas as pd
import pyvista as pv


# ================================================================
# Constants
# ================================================================

GRAVITY: float = 9.81


# ================================================================
# Curve Generation
# ================================================================

def cycloid_curve(radius: float, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a cycloid curve.

    Parameters
    ----------
    radius : float
        Cycloid generating circle radius.
    theta : np.ndarray
        Parameter array.

    Returns
    -------
    x : np.ndarray
        x-coordinates of the curve.
    y : np.ndarray
        y-coordinates of the curve.
    """
    x = radius * (theta - np.sin(theta))
    y = -radius * (1 - np.cos(theta))
    return x, y


def straight_line(start: Tuple[float, float],
                  end: Tuple[float, float],
                  n: int = 400) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a straight line between two points.

    Parameters
    ----------
    start : tuple
        Starting point (x0, y0).
    end : tuple
        Ending point (x1, y1).
    n : int
        Number of points.

    Returns
    -------
    x, y : np.ndarray
        Coordinates of the straight line.
    """
    x = np.linspace(start[0], end[0], n)
    y = np.linspace(start[1], end[1], n)
    return x, y


# ================================================================
# Physics Computations
# ================================================================

def compute_travel_time(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute cumulative travel time along a curve.

    Travel time is computed using numerical integration of

        dt = ds / v

    where:
        ds = arc length element
        v = sqrt(2 g y)

    Parameters
    ----------
    x, y : np.ndarray
        Curve coordinates.

    Returns
    -------
    np.ndarray
        Cumulative travel time along the curve.
    """

    dy_dx = np.gradient(y, x)
    ds = np.sqrt(1 + dy_dx**2) * np.gradient(x)

    velocity = np.sqrt(2 * GRAVITY * np.abs(y))
    velocity[0] = velocity[1]  # avoid divide-by-zero

    dt = ds / velocity
    return np.cumsum(dt)


def theoretical_cycloid_time(radius: float) -> float:
    """
    Analytical travel time for a cycloid.

    Parameters
    ----------
    radius : float

    Returns
    -------
    float
        Theoretical travel time.
    """
    return np.pi * np.sqrt(radius / GRAVITY)


# ================================================================
# Interpolation
# ================================================================

def interpolate_position(
    x: np.ndarray,
    y: np.ndarray,
    t_curve: np.ndarray,
    t_now: float
) -> Tuple[float, float]:
    """
    Interpolate particle position along a curve at a given time.

    Parameters
    ----------
    x, y : np.ndarray
        Curve coordinates.
    t_curve : np.ndarray
        Cumulative time along the curve.
    t_now : float
        Current physical time.

    Returns
    -------
    tuple
        Interpolated (x, y) position.
    """

    if t_now <= t_curve[0]:
        return x[0], y[0]

    if t_now >= t_curve[-1]:
        return x[-1], y[-1]

    idx = np.searchsorted(t_curve, t_now)

    t0, t1 = t_curve[idx - 1], t_curve[idx]
    x0, x1 = x[idx - 1], x[idx]
    y0, y1 = y[idx - 1], y[idx]

    alpha = (t_now - t0) / (t1 - t0)

    x_interp = x0 + alpha * (x1 - x0)
    y_interp = y0 + alpha * (y1 - y0)

    return x_interp, y_interp


# ================================================================
# Visualization
# ================================================================

def animate_paths(
    cycloid_coords: np.ndarray,
    straight_coords: np.ndarray,
    x_c: np.ndarray,
    y_c: np.ndarray,
    x_s: np.ndarray,
    y_s: np.ndarray,
    t_c: np.ndarray,
    t_s: np.ndarray,
    t_anim: np.ndarray,
    output_file: str = "results/brachistochrone_simulation.mp4"
):
    """
    Animate bead motion along two curves using PyVista.

    Parameters
    ----------
    cycloid_coords : np.ndarray
    straight_coords : np.ndarray
    x_c, y_c : np.ndarray
        Cycloid coordinates.
    x_s, y_s : np.ndarray
        Straight line coordinates.
    t_c, t_s : np.ndarray
        Travel time arrays.
    t_anim : np.ndarray
        Animation time samples.
    output_file : str
        Output video filename.
    """

    plotter = pv.Plotter(off_screen=True)
    plotter.open_movie(output_file, framerate=60)

    curve_c = pv.Spline(cycloid_coords, len(x_c))
    curve_s = pv.Spline(straight_coords, len(x_s))

    plotter.add_mesh(curve_c, color="green", line_width=4)
    plotter.add_mesh(curve_s, color="red", line_width=4)

    bead_c = pv.Sphere(radius=0.05)
    bead_s = pv.Sphere(radius=0.05)

    actor_c = plotter.add_mesh(bead_c, color="green")
    actor_s = plotter.add_mesh(bead_s, color="red")

    for t in t_anim:

        x_c_pos, y_c_pos = interpolate_position(x_c, y_c, t_c, t)
        x_s_pos, y_s_pos = interpolate_position(x_s, y_s, t_s, t)

        actor_c.SetPosition(x_c_pos, y_c_pos, 0)
        actor_s.SetPosition(x_s_pos, y_s_pos, 0)

        plotter.write_frame()

    plotter.close()


# ================================================================
# Results
# ================================================================

def generate_results(radius: float,
                     T_c: float,
                     T_s: float) -> pd.DataFrame:
    """
    Generate results table comparing travel times.
    """

    T_theory = theoretical_cycloid_time(radius)

    data = {
        "Quantity": [
            "Cycloid Travel Time (simulated)",
            "Cycloid Travel Time (theoretical)",
            "Difference (Simulated - Theoretical)",
            "Percent Error",
            "Straight Path Travel Time",
            "Cycloid Advantage"
        ],
        "Value": [
            T_c,
            T_theory,
            T_c - T_theory,
            abs((T_c - T_theory) / T_theory) * 100,
            T_s,
            T_s - T_c
        ],
        "Units": ["s", "s", "s", "%", "s", "s"]
    }

    return pd.DataFrame(data)


# ================================================================
# Main Program
# ================================================================

def main():
    """
    Run the brachistochrone simulation.
    """

    radius = 1
    theta = np.linspace(0, np.pi, 2000)

    x_c, y_c = cycloid_curve(radius, theta)
    x_s, y_s = straight_line((x_c[0], y_c[0]), (x_c[-1], y_c[-1]))

    z_c = np.zeros_like(x_c)
    z_s = np.zeros_like(x_s)

    cycloid_coords = np.column_stack((x_c, y_c, z_c))
    straight_coords = np.column_stack((x_s, y_s, z_s))

    t_c = compute_travel_time(x_c, y_c)
    t_s = compute_travel_time(x_s, y_s)

    T_c = t_c[-1]
    T_s = t_s[-1]

    results = generate_results(radius, T_c, T_s)

    print(results.to_string(index=False))
    results.to_csv("results/brachistochrone_results.csv", index=False)

    t_max = max(T_c, T_s)
    t_anim = np.linspace(0, t_max, 1000)

    animate_paths(
        cycloid_coords,
        straight_coords,
        x_c, y_c,
        x_s, y_s,
        t_c, t_s,
        t_anim
    )


if __name__ == "__main__":
    main()