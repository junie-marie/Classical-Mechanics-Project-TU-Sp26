import numpy as np
import pandas as pd
import pyvista as pv

# =========================
# Physics Parameters:
# =========================
g = 9.81
R = 1
theta = np.linspace(0, np.pi, 2000)

# ==================================================
# Compute Travel Time Via Numerical Integration
# ==================================================
def compute_time(x,y):
    # approximate arc length segments
    dy_dx = np.gradient(y, x)
    ds = np.sqrt(1 + dy_dx**2) * np.gradient(x)

    # velocity along curve
    v = np.sqrt(2 * g * np.abs(y))

    # avoid divide by zero at the starting point
    v[0] = v[1]

    dt = ds / v
    t_cumulative = np.cumsum(dt)
    return t_cumulative


# =========================
# Interpolation function
# =========================
def interpolate_position(x, y, t_cumulative, t_now):
    """Return x, y at a given physical time t_now using linear interpolation"""
    if t_now <= t_cumulative[0]:
        return x[0], y[0]
    elif t_now >= t_cumulative[-1]:
        return x[-1], y[-1]
        
    idx = np.searchsorted(t_cumulative, t_now)
    t0, t1 = t_cumulative[idx-1], t_cumulative[idx]
    x0, x1 = x[idx-1], x[idx]
    y0, y1 = y[idx-1], y[idx]
    alpha = (t_now - t0) / (t1 - t0)

    return x0 + alpha*(x1 - x0), y0 + alpha*(y1 - y0)

# =========================
# PyVista Scene
# =========================
def scene(c_coords, s_coords, x_c, y_c, x_s, y_s, t_c, t_s, t_anim):
    plotter = pv.Plotter(off_screen=True)
    plotter.open_movie("brachistochrone.mp4", framerate=60)

    # Add curves
    curve_c = pv.Spline(c_coords, len(x_c))
    curve_s = pv.Spline(s_coords, len(x_s))

    plotter.add_mesh(curve_c, color="green", line_width=4)
    plotter.add_mesh(curve_s, color="red", line_width=4)

    # Beads
    bead_c = pv.Sphere(radius=0.05)
    bead_s = pv.Sphere(radius=0.05)
    actor_c = plotter.add_mesh(bead_c, color="green")
    actor_s = plotter.add_mesh(bead_s, color="red")

    # view along +Z axis, Y up
    x_range = x_c.max() - x_c.min()
    y_range = y_c.max() - y_c.min()
    z_distance = max(x_range, y_range) * 2  # 1.5× scene size

    plotter.camera_position = [
        ((x_c.max() + x_c.min())/2, (y_c.max() + y_c.min())/2, z_distance),  # camera position
        ((x_c.max() + x_c.min())/2, (y_c.max() + y_c.min())/2, 0),           # look at center
        (0, 1, 0)  # flipped Y for upside down
    ]

    for i, t in enumerate(t_anim):
        # Get bead positions by interpolating physical time
        x_c_pos, y_c_pos = interpolate_position(x_c, y_c, t_c, t)
        x_s_pos, y_s_pos = interpolate_position(x_s, y_s, t_s, t)

        # Update bead positions
        actor_c.SetPosition(x_c_pos, y_c_pos, 0)
        actor_s.SetPosition(x_s_pos, y_s_pos, 0)

        # Write frame
        plotter.write_frame()

    plotter.close()

if __name__ == '__main__':
    # Cycloid
    x_c = R * (theta - np.sin(theta))
    y_c = -R * (1 - np.cos(theta))
    z_c = np.zeros_like(x_c)

    # Straight line (same endpoints)
    x_s = np.linspace(x_c[0], x_c[-1], 400)
    y_s = np.linspace(y_c[0], y_c[-1], 400)
    z_s = np.zeros_like(x_s)

    c_coords = np.column_stack((x_c, y_c, z_c))
    s_coords = np.column_stack((x_s, y_s, z_s))

    t_c = compute_time(x_c, y_c)
    t_s = compute_time(x_s, y_s)
    T_c = t_c[-1]
    T_s = t_s[-1]
    T_theory = np.pi * np.sqrt(R / g)
    tc_diff = T_c - T_theory
    percent_error = abs((T_c - T_theory) / T_theory) * 100
    advantage = T_s - T_c

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
            tc_diff,
            percent_error,
            T_s,
            advantage
        ],
        "Units": [
            "s",
            "s",
            "s",
            "%",
            "s",
            "s"
        ]
    }

    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    df.to_csv("brachistochrone_results.csv", index=False)

    t_max = max(T_c, T_s)
    t_anim = np.linspace(0, t_max, 1000)

    scene(c_coords, s_coords, x_c, y_c, x_s, y_s, t_c, t_s, t_anim)