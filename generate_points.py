from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from typing import Final

# ===================================
# Shared Constants (DO NOT CHANGE)
# ===================================
X_F: Final = 185.708   # centimeters, endpoint X
Y_F: Final = 10   # centimeters, endpoint Y
NUM_POINTS: Final = 50

# ============================
# Abstract Curve Generator
# ============================
class CurveGenerator(ABC):
    """
    Abstract base class for any 2D curve to be compared with the brachistochrone.
    Subclasses must implement generate_points().
    """
    @abstractmethod
    def generate_points(self) -> pd.DataFrame:
        """
        Generate points for the curve.

        Returns:
            pd.DataFrame: Columns ['X', 'Y']
        """
        pass

    def export_to_excel(self, filename: str) -> None:
        """
        Export the generated points to an Excel file.
        """
        df = self.generate_points()
        df.to_excel(filename, index=False)

# ============================
# Implementations
# ============================

class ChaseCurve(CurveGenerator):
    """
    Author: Chase Call
    Curve Type: Wavy
    """
    def __init__(self, x_f: float, y_f: float, num_points: int = 50):
        self.x_f = x_f
        self.y_f = y_f
        self.num_points = num_points

    def generate_points(self) -> pd.DataFrame:
        t_guess = 4.0
        t_f, R = JuniperMarieCurve.solve_cycloid_params(self.x_f, self.y_f, t_guess)
        t = np.linspace(0, t_f, self.num_points)

        x = R * (t - np.sin(t))
        y = self.y_f - R * (1 - np.cos(t))

        y = y + 0.75 * np.sin(2 * t)

        return pd.DataFrame({'X': x, 'Y': y})
    
# ===============================================================================

class KatieCurve(CurveGenerator):
    """
    Author: __________________
    Curve Type: ______________
    """
    def generate_points(self) -> pd.DataFrame:
        # TODO: teammate implements their own formula
        # Placeholder: all zeros (replace with real formula)
        x = np.zeros_like(x) # replace me
        y = np.zeros_like(x) # replace me
        return pd.DataFrame({'X': x, 'Y': y})
    
# ===============================================================================

class MarcCurve(CurveGenerator):
    """
    Author: Marc Escuderos Gomez
    Curve Type: Power Curve
    """
    def __init__(self, x_f: float, y_f: float, num_points: int = NUM_POINTS):
        self.x_f = x_f
        self.y_f = y_f
        self.num_points = num_points

    def generate_points(self) -> pd.DataFrame:
        x = np.linspace(0, self.x_f, self.num_points)
        u = x / self.x_f
        y = self.y_f * (1 - u)**3
        return pd.DataFrame({'X': x, 'Y': y})
        
# ===============================================================================
    
class JuniperMarieCurve(CurveGenerator):
    """
    Author: Juniper-Marie Rahal
    Curve Type: Cycloid (the main curve)
    """
    def __init__(self, x_f: float, y_f: float, num_points: int = 50):
        self.x_f = x_f
        self.y_f = y_f
        self.num_points = num_points
    
    def generate_points(self) -> pd.DataFrame:
        """
        Generate cycloid points from top-left (0, y_f) to bottom-right (x_f, 0).

        Args:
            x_f (float): horizontal distance to endpoint
            y_f (float): vertical distance to endpoint
            num_points (int): number of points to generate (default 50)

        Returns:
            pd.DataFrame: DataFrame with columns ['X', 'Y']
        """
        t_guess = 4.0
        t_f, R = JuniperMarieCurve.solve_cycloid_params(self.x_f, self.y_f, t_guess)
        t = np.linspace(0, t_f, self.num_points)
        x = R * (t - np.sin(t))
        y = self.y_f - R * (1 - np.cos(t))  # flip vertically
        df = pd.DataFrame({'X': x, 'Y': y})
        return df
    
    @staticmethod
    def solve_cycloid_params(x_f: float, y_f: float, t_guess=4.0) -> tuple[float, float]:
        """
        Solve for cycloid parameters R and t_f given endpoint (x_f, y_f).

        Args:
            x_f (float): horizontal distance to endpoint
            y_f (float): vertical distance to endpoint (top -> bottom)
            t_guess (float): initial guess for t_f (default 4.0)

        Returns:
            tuple: (t_f, R)
        """
        def eq(t):
            return (x_f / (t - np.sin(t))) - (y_f / (1 - np.cos(t)))

        t_f = fsolve(eq, t_guess)[0]
        R = x_f / (t_f - np.sin(t_f))
        return t_f, R


# ============================
# Usage
# ============================
if __name__ == "__main__":
    """
    Everyone generates their own curve.
    Once your subclass is done, uncomment the line in the 'teammates' dictionary
        with your name (and replace 'curveType' with the appropriate name in the filename)
        before running the script. After your points have been generated,
        push this updated file AND your .xlsx file to the GitHub repo.

    If you need help cloning, pushing, or pulling from GitHub, please let me know!
    --  Juniper-Marie
        01 March 2026
    """
    teammates = [
        ("results/chase_curveType_points.xlsx", ChaseCurve(X_F, Y_F)),
        # ("results/katie_curveType_points.xlsx", KatieCurve(X_F, Y_F)),
        ("results/marc_powercurve_points.xlsx", MarcCurve(X_F, Y_F)),
        ("results/junipermarie_brachistochrone_points.xlsx", JuniperMarieCurve(X_F, Y_F))
    ]

    for filename, generator in teammates:
        generator.export_to_excel(filename) 
