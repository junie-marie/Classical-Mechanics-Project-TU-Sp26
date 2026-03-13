# Classical-Mechanics-Project-TU-Sp26
The goal of this project is to solve the Brachistochrone Problem, either experimentally or via simulation.

## Authors
Juniper-Marie Rahal, Chase Call, Katie Mitchell, and Marc Gomez.

## Brachistochrone Problem
Given two points A and B in a vertical plane, find the shortest path for a bead sliding from A to B with no friction and under uniform gravity.
_Fun Fact: this problem was first proposed by Bernoulli in 1696!_

## Approach
Our approach to this problem:

**1. Experimental (3D Printing): "3d_models/"**

   Design and fabricate five tracks using 3D printing: one following the optimal trajectory and four varied paths.

   Use identical objects to slide down both tracks and measure the time taken.
   
**2. Computational Simulation: "simulation/"**

   Numerically model the Brachistochrone curve and a comparison trajectory.

   Simulate the motion of an object using Newtonian mechanics to compare falling times.

## Constrains
The following constraints must be considered

**3D Printing:**
1. A total material amount is given to limit excessive usage.
2. The printer has a limited run time for each usage.
3. The printer has a maximum printable size of 20cm×20cm×20cm.

**Simulation:**
1. Total computational time must be reasonable (<10mins).
2. The least accuracy of the model (numerical vs theoretical <1%).
3. The size of the output files (<1GB).
