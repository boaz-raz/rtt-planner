# RRT Planner for Rectangular Robot

## Overview

This project is a Python implementation of a **Rapidly-exploring Random Tree (RRT)** planner, built from scratch. The goal is to find a collision-free path for a non-point robot in a 2D environment with obstacles.

Unlike simpler planners that treat the robot as a point, this planner models a **rectangular robot** (1.0m long, 0.5m wide). This introduces a more complex configuration space, **`(x, y, θ)`**, where `θ` (theta) is the robot's orientation. The planner must find a path that not only avoids obstacles but also ensures the robot can maneuver its body through narrow spaces without its corners colliding.

## Key Features

* **Configuration Space Planning:** The RRT tree grows in `(x, y, θ)` space, respecting the robot's orientation.

* **Oriented Bounding Box (OBB) Collision Checking:** A robust collision checker verifies if the robot's rectangular footprint (at any orientation) overlaps with any rectangular obstacles.

* **Live Animation:** The script uses `matplotlib` in interactive mode to visualize the RRT tree's growth in real-time, showing the planning process live.

* **Two Scenarios:** The planner is configured to solve two distinct environments as required by the assignment:

  1. **Narrow Passage:** A U-shaped environment that requires precise, orientation-aware maneuvers to enter and reach the goal.

  2. **Scattered Obstacles:** A more open environment with multiple obstacles that the planner must navigate around.

## Setup

## Create a virtual environment (only once)

```bash

python3 -m vG .venv
```

## Activate it

```bash

source .venv/bin/activate
```

## Install the packages you need

The only dependencies are `numpy` and `matplotlib`.

```bash

pip install -r requirements.txt
```

## Run Program

```bash

python3 rrt_planner.py
```

### Switching Scenarios

To switch between the "Narrow Passage" and "Scattered" environments, open `rrt_planner.py` and edit the `if __name__ == '__main__':` block at the **very end of the file**.

Comment one line and uncomment the other to select the scenario you wish to run:

```python

if __name__ == '__main__':

    # --- TO SUBMIT, RUN AND RECORD EACH OF THESE ---

    # Run scenario 1
    run_narrow_passage_scenario()

    # Run scenario 2 (Comment out the line above and uncomment the line below)
    # run_scattered_scenario()
```
