# Solving Complex Dexterous Manipulation Tasks with Trajectory Optimisation and Reinforcement Learning

This repository contains code for the paper: <insert link>. Results can be found on the main project page <a href="https://dexterous-manipulation.github.io">here</a>.
Requirements:
* Mujoco-py
* Pytorch
* Numpy
* joblib
* <a href="https://github.com/henrycharlesworth/dexterous-gym">Dexterous-Gym</a>

Install with ```pip install -e .```

TOPDM contains the code for the trajectory optimisation algorithm. See ```SCDM/TOPDM/example_experiments.sh``` for examples of how to run this.

TD3_plus_demos contains the code for combining demonstrations with reinforcement learning for the PenSpin task. See ```SCDM/TD3_plus_demos/run_experiment.sh``` to run.

We also provide prerun trajectories for all of the environments in ```SCDM/TOPDM/prerun_trajectories```, as well as a file to render these (```SCDM/TOPDM/prerun_trajectories/render_demonstrations.py```)
