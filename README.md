# Solving Complex Dexterous Manipulation Tasks with Trajectory Optimisation and Reinforcement Learning

This repository contains code for our ICML 2021 paper: <a href="https://arxiv.org/abs/2009.05104">Solving Complex Dexterous Manipulation Tasks with Trajectory Optimisation and Reinforcement Learning</a> (link to arXiv version). Videos showcasing the obtained results can be found on the <a href="https://dexterous-manipulation.github.io">main project page</a>.
Requirements:
* Mujoco-py
* Pytorch
* Numpy
* joblib
* <a href="https://github.com/henrycharlesworth/dexterous-gym">Dexterous-Gym</a>

Install with ```pip install -e .```

TOPDM contains the code for the trajectory optimisation algorithm. See ```SCDM/TOPDM/example_experiments.sh``` for examples of how to run this. Note that this cleaned version of the code seems to be running more slowly than an earlier version - currently looking into this.

TD3_plus_demos contains the code for combining demonstrations with reinforcement learning for the PenSpin task. See ```SCDM/TD3_plus_demos/run_experiment.sh``` to run.

We also provide prerun trajectories for all of the environments in ```SCDM/TOPDM/prerun_trajectories```, as well as a file to render these (```SCDM/TOPDM/prerun_trajectories/render_demonstrations.py```)

We later on also added a version of TOPDM applied to the Humanoid-v3 environment in OpenAI's gym. This is contained in SCDM/TOPDM/humanoid_experiments.
