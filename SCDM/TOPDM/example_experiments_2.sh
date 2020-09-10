#!/bin/bash

python trajectory_generator.py --dir_name "TwoEggCatchUnderArm-v0" --expt_name "traj_1" --env "TwoEggCatchUnderArm-v0" --num_envs 20 --num_samples_per_search 2000 --base_iterations 80

python trajectory_generator.py --dir_name "PenSpin-v0" --expt_name "traj_1" --env "PenSpin-v0" --num_envs 20 --num_samples_per_search 4000 --base_iterations 80 --traj_len 250
