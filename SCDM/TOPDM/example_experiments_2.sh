#!/bin/bash

python trajectory_generator.py --expt_tag "traj_1" --env "TwoEggCatchUnderArm-v0" --num_envs 20 --num_samples_per_it 2000 --num_iterations 80

python trajectory_generator.py --expt_tag "traj_1" --env "PenSpin-v0" --num_envs 20 --num_samples_per_it 4000 --num_iterations 80 --traj_len 250
