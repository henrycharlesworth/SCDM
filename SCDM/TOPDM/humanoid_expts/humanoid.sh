#!/bin/bash

python trajectory_generator.py --expt_tag "results" --env "Humanoid-v3" --num_envs 12 --tau 40 --num_iterations 40 --num_samples_per_it 1020 --traj_len 1000 --not-goal-based --mask-done --tau_scaler 1.02
