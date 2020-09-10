#!/bin/bash

python trajectory_generator.py --expt_tag "traj_1" --env "EggHandOver-v0" --num_envs 20

python trajectory_generator.py --expt_tag "traj_1" --env "BlockHandOver-v0" --num_envs 20

python trajectory_generator.py --expt_tag "traj_1" --env "PenHandOver-v0" --num_envs 20

python trajectory_generator.py --expt_tag "traj_1" --env "EggCatchUnderarm-v0" --num_envs 20

python trajectory_generator.py --expt_tag "traj_1" --env "EggCatchOverarm-v0" --num_envs 20

python trajectory_generator.py --expt_tag "traj_1" --env "TwoEggCatchUnderArm-v0" --num_envs 20 --num_samples_per_it 2000 --num_iterations 80

python trajectory_generator.py --expt_tag "traj_1" --env "PenSpin-v0" --num_envs 20 --num_samples_per_it 4000 --num_iterations 80 --traj_len 250