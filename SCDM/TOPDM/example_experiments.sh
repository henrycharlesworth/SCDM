#!/bin/bash

python trajectory_generator.py --expt_tag "traj_1" --env "EggHandOver-v0" --num_envs 20 --tau_scaler=1.1
python trajectory_generator.py --expt_tag "traj_2" --env "EggHandOver-v0" --num_envs 20 --tau_scaler=1.0

#python trajectory_generator.py --dir_name "BlockHandOver-v0" --expt_name "traj_1" --env "BlockHandOver-v0" --num_envs 20

#python trajectory_generator.py --dir_name "PenHandOver-v0" --expt_name "traj_1" --env "PenHandOver-v0" --num_envs 20

#python trajectory_generator.py --dir_name "EggCatchUnderarm-v0" --expt_name "traj_1" --env "EggCatchUnderarm-v0" --num_envs 20

#python trajectory_generator.py --dir_name "EggCatchOverarm-v0" --expt_name "traj_1" --env "EggCatchOverarm-v0-v0" --num_envs 20

#python trajectory_generator.py --dir_name "TwoEggCatchUnderArm-v0" --expt_name "traj_1" --env "TwoEggCatchUnderArm-v0" --num_envs 20 --num_samples_per_search 2000 --base_iterations 80

#python trajectory_generator.py --dir_name "PenSpin-v0" --expt_name "traj_1" --env "PenSpin-v0" --num_envs 20 --num_samples_per_search 4000 --base_iterations 80 --traj_len 250
