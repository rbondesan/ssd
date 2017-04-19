#!/bin/bash

# Script to run the train_agent.py python3 code

# Parameters:

# game
n_apples=5
n_tagged=5
width=33 # Width of screen, always odd
height=11 # Height, always odd
size_obs_ahead=15 # number of sites the players can see in front of them
size_obs_side=10 # number of sites the players can see on their side

# hyper-parameters
C_in=3 # in channel into the nn (3 for RGB)
C_H=32 # number of hidden units (or channels)
C_out=8 # out channels = number of actions.
kernel_size=5 
stride=2
gamma=.99 # gamma = discount of reward
n_episodes=1
tmax_episode=1000

# for replay_memory
capacity=2
batch_size=1

# for policy
eps_start=0.9
eps_end=0.05
decay_rate=200

args="--n_apples $n_apples --n_tagged $n_tagged --width $width --height $height --size_obs_ahead $size_obs_ahead --size_obs_side $size_obs_side --C_in $C_in --C_H $C_H --C_out $C_out --kernel_size $kernel_size --stride $stride --capacity $capacity --batch_size $batch_size --eps_start $eps_start --eps_end $eps_end --decay_rate $decay_rate --gamma $gamma --n_episodes $n_episodes --tmax_episode $tmax_episode"

echo $args

cd modules/
python3 train_agents.py $args
