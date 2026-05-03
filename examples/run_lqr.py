#!/usr/bin/env python
# Demonstration of a linear controller using full state feedback (LQR).

import argparse
import numpy as np
import gymnasium as gym
import gym_CartPole_BT

# Parse any arguments provided at the command-line
parser = argparse.ArgumentParser(description="Test this gym environment.")
parser.add_argument(
    "-e",
    "--env",
    type=str,
    default="CartPole-BT-dL-v1",
    help="gym environment",
)
parser.add_argument("-s", "--show", help="display output", action="store_true")
parser.add_argument(
    "-r", "--render", help="render animation", action="store_true"
)
args = parser.parse_args()

# Create and initialize environment
if args.show:
    print(f"\nInitializing environment '{args.env}'...")
render_mode = "human" if args.render else None
env = gym.make(args.env, render_mode=render_mode)
observation, info = env.reset()

# Get target state
xp = env.unwrapped.goal_state.reshape(4, 1)

# Control vector (shape (1,) in this case)
u = np.zeros(1)

# We will keep track of the cumulative rewards
cum_reward = 0.0

if args.show:
    print(f"{'k':>3s}  {'u':>5s} {'reward':>6s} {'cum_reward':>10s}")
    print("-" * 28)

# Gain matrix (K) for optimal control
# (Calculated using control.lqr with Q=np.eye(4), R=0.0001)
# See https://python-control.readthedocs.io/en/latest/generated/control.lqr.html
gain = np.array([-100.00, -197.54, 1491.28, 668.44])

# Run one episode
terminated = truncated = False
while not (terminated or truncated):
    # Linear quadratic regulator: u[t] = -K(x[t] - x_goal)
    u[:] = -np.dot(gain, env.unwrapped.state - env.unwrapped.goal_state)

    # Run simulation one time-step
    observation, reward, terminated, truncated, info = env.step(u)

    # Process the reward
    cum_reward += reward

    # Print updates
    if args.show:
        print(
            f"{env.unwrapped.time_step:3d}: {u[0]:5.1f} {reward:6.2f} "
            f"{cum_reward:10.1f}"
        )

if args.render:
    input("Press enter to close animation window")

env.close()
