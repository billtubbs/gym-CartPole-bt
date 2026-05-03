#!/usr/bin/env python
# Demonstration of a linear controller with Kalman filter state estimator
# (LQG) for a partially-observed cart-pole system.

import argparse
import numpy as np
import gymnasium as gym
import gym_CartPole_BT

# Parse any arguments provided at the command-line
parser = argparse.ArgumentParser(description='Test this gym environment.')
parser.add_argument('-e', '--env', type=str, default='CartPole-BT-p2-dL-v1',
                    help="gym environment")
parser.add_argument('-s', "--show", help="display output",
                    action="store_true")
parser.add_argument('-r', "--render", help="render animation",
                    action="store_true")
args = parser.parse_args()

# Create and initialize environment
if args.show:
    print(f"\nInitializing environment '{args.env}'...")
render_mode = "human" if args.render else None
env = gym.make(args.env, render_mode=render_mode)

# Discrete-time state-space matrices of linear model of the cart-pole
# system in the upright position (s=1):
A = np.array([
    [ 1.00000000e+00,  4.97507794e-02,  2.49480674e-03,  4.15939085e-05],
    [ 0.00000000e+00,  9.90045685e-01,  9.97511222e-02,  2.49480674e-03],
    [ 0.00000000e+00, -1.24740337e-04,  1.00750522e+00,  5.01250418e-02],
    [ 0.00000000e+00, -4.98755611e-03,  3.00500770e-01,  1.00750522e+00]
], dtype='float32')
B = np.array([
    [0.00024922],
    [0.00995432],
    [0.00012474],
    [0.00498756]
], dtype='float32')
C = env.unwrapped.output_matrix
D = np.array([[0.], [0.]], dtype='float32')

observation, info = env.reset()

# Get target state
xp = env.unwrapped.goal_state.reshape(4, 1)

# True state - for monitoring purposes only
x = env.unwrapped.state.reshape(4, 1) - xp

# State estimates
x_est = np.zeros((4, 1))

# Initialize with observed states (unobserved states are zero)
x_est[[0, 2], :] = observation.reshape(-1, 1) - C @ xp

# Control vector (shape (1,) in this case)
u = np.zeros(1)

# We will keep track of the cumulative rewards
cum_reward = 0.0

if args.show:
    print(f"{'k':>3s}  {'x':>27s} {'x_est':>27s} {'u':>5s} {'reward':>6s} {'cum_reward':>10s}")
    print("-" * 83)

# Discrete-time Kalman filter gain matrix:
kf_gain = np.array([
    [ 1.03962720e+00,  2.07302137e-03],
    [ 7.90472757e-01,  9.32428877e-02],
    [-7.72253147e-04,  1.06420691e+00],
    [-1.75014331e-02,  1.44210117e+00]
], dtype='float32')

# Controller gain matrix (K) for optimal control:
# (Calculated using lqr function with Q=np.eye(4), and R=0.01**2)
lqr_gain = np.array([[-100., -197.5366, 1491.2808, 668.4449]])

if args.show:
    print(f"{env.unwrapped.time_step:3d}: "
          f"{np.array2string(x.T, precision=1, suppress_small=True):>27s} "
          f"{np.array2string(x_est.T, precision=1, suppress_small=True):>27s} "
          f"{u[0]:5.1f} {'-':>6s} {cum_reward:10.1f}")

# Run one episode
terminated = truncated = False
while not (terminated or truncated):

    # Compute LQR control action: u[t] = -K x_est[t]
    u[:] = -lqr_gain @ x_est

    # Output measurement
    ym = observation.reshape(2, 1) - C @ xp

    # Update Kalman filter state estimates
    error = ym - C @ x_est
    x_est = A @ x_est + B @ u.reshape(-1, 1) + kf_gain @ error

    # Run simulation one time-step
    observation, reward, terminated, truncated, info = env.step(u)

    # Get true env state - for monitoring purposes only
    x = env.unwrapped.state.reshape(4, 1) - xp

    # Process the reward
    cum_reward += reward

    # Print updates
    if args.show:
        print(f"{env.unwrapped.time_step:3d}: "
              f"{np.array2string(x.T, precision=1, suppress_small=True):>27s} "
              f"{np.array2string(x_est.T, precision=1, suppress_small=True):>27s} "
              f"{u[0]:5.1f} {reward:6.2f} {cum_reward:10.1f}")

if args.render:
    input("Press enter to close animation window")

env.close()
