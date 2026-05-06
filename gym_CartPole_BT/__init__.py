import math
import numpy as np
from gymnasium.envs.registration import register

_p2 = [[1, 0, 0, 0], [0, 0, 1, 0]]  # output matrix: x-position and pole angle only
_x2_initial = (-1.0, 0.0, np.pi, 0.0)  # initial state: 2 units left of goal
_x2_goal = (1.0, 0.0, np.pi, 0.0)      # goal state: 1 unit right of centre

# Measurement noise sigma vectors [x (m), x_dot (m/s), theta (rad), theta_dot (rad/s)]
_nL = [0.01, 0.03, math.radians(1.0), math.radians(0.6)]
_nH = [0.05, 0.15, math.radians(5.0), math.radians(3.0)]
# p2 output is [x, theta] only
_p2_nL = [0.01, math.radians(1.0)]
_p2_nH = [0.05, math.radians(5.0)]

# Basic cart-pendulum system (7 variants)
register(
    id="CartPole-BT-v1",
    entry_point="gym_CartPole_BT.envs.cartpole_bt:CartPoleBTEnv",
    max_episode_steps=100,
)
register(
    id="CartPole-BT-dL-v1",
    entry_point="gym_CartPole_BT.envs.cartpole_bt:CartPoleBTEnv",
    max_episode_steps=100,
    kwargs={"disturbances": "low"},
)
register(
    id="CartPole-BT-dH-v1",
    entry_point="gym_CartPole_BT.envs.cartpole_bt:CartPoleBTEnv",
    max_episode_steps=100,
    kwargs={"disturbances": "high"},
)
register(
    id="CartPole-BT-vL-v1",
    entry_point="gym_CartPole_BT.envs.cartpole_bt:CartPoleBTEnv",
    max_episode_steps=100,
    kwargs={"initial_state_variance": "low"},
)
register(
    id="CartPole-BT-vH-v1",
    entry_point="gym_CartPole_BT.envs.cartpole_bt:CartPoleBTEnv",
    max_episode_steps=100,
    kwargs={"initial_state_variance": "high"},
)
register(
    id="CartPole-BT-dL-vL-v1",
    entry_point="gym_CartPole_BT.envs.cartpole_bt:CartPoleBTEnv",
    max_episode_steps=100,
    kwargs={"disturbances": "low", "initial_state_variance": "low"},
)
register(
    id="CartPole-BT-dH-vH-v1",
    entry_point="gym_CartPole_BT.envs.cartpole_bt:CartPoleBTEnv",
    max_episode_steps=100,
    kwargs={"disturbances": "high", "initial_state_variance": "high"},
)

# Basic with measurement noise (2 variants)
register(
    id="CartPole-BT-dL-nL-v1",
    entry_point="gym_CartPole_BT.envs.cartpole_bt:CartPoleBTEnv",
    max_episode_steps=100,
    kwargs={"disturbances": "low", "measurement_noise": _nL},
)
register(
    id="CartPole-BT-dL-nH-v1",
    entry_point="gym_CartPole_BT.envs.cartpole_bt:CartPoleBTEnv",
    max_episode_steps=100,
    kwargs={"disturbances": "low", "measurement_noise": _nH},
)

# Partially observable: x-position and pole angle only (5 variants)
register(
    id="CartPole-BT-p2-v1",
    entry_point="gym_CartPole_BT.envs.cartpole_bt:CartPoleBTEnv",
    max_episode_steps=100,
    kwargs={"output_matrix": _p2},
)
register(
    id="CartPole-BT-p2-dL-v1",
    entry_point="gym_CartPole_BT.envs.cartpole_bt:CartPoleBTEnv",
    max_episode_steps=100,
    kwargs={"output_matrix": _p2, "disturbances": "low"},
)
register(
    id="CartPole-BT-p2-dH-v1",
    entry_point="gym_CartPole_BT.envs.cartpole_bt:CartPoleBTEnv",
    max_episode_steps=100,
    kwargs={"output_matrix": _p2, "disturbances": "high"},
)
register(
    id="CartPole-BT-p2-vL-v1",
    entry_point="gym_CartPole_BT.envs.cartpole_bt:CartPoleBTEnv",
    max_episode_steps=100,
    kwargs={"output_matrix": _p2, "initial_state_variance": "low"},
)
register(
    id="CartPole-BT-p2-vH-v1",
    entry_point="gym_CartPole_BT.envs.cartpole_bt:CartPoleBTEnv",
    max_episode_steps=100,
    kwargs={"output_matrix": _p2, "initial_state_variance": "high"},
)

# Partially observable with measurement noise (2 variants)
register(
    id="CartPole-BT-p2-dL-nL-v1",
    entry_point="gym_CartPole_BT.envs.cartpole_bt:CartPoleBTEnv",
    max_episode_steps=100,
    kwargs={"output_matrix": _p2, "disturbances": "low", "measurement_noise": _p2_nL},
)
register(
    id="CartPole-BT-p2-dL-nH-v1",
    entry_point="gym_CartPole_BT.envs.cartpole_bt:CartPoleBTEnv",
    max_episode_steps=100,
    kwargs={"output_matrix": _p2, "disturbances": "low", "measurement_noise": _p2_nH},
)

# Displaced initial state: 2 units left of goal (3 variants)
register(
    id="CartPole-BT-x2-v1",
    entry_point="gym_CartPole_BT.envs.cartpole_bt:CartPoleBTEnv",
    max_episode_steps=100,
    kwargs={"goal_state": _x2_goal, "initial_state": _x2_initial},
)
register(
    id="CartPole-BT-x2-dL-v1",
    entry_point="gym_CartPole_BT.envs.cartpole_bt:CartPoleBTEnv",
    max_episode_steps=100,
    kwargs={"goal_state": _x2_goal, "initial_state": _x2_initial, "disturbances": "low"},
)
register(
    id="CartPole-BT-x2-dH-v1",
    entry_point="gym_CartPole_BT.envs.cartpole_bt:CartPoleBTEnv",
    max_episode_steps=100,
    kwargs={"goal_state": _x2_goal, "initial_state": _x2_initial, "disturbances": "high"},
)

# Displaced initial state with measurement noise (2 variants)
register(
    id="CartPole-BT-x2-dL-nL-v1",
    entry_point="gym_CartPole_BT.envs.cartpole_bt:CartPoleBTEnv",
    max_episode_steps=100,
    kwargs={"goal_state": _x2_goal, "initial_state": _x2_initial, "disturbances": "low", "measurement_noise": _nL},
)
register(
    id="CartPole-BT-x2-dL-nH-v1",
    entry_point="gym_CartPole_BT.envs.cartpole_bt:CartPoleBTEnv",
    max_episode_steps=100,
    kwargs={"goal_state": _x2_goal, "initial_state": _x2_initial, "disturbances": "low", "measurement_noise": _nH},
)
