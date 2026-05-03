import numpy as np
from gymnasium.envs.registration import register

_p2 = [[1, 0, 0, 0], [0, 0, 1, 0]]  # output matrix: x-position and pole angle only
_x2 = (-2.0, 0.0, np.pi, 0.0)       # initial state: 2 units left of goal

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

# Displaced initial state: 2 units left of goal (3 variants)
register(
    id="CartPole-BT-x2-v1",
    entry_point="gym_CartPole_BT.envs.cartpole_bt:CartPoleBTEnv",
    max_episode_steps=100,
    kwargs={"initial_state": _x2},
)
register(
    id="CartPole-BT-x2-dL-v1",
    entry_point="gym_CartPole_BT.envs.cartpole_bt:CartPoleBTEnv",
    max_episode_steps=100,
    kwargs={"initial_state": _x2, "disturbances": "low"},
)
register(
    id="CartPole-BT-x2-dH-v1",
    entry_point="gym_CartPole_BT.envs.cartpole_bt:CartPoleBTEnv",
    max_episode_steps=100,
    kwargs={"initial_state": _x2, "disturbances": "high"},
)
