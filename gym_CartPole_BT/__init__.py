import numpy as np
from gymnasium.envs.registration import register

_p2 = [
    [1, 0, 0, 0],
    [0, 0, 1, 0],
]  # output matrix: x-position and pole angle only
_x2_initial = (-1.0, 0.0, np.pi, 0.0)  # initial state: 2 units left of goal
_x2_goal = (1.0, 0.0, np.pi, 0.0)  # goal state: 1 unit right of centre
_a1_initial = (0.0, 0.0, 0.0, 0.0)  # swing-up initial state: pole hanging
_a1_goal = (0.0, 0.0, np.pi, 0.0)   # swing-up goal state: pole upright

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
    id="CartPole-BT-dL-vL-nL-v1",
    entry_point="gym_CartPole_BT.envs.cartpole_bt:CartPoleBTEnv",
    max_episode_steps=100,
    kwargs={
        "disturbances": "low",
        "initial_state_variance": "low",
        "measurement_noise": "low",
    },
)
register(
    id="CartPole-BT-dL-vL-nH-v1",
    entry_point="gym_CartPole_BT.envs.cartpole_bt:CartPoleBTEnv",
    max_episode_steps=100,
    kwargs={
        "disturbances": "low",
        "initial_state_variance": "low",
        "measurement_noise": "high",
    },
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
    kwargs={
        "output_matrix": _p2,
        "disturbances": "low",
        "measurement_noise": "low",
    },
)
register(
    id="CartPole-BT-p2-dL-nH-v1",
    entry_point="gym_CartPole_BT.envs.cartpole_bt:CartPoleBTEnv",
    max_episode_steps=100,
    kwargs={
        "output_matrix": _p2,
        "disturbances": "low",
        "measurement_noise": "high",
    },
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
    kwargs={
        "goal_state": _x2_goal,
        "initial_state": _x2_initial,
        "disturbances": "low",
    },
)
register(
    id="CartPole-BT-x2-dH-v1",
    entry_point="gym_CartPole_BT.envs.cartpole_bt:CartPoleBTEnv",
    max_episode_steps=100,
    kwargs={
        "goal_state": _x2_goal,
        "initial_state": _x2_initial,
        "disturbances": "high",
    },
)

# Displaced initial state with measurement noise (2 variants)
register(
    id="CartPole-BT-x2-dL-nL-v1",
    entry_point="gym_CartPole_BT.envs.cartpole_bt:CartPoleBTEnv",
    max_episode_steps=100,
    kwargs={
        "goal_state": _x2_goal,
        "initial_state": _x2_initial,
        "disturbances": "low",
        "measurement_noise": "low",
    },
)
register(
    id="CartPole-BT-x2-dL-nH-v1",
    entry_point="gym_CartPole_BT.envs.cartpole_bt:CartPoleBTEnv",
    max_episode_steps=100,
    kwargs={
        "goal_state": _x2_goal,
        "initial_state": _x2_initial,
        "disturbances": "low",
        "measurement_noise": "high",
    },
)

# Swing-up: pole starts hanging (θ=0), goal is upright (θ=π), sin-sqr reward
register(
    id="CartPole-BT-r2-a1-v1",
    entry_point="gym_CartPole_BT.envs.cartpole_bt:CartPoleBTEnv",
    max_episode_steps=200,
    kwargs={
        "goal_state": _a1_goal,
        "initial_state": _a1_initial,
        "reward_function": "sinthetasqr_xsqr",
    },
)
register(
    id="CartPole-BT-r2-a1-dL-v1",
    entry_point="gym_CartPole_BT.envs.cartpole_bt:CartPoleBTEnv",
    max_episode_steps=200,
    kwargs={
        "goal_state": _a1_goal,
        "initial_state": _a1_initial,
        "reward_function": "sinthetasqr_xsqr",
        "disturbances": "low",
    },
)
register(
    id="CartPole-BT-r2-a1-dH-v1",
    entry_point="gym_CartPole_BT.envs.cartpole_bt:CartPoleBTEnv",
    max_episode_steps=200,
    kwargs={
        "goal_state": _a1_goal,
        "initial_state": _a1_initial,
        "reward_function": "sinthetasqr_xsqr",
        "disturbances": "high",
    },
)
register(
    id="CartPole-BT-r2-a1-dL-nL-v1",
    entry_point="gym_CartPole_BT.envs.cartpole_bt:CartPoleBTEnv",
    max_episode_steps=200,
    kwargs={
        "goal_state": _a1_goal,
        "initial_state": _a1_initial,
        "reward_function": "sinthetasqr_xsqr",
        "disturbances": "low",
        "measurement_noise": "low",
    },
)
register(
    id="CartPole-BT-r2-a1-dL-nH-v1",
    entry_point="gym_CartPole_BT.envs.cartpole_bt:CartPoleBTEnv",
    max_episode_steps=200,
    kwargs={
        "goal_state": _a1_goal,
        "initial_state": _a1_initial,
        "reward_function": "sinthetasqr_xsqr",
        "disturbances": "low",
        "measurement_noise": "high",
    },
)
