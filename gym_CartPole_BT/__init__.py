from gymnasium.envs.registration import register

register(
    id="CartPoleBT-v1",
    entry_point="gym_CartPole_BT.envs.cartpole_bt:CartPoleBTEnv",
    max_episode_steps=100,
)
