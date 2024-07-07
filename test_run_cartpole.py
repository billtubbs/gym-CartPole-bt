import gymnasium as gym
from gym_CartPole_BT.envs.cartpole_bt import CartPoleBTEnv
env_config = {'render_mode': 'human'}
env = CartPoleBTEnv(**env_config)

env.render()

history = []
prior_observation, info = env.reset(seed=42)
for i in range(100):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    history.append((i, prior_observation, action, reward))
    env.render()
    if terminated or truncated:
        break
    prior_observation = observation
env.close()
