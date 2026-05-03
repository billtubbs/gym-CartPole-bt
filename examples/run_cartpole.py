import gymnasium as gym
from gym_CartPole_BT.envs.cartpole_bt import CartPoleBTEnv

env = CartPoleBTEnv(render_mode="human", initial_state_variance="low")

history = []
prior_observation, info = env.reset(seed=42)
for i in range(100):
    action = [0.0]
    observation, reward, terminated, truncated, info = env.step(action)
    history.append((i, prior_observation, action, reward))
    if terminated or truncated:
        break
    prior_observation = observation
env.close()
