import gymnasium as gym
import gym_CartPole_BT

env = gym.make(
    "CartPole-BT-v1", render_mode="human", initial_state_variance="high"
)

history = []
prior_observation, info = env.reset()
for i in range(100):
    action = [0.0]
    observation, reward, terminated, truncated, info = env.step(action)
    history.append((i, prior_observation, action, reward))
    if terminated or truncated:
        break
    prior_observation = observation
env.close()
