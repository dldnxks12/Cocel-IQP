
from gymnasium.envs.registration import register
import gymnasium as gym
import environment
import sys
import os

register(
    id='InvertedQuadruplePendulum-v4',
    entry_point="environment.inverted_quadruple_pendulum_v4:InvertedQuadruplePendulumEnv",
    max_episode_steps=1000
)

"""
from gymnasium.envs.registration import register

register( 
        id = 'InvertedTriplePendulum-v4',
        entry_point = "environment.inverted_triple_pendulum_v4:InvertedTriplePendulumEnv",
        max_episode_steps = 1000
)

"""


"""
xml_file = os.getcwd()+"/environment/assets/inverted_triple_pendulum.xml"
env = gym.make("InvertedTriplePendulum-v4", model_path=xml_file)
eval_env = gym.make("InvertedTriplePendulum-v4", render_mode="human", model_path=xml_file)

for episode in range(10):
    if episode == 9:
        obs, info = eval_env.reset()
        for _ in range(1000):
            action = eval_env.action_space.sample() #policy.get_action(obs, exploration=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            if terminated or truncated:
                break
    else:
        obs, info = env.reset()
        for _ in range(1000):
            action = env.action_space.sample() #policy.get_action(obs, exploration=False)
            print(action)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
"""
xml_file = os.getcwd()+"/environment/assets/inverted_quadruple_pendulum.xml"
env = gym.make("InvertedQuadruplePendulum-v4", model_path=xml_file)
eval_env = gym.make("InvertedQuadruplePendulum-v4", render_mode="human", model_path=xml_file)

for episode in range(10):
    obs, info = eval_env.reset()
    for _ in range(1000):
        action = eval_env.action_space.sample() #policy.get_action(obs, exploration=False)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        if terminated or truncated:
            break
