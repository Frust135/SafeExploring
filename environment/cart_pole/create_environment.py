import numpy as np
import gymnasium as gym

def cart_pole_normal():
    from models.cart_pole.model import SarsaModel
    from utils.create_graph import create_graph
    env = gym.make("CartPole-v1", render_mode='rgb_array')    
    model_sarsa = SarsaModel()    
    data_graph_reward = []
    for episode in range(500):
        state = model_sarsa.discretize_state(env.reset()[0])
        action = model_sarsa.get_action(state, 0, env)
        rewards = 0
        for i in range(100):
            raw_state, reward, terminated, truncated , info = env.step(action)
            if terminated: reward = 1
            else: reward = -1
            next_action = model_sarsa.get_action(state, episode, env)
            next_state = model_sarsa.discretize_state(raw_state)

            model_sarsa.update(state, action, reward, next_state, next_action)
            state = next_state
            action = next_action
            if truncated:
                break
            rewards+=reward
        data_graph_reward.append((episode, rewards))
        print('Episode: {0} - Recompensa {1}'.format(episode,rewards))
    create_graph(data_graph_reward, [(1,1)], [(1,1)], [(1,1)])

def cart_pole_controlled():
    pass
