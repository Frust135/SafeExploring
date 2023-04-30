import numpy as np
import gymnasium as gym

def luna_lander_normal():
    from models.luna_lander.model import SarsaModel
    from utils.create_graph import create_graph
    env = gym.make("Breakout-ramNoFrameskip-v4", render_mode='rgb_array')    
    n_actions = env.action_space.n
    model_sarsa = SarsaModel(n_actions, 99999999)
    state = model_sarsa.discretize_state(env.reset()[0])
    action = model_sarsa.get_action(state, 0, env)
    
    data_graph_reward = []
    for episode in range(1000):
        env.reset()
        rewards = 0
        for i in range(1000):
            raw_state, reward, terminated, truncated , info = env.step(action)
            next_action = model_sarsa.get_action(state, episode, env)
            next_state = model_sarsa.discretize_state(raw_state)

            model_sarsa.update(state, action, reward, next_state, next_action)
            state = next_state
            action = next_action
            if terminated or truncated:
                break
            rewards+=reward
        data_graph_reward.append((episode, rewards))
        print('Episode: {0} - Recompensa {1}'.format(episode,rewards))
    create_graph(data_graph_reward, [(1,1)], [(1,1)], [(1,1)])

def luna_lander_controlled():
    pass
