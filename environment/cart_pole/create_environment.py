import numpy as np
import gym
import time


def cart_pole_normal(mlp=None):
    from models.cart_pole.model import SarsaModel
    from utils.create_graph import create_graph
    env = gym.make("CartPole-v1", render_mode='rgb_array')
    model_sarsa = SarsaModel()
    data_graph_reward = []
    data_graph_danger = []
    for episode in range(500):
        # if episode > 490:
        #     env = gym.make("CartPole-v1", render_mode='human')
        observation = env.reset()
        state, danger_state, pole_theta = model_sarsa.discretize_state(observation[0])
        action = model_sarsa.get_action(state, 1, env, mlp)
        rewards = 0
        danger_states = 0
        terminated = False
        truncated = False
        for i in range(100):
            # if episode > 490:
            #     time.sleep(.1)
            raw_state, reward, terminated, truncated, info = env.step(action)
            # if not terminated:
            #     reward = 10
            # else:
            #     reward = -1
            next_action = model_sarsa.get_action(state, episode, env, mlp)
            next_state, danger_state, pole_theta = model_sarsa.discretize_state(raw_state)

            model_sarsa.update(state, action, reward, next_state, next_action)
            state = next_state
            action = next_action
            rewards += reward
            danger_states += danger_state
            # print(danger_state)
            # if terminated: break
        data_graph_reward.append((episode, rewards))
        data_graph_danger.append((episode, danger_states))
        print('Episode: {0} - Recompensa: {1} - Estado Peligroso: {2}'.format(episode, rewards, danger_states))
    create_graph(data_graph_reward, [(1,1)], data_graph_danger, [(1,1)])
    return data_graph_reward, data_graph_danger


def cart_pole_controlled():
    from models.cart_pole.model import SarsaModel
    from models.cart_pole.neuronal_network import MLP
    from utils.create_graph import create_graph
    env = gym.make("CartPole-v1", render_mode='rgb_array')
    model_sarsa = SarsaModel(controlled=True)
    states_array = []
    actions_array = []
    danger_state_array = []
    for episode in range(500):
        state, danger_state, pole_theta = model_sarsa.discretize_state(env.reset()[0])
        action = model_sarsa.get_action(state, 0, env)
        for i in range(100):
            raw_state, reward, terminated, truncated, info = env.step(action)
            # if pole_theta > -20 and pole_theta < 20:
            #     reward = 1
            # if pole_theta < -40 or pole_theta > 40:
            #     reward = -100
            # else:
            #     reward = -1
            next_action = model_sarsa.get_action(state, episode, env)
            next_state, danger_state, pole_theta = model_sarsa.discretize_state(raw_state)

            model_sarsa.update(state, action, reward, next_state, next_action)
            states_array.append(state)
            actions_array.append(action)
            danger_state_array.append(danger_state)

            state = next_state
            action = next_action
            # if terminated: break
    data = {
        'states': states_array,
        'actions': actions_array,
        'danger_state': danger_state_array
    }
    mlp = MLP()
    mlp.train(data)
    data_without_mlp, data_danger_without_mlp = cart_pole_normal()
    data_with_mlp, data_danger_with_mlp = cart_pole_normal(mlp)
    create_graph(data_with_mlp, data_without_mlp,
                 data_danger_with_mlp, data_danger_without_mlp)
    return True
