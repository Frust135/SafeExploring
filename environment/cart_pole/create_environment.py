import numpy as np
import gym
import time


def cart_pole_normal(mlp=None):
    from models.cart_pole.model import SarsaModel
    env = gym.make("CartPole-v1", render_mode='rgb_array')
    model_sarsa = SarsaModel()
    data_graph_reward = []
    data_graph_danger = []
    for episode in range(500):
        # if mlp:
        #     env = gym.make("CartPole-v1", render_mode='human')
        observation = env.reset()
        state, danger_state, pole_theta = model_sarsa.discretize_state(observation[0])
        action = model_sarsa.get_action(state, 1, env, pole_theta, mlp)
        rewards = 0
        danger_states = 0        
        for i in range(200):
            # if mlp:
            #     time.sleep(.2)
            raw_state, reward, terminated, truncated, info = env.step(action)
            next_action = model_sarsa.get_action(state, episode, env, pole_theta, mlp)
            next_state, danger_state, pole_theta = model_sarsa.discretize_state(raw_state)

            model_sarsa.update(state, action, reward, next_state, next_action)
            state = next_state
            action = next_action
            rewards += reward
            danger_states += danger_state
        data_graph_reward.append(rewards)
        data_graph_danger.append(danger_states)
        # print('Episode: {0} - Recompensa: {1} - Estado Peligroso: {2}'.format(episode, rewards, danger_states))
    # create_graph([(1,1)],data_graph_reward, [(1,1)],  data_graph_danger)
    return data_graph_reward, data_graph_danger


def cart_pole_controlled():
    from models.cart_pole.model import SarsaModel
    from models.cart_pole.neuronal_network import MLP
    from utils.create_graph import create_graph_with_average
    env = gym.make("CartPole-v1", render_mode='rgb_array')
    model_sarsa = SarsaModel(controlled=True)
    states_array = []
    pole_theta_array = []
    actions_array = []
    danger_state_array = []
    for episode in range(5000):
        state, danger_state, pole_theta = model_sarsa.discretize_state(env.reset()[0])
        action = model_sarsa.get_action(state, 0, env, pole_theta)
        for i in range(200):
            raw_state, reward, terminated, truncated, info = env.step(action)
            next_action = model_sarsa.get_action(state, episode, env, pole_theta)
            next_state, danger_state, pole_theta = model_sarsa.discretize_state(raw_state)

            model_sarsa.update(state, action, reward, next_state, next_action)
            states_array.append(state)
            actions_array.append(action)
            danger_state_array.append(danger_state)
            pole_theta_array.append(pole_theta)

            state = next_state
            action = next_action
            if pole_theta < 320 and pole_theta > 40: break
    data = {
        'states': states_array,
        'actions': actions_array,
        'pole_theta': pole_theta_array,
        'danger_state': danger_state_array
    }
    mlp = MLP()
    mlp.train(data)
    array_data_with_mlp = []
    array_data_without_mlp = []
    array_data_danger_with_mlp = []
    array_data_danger_without_mlp = []
    for episode in range(10):
        data_without_mlp, data_danger_without_mlp = cart_pole_normal()
        data_with_mlp, data_danger_with_mlp = cart_pole_normal(mlp)

        array_data_with_mlp.append(data_with_mlp)
        array_data_without_mlp.append(data_without_mlp)
        array_data_danger_with_mlp.append(data_danger_with_mlp)
        array_data_danger_without_mlp.append(data_danger_without_mlp)        
    create_graph_with_average(
        array_data_with_mlp, array_data_without_mlp, array_data_danger_with_mlp, array_data_danger_without_mlp
    )
    return True
