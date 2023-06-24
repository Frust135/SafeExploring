import numpy as np
import gym
import warnings
import time


def cart_pole_normal(mlp=None):    
    from utils.validate_model import validate_model_cart_pole
    from models.cart_pole.model import SarsaModel
    warnings.filterwarnings("ignore")
    env = gym.make("CartPole-v1", render_mode='rgb_array')
    model_sarsa = SarsaModel()
    data_graph_reward = []
    data_graph_danger = []
    data_validate_model = []
    for episode in range(500):
        # print(episode)
        # if mlp:
        #     env = gym.make("CartPole-v1", render_mode='human')
        observation = env.reset()
        state, pole_theta = model_sarsa.discretize_state(observation[0])
        action = model_sarsa.get_action(state, 1, env, pole_theta, mlp)
        rewards = 0
        danger_states = 0
        for i in range(200):
            danger_state = get_danger_state(pole_theta, action)
            next_action = model_sarsa.get_action(state, episode, env, pole_theta, mlp)
            raw_state, reward, terminated, truncated, info = env.step(action)
            next_state, pole_theta = model_sarsa.discretize_state(raw_state)
            model_sarsa.update(state, action, reward, next_state, next_action)
            
            data_validate_model.append({
                'states': state, 
                'actions': action, 
                'pole_theta': pole_theta, 
                'danger_state': danger_state
            })
            state = next_state
            action = next_action
            rewards += reward
            danger_states += danger_state
            if pole_theta < 270 and pole_theta > 90 : break
        data_graph_reward.append(rewards)
        data_graph_danger.append(danger_states)
    if mlp:
        validate_model_cart_pole(mlp, data_validate_model)
    return data_graph_reward, data_graph_danger


def cart_pole_controlled():
    from models.cart_pole.model import SarsaModel
    from models.cart_pole.neuronal_network import MLP
    from utils.create_graph import create_graph_with_average    
    warnings.filterwarnings("ignore")
    print('### Iniciando Entrenamiento ###')
    env = gym.make("CartPole-v1", render_mode='rgb_array')
    model_sarsa = SarsaModel(controlled=True)
    states_array = []
    pole_theta_array = []
    actions_array = []
    danger_state_array = []
    for episode in range(1500):
        state, pole_theta = model_sarsa.discretize_state(env.reset()[0])
        action = model_sarsa.get_action(state, 0, env, pole_theta)
        for i in range(200):
            danger_state = get_danger_state(pole_theta, action)
            raw_state, reward, terminated, truncated, info = env.step(action)
            next_action = model_sarsa.get_action(state, episode, env, pole_theta)
            next_state, pole_theta = model_sarsa.discretize_state(raw_state)            

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
    print('### Iniciando Validación ###')
    for episode in range(10):
        data_without_mlp, data_danger_without_mlp = cart_pole_normal()
        data_with_mlp, data_danger_with_mlp = cart_pole_normal(mlp)        
        array_data_with_mlp.append(data_with_mlp)
        array_data_without_mlp.append(data_without_mlp)
        array_data_danger_with_mlp.append(data_danger_with_mlp)
        array_data_danger_without_mlp.append(data_danger_without_mlp)        
    create_graph_with_average(
        'CartPole',
        array_data_with_mlp, array_data_without_mlp, 
        array_data_danger_with_mlp, array_data_danger_without_mlp
    )
    return True


def get_danger_state(pole_theta, action):
    # Izquierda
    if action == 0:
        if pole_theta > 10 and pole_theta < 40:
            return 1
    # Derecha
    else:
        if pole_theta < 350 and pole_theta > 320:
            return 1
    return 0
