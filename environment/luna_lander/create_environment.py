import gym


def luna_lander_normal():
    from models.luna_lander import model
    env = gym.make("LunarLander-v2", render_mode="human",
                   enable_wind=True, wind_power=15)

    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    model_sarsa = model.SarsaModel(n_states, n_actions)
    for episode in range(500):
        rewards = 0

        observation = env.reset()
        state = 3
        action = model_sarsa.get_action(state, episode, env)
        for step in range(400):
            next_observation, reward, terminated, truncated, info = env.step(action)
            next_action = model_sarsa.get_action(state, episode, env)
            next_state = model_sarsa.convert_state(next_observation)
            model_sarsa.update(state, action, reward, next_state, next_action)
            if terminated or truncated:
                break
            
            action = next_action
            state = next_state            
            rewards += reward
        text = 'Episodio: {0} - Recompensa acumulada: {1}'.format(
            episode, rewards)
        print(text)
    return True


def luna_lander_controlled():
    pass
