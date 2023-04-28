import gym


def luna_lander_normal():
    
    from models.luna_lander import model
    env = gym.make("ALE/DemonAttack-v5", render_mode="human")

    n_actions = env.action_space.n
    n_states = env.observation_space.shape[0]
    model_sarsa = model.SarsaModel(n_states, n_actions)    
    for episode in range(2000):
        rewards = 0

        state = env.reset()
        action = model_sarsa.get_action(state, episode, env)
        for step in range(400):
            env.render()
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
