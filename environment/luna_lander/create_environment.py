def luna_lander_normal():
    from .luna_lander import LunaLanderEnviorment
    for episode in range(150):
        env = LunaLanderEnviorment()
        rewards, actions = env.run(episode)
        text = 'Episodio: {0} - Recompensa acumulada: {1}'.format(
            episode, rewards)
        print(text)
    return True
