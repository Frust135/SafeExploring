def cliff_walking_normal(controlled_Q=None):
    """
    Crea el entorno de CliffWalking con el agente en la posición (0, 3), y una meta en la posición (11, 3),
    y estados peligrosos desde la posición (1, 3) hasta la (10, 3)
    """
    from .cliffwalking import CliffwalkingEnviorment
    from models.cliffwalking import model

    # Escenario
    red_flags = [
        [1, 3], [2, 3], [3, 3], [4, 3], [5, 3],
        [6, 3], [7, 3], [8, 3], [9, 3], [10, 3],
    ]

    green_flags = [[11, 3]]    

    # Sarsa
    n_states = 48
    n_actions = 4
    initial_state = 37
    goal_state = 48
    col_environment = 12
    row_environment = 4
    range_danger = [38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
    model_sarsa = model.SarsaModel(col_environment, row_environment, range_danger,  n_states, n_actions, initial_state, goal_state, controlled_Q)

    # states_array = []
    # actions_array = []
    # qvalues_array = []
    # finished_count = 0
    # danger_state_array = []
    for episode in range(100):
        actions = []
        env = CliffwalkingEnviorment(
            px_width=768,
            px_height=256,
            player_position=[0, 3],
            red_flags=red_flags,
            green_flags=green_flags,            
        )
        rewards, actions = model_sarsa.run(episode)
        env.run(actions)
        text = 'Episodio: {0} - Recompensa acumulada: {1}'.format(episode, sum(rewards))
        print(text)
    #     states, actions, qvalues, danger_state, rewards = model_sarsa.run_csv(episode)
    #     states_array.extend(states)
    #     actions_array.extend(actions)
    #     qvalues_array.extend(qvalues)
    #     danger_state_array.extend(danger_state)

    #     if rewards == -12: finished_count += 1
    #     if finished_count == 5: break
    # col = ['Estado', 'Acción', 'Q-Value', 'Estado Peligroso']
    # from utils import csv
    # csv.create_csv(col, states_array, actions_array, qvalues_array, danger_state_array)
    return True

def cliff_walking_controlled():
    """
    Crea el entorno de controlado de CliffWalking
    """
    from .cliffwalking import CliffwalkingEnviorment
    from models.cliffwalking import model

    # Escenario
    red_flags = [
        [1, 3], [2, 3], [3, 3], [4, 3]
    ]

    green_flags = [[5, 3]]

    # Sarsa
    n_states = 24
    n_actions = 4
    initial_state = 19
    goal_state = 24
    col_environment = 6
    row_environment = 4
    range_danger = [20, 21, 22, 23]
    model_sarsa_controlled = model.SarsaModel(col_environment, row_environment, range_danger, n_states, n_actions, initial_state, goal_state)

    states_array = []
    actions_array = []
    qvalues_array = []
    finished_count = 0
    danger_state_array = []

    for episode in range(100):
        actions = []
        env = CliffwalkingEnviorment(
            px_width=384,
            px_height=256,
            player_position=[0, 3],
            red_flags=red_flags,
            green_flags=green_flags,
        )
        # rewards, actions = model_sarsa_controlled.run(episode)
        # env.run(actions)
        # text = 'Episodio: {0} - Recompensa acumulada: {1}'.format(episode, sum(rewards))
        # print(text)
        states, actions, qvalues, danger_state, rewards = model_sarsa_controlled.run_csv(episode)
        states_array.extend(states)
        actions_array.extend(actions)
        qvalues_array.extend(qvalues)
        danger_state_array.extend(danger_state)
        print(rewards)
        if rewards == -6: finished_count += 1
        if finished_count == 5: break
    col = ['Estado', 'Acción', 'Q-Value', 'Estado Peligroso']
    from utils import csv
    csv.create_csv(col, states_array, actions_array, qvalues_array, danger_state_array)
    # cliff_walking_normal(model_sarsa_controlled.Q)
    return True