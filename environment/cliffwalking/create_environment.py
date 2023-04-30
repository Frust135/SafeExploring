def cliff_walking_normal(mlp=None):
    """
    Crea el entorno de CliffWalking con el agente en la posición (0, 3), y una meta en la posición (11, 3),
    y estados peligrosos desde la posición (1, 3) hasta la (10, 3)
    """
    from .cliffwalking import CliffwalkingEnviorment
    from models.cliffwalking.model import SarsaModel

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
    model_sarsa = SarsaModel(col_environment, row_environment, range_danger,
                             n_states, n_actions, initial_state, goal_state)
    data_graph_reward = []
    data_graph_danger_state = []

    for episode in range(100):
        aux_reward = 0
        aux_danger_state = 0
        # env = CliffwalkingEnviorment(
        #     px_width=768,
        #     px_height=256,
        #     player_position=[0, 3],
        #     red_flags=red_flags,
        #     green_flags=green_flags,
        # )
        rewards = []
        actions = []
        state = model_sarsa.initial_state
        action = model_sarsa.get_action(state, episode)
        for i in range(100):
            action, next_state, next_action, reward, finished = model_sarsa.run_not_controlled(
                action, state, episode, mlp)
            actions.append(action)
            rewards.append(reward)
            if reward == -100:
                aux_danger_state += 1
            action = next_action
            state = next_state
            if finished:
                break
        # env.run(actions)
        text = 'Episodio: {0} - Recompensa acumulada: {1}'.format(
            episode, sum(rewards))
        aux_reward += sum(rewards)
        data_graph_danger_state.append((episode, aux_danger_state))
        data_graph_reward.append((episode, sum(rewards)))
        # print(text)

    print('#################')
    print('Recompensa total acumulada: {0}'.format(aux_reward))
    return data_graph_reward, data_graph_danger_state


def cliff_walking_controlled():
    """
    Crea el entorno de controlado de CliffWalking
    """
    from .cliffwalking import CliffwalkingEnviorment
    from models.cliffwalking.neuronal_network import MLP
    from models.cliffwalking.model import SarsaModel
    from utils.create_graph import create_graph

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
    model_sarsa_controlled = SarsaModel(
        col_environment, row_environment, range_danger, n_states, n_actions, initial_state, goal_state)

    states_array = []
    actions_array = []
    x_locations_array = []
    y_locations_array = []
    danger_state_array = []

    for episode in range(20):
        actions = []
        # env = CliffwalkingEnviorment(
        #     px_width=384,
        #     px_height=256,
        #     player_position=[0, 3],
        #     red_flags=red_flags,
        #     green_flags=green_flags,
        # )
        states, actions, rewards, danger_state, x_locations, y_locations = model_sarsa_controlled.run_controlled(
            episode)
        # env.run(actions)
        text = 'Episodio: {0} - Recompensa acumulada: {1}'.format(
            episode, sum(rewards))
        # print(text)

        states_array.extend(states)
        actions_array.extend(actions)
        x_locations_array.extend(x_locations)
        y_locations_array.extend(y_locations)
        danger_state_array.extend(danger_state)
    data = {
        'states': states_array,
        'actions': actions_array,
        'x_locations': x_locations_array,
        'y_locations': y_locations_array,
        'danger_state': danger_state_array
    }
    mlp = MLP()
    mlp.train(data)
    data_without_mlp, data_danger_without_mlp = cliff_walking_normal()
    data_with_mlp, data_danger_with_mlp = cliff_walking_normal(mlp)
    create_graph(data_with_mlp, data_without_mlp,
                 data_danger_with_mlp, data_danger_without_mlp)
    return True
