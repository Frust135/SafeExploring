import numpy as np


class SarsaModel():
    def __init__(self, col_len, row_len, range_danger, n_states, n_actions, initial_state, goal_state):
        self.environment = Environment(col_len, row_len)

        self.col_len = col_len
        self.row_len = row_len

        self.n_states = n_states
        self.n_actions = n_actions

        self.range_danger = range_danger

        self.alpha = 0.1
        self.gamma = 0.99

        self.epsilon_final = 0.01
        self.epsilon_inicial = 1
        self.decay_rate = 0.995

        self.initial_state = initial_state
        self.goal_state = goal_state

        self.state_policy = self.create_state_policy()
        self.Q = self.initialize_policy()
        self.modify_policy_by_state()

    def initialize_policy(self):
        """
        Inicializa la política del agente, la matriz de transición de estados, y modifica
        las probabilidades de ocurrencia de las acciones que colisionan con los limites
        del escenario
        """
        p = (1/self.n_actions)
        policy = np.ones((self.n_states, self.n_actions))*p
        return policy

    def create_state_policy(self):
        """
        Crea una matriz en dónde indica en función del estado actual y la acción seleccionada, cuál
        será el nuevo estado del agente
        """
        matrix_problem = self.environment.matrix
        matrix = np.zeros((self.n_states, self.n_actions))
        for row_index, row_problem in enumerate(matrix_problem):
            for col_index, value in enumerate(row_problem):
                if value > 0:
                    # Izquierda
                    if not col_index == 0:
                        if row_problem[col_index-1] > 0:
                            matrix[int(value-1)][0] = row_problem[col_index-1]
                    # Arriba
                    if not row_index == 0:
                        if matrix_problem[row_index-1][col_index] > 0:
                            matrix[int(
                                value-1)][1] = matrix_problem[row_index-1][col_index]
                    # Derecha
                    if not col_index == (self.col_len-1):
                        if row_problem[col_index+1] > 0:
                            matrix[int(value-1)][2] = row_problem[col_index+1]
                    # Abajo
                    if not row_index == (self.row_len-1):
                        if matrix_problem[row_index+1][col_index] > 0:
                            matrix[int(
                                value-1)][3] = matrix_problem[row_index+1][col_index]
        self.state_policy = matrix
        return matrix

    def modify_policy_by_state(self):
        '''
        Actualiza la probabilidad de las acciones considerando movimientos no permitidos,
        es decir, movimientos que colisionen con los bordes del escenario
        '''
        for row_index in range(0, self.n_states):
            for col_index in range(0, self.n_actions):
                if self.state_policy[row_index][col_index] == 0:
                    self.Q[row_index][col_index] = 0
                    row_aux = self.Q[row_index]
                    row_aux[row_aux != 0] = 1/len(row_aux[row_aux != 0])
                    self.Q[row_index] = row_aux
        return True

    def get_qvalue_actions(self, state):
        '''
        Retorna las probabilidades de cada acción
        '''
        from copy import copy
        current_state_row = copy(self.Q[state-1,])
        return current_state_row

    def get_action(self, state, episode, past_action=None):
        '''
        Obtiene una acción en función del estado actual e iteración del agente, esta acción se obtiene
        de la tabla Q
        '''
        actions = self.get_qvalue_actions(state)
        epsilon = max(self.epsilon_final, self.epsilon_inicial *
                      self.decay_rate**episode)
        if np.random.rand() < epsilon:
            # if past_action:
            #     actions[past_action] = 0
            valid_actions = np.where(actions != 0)[0]
            action = np.random.choice(valid_actions)
        else:
            while True:
                action = np.argmax(actions)
                if actions[action] == 0 or action == past_action:
                    actions[action] = np.min(actions) - 1
                else:
                    break
        return action

    def get_location(self, state):
        index = np.where(self.environment.matrix == state)
        return index[1][0], index[0][0]

    def update(self, current_state, action):
        '''
        Realiza la acción y actualiza la política, la tabla Q, y el nuevo estado
        '''
        next_state = int(self.state_policy[current_state-1, action])
        finished = False
        # Meta
        if next_state == self.goal_state:
            finished = True
            reward = 0
        # Acantilado
        elif next_state in self.range_danger:
            reward = -100
            next_state = self.initial_state
        # Otros estados
        else:
            reward = -1
        return reward, next_state, finished

    def run_not_controlled(self, action, state, episode, mlp=None):
        prediction = 0
        if mlp:
            x_location, y_location = self.get_location(state)
            data = {
                'states': state,
                'actions': action,
                'x_locations': x_location,
                'y_locations': y_location,
            }
            prediction = mlp.predict_data(data)
        if prediction == 1:
            another_action = self.get_action(state, episode, action)
            action = another_action
        reward, next_state, finished = self.update(state, action)
        next_action = self.get_action(next_state, episode)
        self.Q[state-1, action] = self.Q[state-1, action] + self.alpha * \
            (reward + self.gamma *
                self.Q[next_state-1, next_action] - self.Q[state-1, action])
        return action, next_state, next_action, reward, finished

    def run_controlled(self, episode):
        rewards = []
        actions = []
        states = []
        x_locations = []
        y_locations = []
        danger_state = []

        state = self.initial_state
        action = self.get_action(state, episode)
        for i in range(150):
            x_location, y_location = self.get_location(state)
            reward, next_state, finished = self.update(state, action)
            next_action = self.get_action(next_state, episode)
            self.Q[state-1, action] = self.Q[state-1, action] + self.alpha * \
                (reward + self.gamma *
                 self.Q[next_state-1, next_action] - self.Q[state-1, action])

            states.append(state)
            actions.append(action)
            x_locations.append(x_location)
            y_locations.append(y_location)
            rewards.append(reward)
            if reward == -100:
                danger_state.append(1)
            else:
                danger_state.append(0)
            action = next_action
            state = next_state
            if finished:
                break

        return states, actions, rewards, danger_state, x_locations, y_locations


class Environment():
    def __init__(self, col_len, row_len):
        self.matrix_col_len = col_len
        self.matrix_row_len = row_len
        self.matrix = self.create_matrix()

    def create_matrix(self):
        matrix = np.ones(shape=(self.matrix_row_len, self.matrix_col_len))
        matrix = self.define_states(matrix)
        return matrix

    def define_states(self, matrix):
        state = 1
        for row_index, row in enumerate(matrix):
            for col_index, col in enumerate(row):
                matrix[row_index][col_index] = state
                state += 1
        return matrix
