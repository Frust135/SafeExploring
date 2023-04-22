import numpy as np

class SarsaModel():
    def __init__(self, col_len, row_len, range_danger, n_states, n_actions, initial_state, goal_state, controlled_Q=None):
        self.environment = Environment(col_len, row_len)

        self.col_len = col_len
        self.row_len = row_len

        self.n_states = n_states
        self.n_actions = n_actions

        self.range_danger = range_danger

        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon = 0.1

        self.initial_state = initial_state
        self.goal_state = goal_state
        
        if controlled_Q is None: self.controlled_Q = np.empty((0, 0))
        else: self.controlled_Q = controlled_Q

        self.state_policy = self.create_state_policy()
        self.Q = self.initialize_policy()
        self.modify_policy_by_state()        
    
    def initialize_policy(self):
        """
        Inicializa la política del agente, la matriz de transición de estados, y modifica
        las probabilidades de ocurrencia de las acciones que colisionan con los limites
        del escenario
        """
        p=(1/self.n_actions)
        policy = np.ones((self.n_states,self.n_actions))*p
        if self.controlled_Q.any():
            len_controlled_Q = len(self.controlled_Q)
            policy[:len_controlled_Q, :] = self.controlled_Q
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
                        if row_problem[col_index-1] > 0: matrix[int(value-1)][0] = row_problem[col_index-1]
                    # Arriba
                    if not row_index == 0:
                        if matrix_problem[row_index-1][col_index] > 0: matrix[int(value-1)][1] = matrix_problem[row_index-1][col_index]
                    # Derecha
                    if not col_index == (self.col_len-1):
                        if row_problem[col_index+1] > 0: matrix[int(value-1)][2] = row_problem[col_index+1]   
                    # Abajo
                    if not row_index == (self.row_len-1):
                        if matrix_problem[row_index+1][col_index] > 0: matrix[int(value-1)][3] = matrix_problem[row_index+1][col_index]
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
                    row_aux[row_aux!=0] = 1/len(row_aux[row_aux!=0])
                    self.Q[row_index] = row_aux        
        return True
    
    def get_prob_actions(self, state):
        '''
        Retorna las probabilidades de cada acción
        '''
        current_state_row = self.Q[state-1,]                
        return current_state_row

    def get_action(self, state, episode):
        '''
        Obtiene una acción en función del estado actual e iteración del agente, esta acción se obtiene
        de la tabla Q
        '''
        prob_actions = self.get_prob_actions(state)
        q_values_list = prob_actions + np.random.randn(1, self.n_actions) * (1.0 / (episode + 1))
        while True:
            action = np.argmax(q_values_list)
            if prob_actions[action] == 0:
                q_values_list[0][action] = np.min(q_values_list[0]) - 1
            else:
                break 
        return action

    def update(self, current_state, action):
        '''
        Realiza la acción y actualiza la política, la tabla Q, y el nuevo estado
        '''
        next_state = int(self.state_policy[current_state-1, action])
        finished = False
        # Meta
        if next_state == self.goal_state:
            finished = True
            reward = 100
        # Acantilado
        elif next_state in self.range_danger: 
            reward = -100
            next_state = self.initial_state
        # Otros estados
        else: 
            reward = -1
        return reward, next_state, finished
    
    def run(self, episode):
        rewards = []
        actions = []
        state = self.initial_state
        action = self.get_action(state, episode)
        for i in range(500):
            actions.append(action)
            reward, next_state, finished = self.update(state, action)
            next_action = self.get_action(next_state, episode)
            self.Q[state-1, action] = self.Q[state-1, action] + self.alpha * (reward + self.gamma * self.Q[next_state-1, next_action] - self.Q[state-1, action])
            action = next_action
            state = next_state
            if finished: break
            rewards.append(reward)
        return rewards, actions
    
    def run_csv(self, episode):
        rewards = []
        actions = []
        states = []
        qvalues = []
        danger_state = []
        
        state = self.initial_state
        action = self.get_action(state, episode)
        for i in range(150):            
            reward, next_state, finished = self.update(state, action)
            next_action = self.get_action(next_state, episode)
            self.Q[state-1, action] = self.Q[state-1, action] + self.alpha * (reward + self.gamma * self.Q[next_state-1, next_action] - self.Q[state-1, action])
            
            states.append(state)
            actions.append(action)
            qvalues.append(self.Q[state-1, action])
            if reward == -100:
                danger_state.append('danger')
            else:
                danger_state.append('safe')

            action = next_action
            state = next_state            
            
            if finished: break
            rewards.append(reward)

        return states, actions, qvalues, danger_state, sum(rewards)
    
    
    
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
                state+=1
        return matrix