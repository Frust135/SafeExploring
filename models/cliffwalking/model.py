import numpy as np

class SarsaModel():
    def __init__(self, n_states, n_actions):
        self.n_states = n_states
        self.n_actions = n_actions
        self.policy = self.initialize_policy()
        self.Q = np.zeros((self.n_states, self.n_actions))
        self.state_policy = []

    def create_state_policy(self, matrix_problem):
        """
        Crea una matriz en dónde indica en función del estado actual y la acción seleccionada, cuál
        será el nuevo estado del agente
        """
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
                    if not col_index == 11:
                        if row_problem[col_index+1] > 0: matrix[int(value-1)][2] = row_problem[col_index+1]   
                    # Abajo
                    if not row_index == 3:
                        if matrix_problem[row_index+1][col_index] > 0: matrix[int(value-1)][3] = matrix_problem[row_index+1][col_index]
        self.state_policy = matrix
        return matrix
    
    def initialize_policy(self):
        """
        Inicializa la política del agente, en dónde por cada estado, todas las acciones
        son igual de probables
        """
        p=(1/self.n_actions)
        policy = np.ones((self.n_states,self.n_actions))*p
        return policy
    
class Environment():
    def __init__(self):
        self.matrix_col_len = 12
        self.matrix_row_len = 4
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