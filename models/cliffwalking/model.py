import numpy as np

class SarsaModel():
    def __init__(self, n_states, n_actions):
        self.environment = Environment()
        
        self.n_states = n_states
        self.n_actions = n_actions
        #self.Q = np.zeros((self.n_states, self.n_actions))

        self.state_policy = self.create_state_policy()
        self.policy = self.initialize_policy()
        self.modify_policy_by_state()
    
    def initialize_policy(self):
        """
        Inicializa la política del agente, la matriz de transición de estados, y modifica
        las probabilidades de ocurrencia de las acciones que colisionan con los limites
        del escenario
        """
        p=(1/self.n_actions)
        policy = np.ones((self.n_states,self.n_actions))*p
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
                    if not col_index == 11:
                        if row_problem[col_index+1] > 0: matrix[int(value-1)][2] = row_problem[col_index+1]   
                    # Abajo
                    if not row_index == 3:
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
                    self.policy[row_index][col_index] = 0
                    row_aux = self.policy[row_index]
                    row_aux[row_aux!=0] = 1/len(row_aux[row_aux!=0])
                    self.policy[row_index] = row_aux
        return True
    
    
    
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