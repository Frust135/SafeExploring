import numpy as np

class QLearningModel():
    def __init__(self, n_states, n_actions):
        self.n_states = n_states
        self.n_actions = n_actions
        self.policy = self.initialize_policy()
        self.R = np.ones((self.n_states, self.n_actions))*-1
        self.Q = np.zeros((self.n_states, self.n_actions))
    
    def initialize_policy(self):
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