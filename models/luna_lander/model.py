import numpy as np


class SarsaModel():
    def __init__(self, n_states, n_actions):        
        self.n_states = n_states
        self.n_actions = n_actions

        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon = 1

        self.Q = self.initialize_policy()

    def initialize_policy(self):
        """
        Inicializa la política del agente
        """
        p = (1/self.n_actions)
        policy = np.ones((self.n_states, self.n_actions))*p
        return policy

    def convert_state(self, observation):
        """
        Convierte el estado del agente en un entero
        """
        return observation
    

    def get_action(self, state, episode, env):
        '''
        Obtiene una acción en función del estado actual e iteración del agente, esta acción se obtiene
        de la tabla Q
        '''
        new_epsilon = self.epsilon * 1 / ((episode * 0.1) + 1)
        if new_epsilon < 0.1:
            new_epsilon = 0.1
        if np.random.rand() < new_epsilon:
            action = env.action_space.sample()
        else:
            action = env.action_space.sample()
            #return np.argmax(self.Q[state])
        return action

    def update(self, current_state, action, reward, next_state, next_action):
        '''
        Actualiza la tabla Q
        '''
        self.Q[current_state, action] = self.Q[current_state, action] + self.alpha * \
                (reward + self.gamma *
                 self.Q[next_state, next_action] - self.Q[current_state, action])
        return self.Q
