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
        state = np.digitize(observation[0], bins=np.linspace(-2, 2, self.n_states-1))
        return state

    def get_qvalue_actions(self, state):
        '''
        Retorna las probabilidades de cada acción
        '''
        current_state_row = self.Q[state-1,]
        return current_state_row

    def get_action(self, state, episode, env):
        '''
        Obtiene una acción en función del estado actual e iteración del agente, esta acción se obtiene
        de la tabla Q
        '''
        actions = self.get_qvalue_actions(state)
        new_epsilon = self.epsilon * 1 / ((episode * 0.1) + 1)
        if new_epsilon < 0.01: new_epsilon = 0.01
        if np.random.rand() < new_epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(actions)
        return action
    
    def update(self, current_state, action, reward, next_state, next_action):
        '''
        Actualiza la tabla Q
        '''
        self.Q[current_state, action] = self.Q[current_state, action] + self.alpha * \
            (reward + self.gamma * self.Q[next_state, next_action] - self.Q[current_state, action])
        return self.Q