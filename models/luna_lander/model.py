import numpy as np


class SarsaModel():
    def __init__(self, n_states, n_actions):
        self.n_states = self.get_states()
        self.n_actions = n_actions

        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon = 1

        self.Q = self.initialize_policy()

    def get_states(self):
        high_array = [1.5, 1.5, 5, 5, 3.14, 5, 1, 1]
        increase = 1
        count = 0
        for value in high_array:
            count += int(value*increase)
            increase += 5
        return count*2

    def initialize_policy(self):
        """
        Inicializa la política del agente
        """
        # p = (1/self.n_actions)
        policy = np.zeros((self.n_states, self.n_actions))
        return policy

    def convert_state(self, observation):
        """
        Convierte el estado del agente en un entero
        """
        pass
        return observation

    def get_action(self, state, episode, env):
        '''
        Obtiene una acción en función del estado actual e iteración del agente, esta acción se obtiene
        de la tabla Q
        '''
        new_epsilon = self.epsilon * 1 / ((episode * 0.1) + 1)
        if new_epsilon < 0.01:
            new_epsilon = 0.01
        if np.random.rand() < new_epsilon:
            action = env.action_space.sample()
        else:
            return np.argmax(self.Q[state])
        return action

    def update(self, current_state, action, reward, next_state, next_action):
        '''
        Actualiza la tabla Q
        '''
        self.Q[current_state, action] += self.alpha * \
            (reward + self.gamma *
             self.Q[next_state, next_action] - self.Q[current_state, action])

        return self.Q
