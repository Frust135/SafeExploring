import numpy as np

class SarsaModel():
    def __init__(self, n_actions, n_states):        
        self.n_states = n_states
        self.n_actions = n_actions

        self.alpha = 0.1
        self.gamma = 0.99
        
        self.epsilon_final = 0.1
        self.epsilon_inicial = 1
        self.decay_rate = 0.99

        self.Q = self.initialize_policy()

    def initialize_policy(self):
        """
        Inicializa la política del agente
        """
        policy = np.zeros((self.n_states, self.n_actions))
        return policy

    def discretize_state(self, state):
        import hashlib
        arr_bytes = bytearray(state)
        hash = hashlib.sha1(arr_bytes).hexdigest()
        return int(hash[:5], 16)
    

    def get_action(self, state, episode, env):
        '''
        Obtiene una acción en función del estado actual e iteración del agente, esta acción se obtiene
        de la tabla Q
        '''
        epsilon = max(self.epsilon_final, self.epsilon_inicial * self.decay_rate**episode)
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            return np.argmax(self.Q[state])
        return action

    def update(self, current_state, action, reward, next_state, next_action):
        '''
        Actualiza la tabla Q
        '''
        self.Q[current_state, action] = self.Q[current_state, action] + self.alpha * \
                (reward + self.gamma *
                 self.Q[next_state, next_action] - self.Q[current_state, action])
        return self.Q
