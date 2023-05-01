import numpy as np

class SarsaModel():
    def __init__(self):
        self.alpha = 0.1
        self.gamma = 0.99
        
        self.epsilon_final = 0.1
        self.epsilon_inicial = 1
        self.decay_rate = 0.99

        self.poleThetaSpace = np.linspace(-0.20943951, 0.20943951, 10)
        self.poleThetaVelSpace = np.linspace(-4, 4, 10)
        self.cartPosSpace = np.linspace(-2.4, 2.4, 10)
        self.cartVelSpace = np.linspace(-4, 4, 10)

        self.Q = self.initialize_policy()

    def initialize_policy(self):
        """
        Inicializa la política del agente
        """
        states = []
        for i in range(len(self.cartPosSpace)+1):
            for j in range(len(self.cartVelSpace)+1):
                for k in range(len(self.poleThetaSpace)+1):
                    for l in range(len(self.poleThetaVelSpace)+1):
                        states.append((i,j,k,l))
        policy = {}
        for s in states:
            for a in range(2):
                policy[s, a] = 0
        return policy

    def discretize_state(self, observation):
        cartX, cartXdot, cartTheta, cartThetadot = observation
        cartX = int(np.digitize(cartX, self.cartPosSpace))
        cartXdot = int(np.digitize(cartXdot, self.cartVelSpace))
        cartTheta = int(np.digitize(cartTheta, self.poleThetaSpace))
        cartThetadot = int(np.digitize(cartThetadot, self.poleThetaVelSpace))

        return (cartX, cartXdot, cartTheta, cartThetadot)
    

    def get_action(self, state, episode, env):
        '''
        Obtiene una acción en función del estado actual e iteración del agente, esta acción se obtiene
        de la tabla Q
        '''
        epsilon = max(self.epsilon_final, self.epsilon_inicial * self.decay_rate**episode)
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            values = np.array([self.Q[state,a] for a in range(2)])
            action = np.argmax(values)
        return action

    def update(self, current_state, action, reward, next_state, next_action):
        '''
        Actualiza la tabla Q
        '''
        self.Q[current_state, action] = self.Q[current_state, action] + self.alpha * \
                (reward + self.gamma *
                 self.Q[next_state, next_action] - self.Q[current_state, action])
        return self.Q
