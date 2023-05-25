import numpy as np
import math

class SarsaModel():
    def __init__(self,controlled=False):
        self.controlled = controlled

        self.alpha = 0.1
        self.gamma = 0.99
        
        self.epsilon_final = 0.001
        self.epsilon_inicial = 1
        self.decay_rate = 0.99

        self.cart_position = np.linspace(-4.8, 4.8, 1)
        self.cart_velocity = np.linspace(-4, 4, 1)
        self.pole_angle = np.linspace(0, 180, 10)
        self.pole_angular_velocity = np.linspace(-4, 4, 8)

        self.Q = self.initialize_policy()

    def initialize_policy(self):
        """
        Inicializa la política del agente
        """
        states = []
        for i in range(len(self.cart_position)+1):
            for j in range(len(self.cart_velocity)+1):
                for k in range(len(self.pole_angle)+1):
                    for l in range(len(self.pole_angular_velocity)+1):
                        states.append((i,j,k,l))
        policy = {}
        for s in states:
            for a in range(2):
                policy[s, a] = 0
        return policy
    
    def convert_degrees(self, value):
        degree_val = abs(math.degrees(value))
        if degree_val > 180:
            diff = degree_val - 180
            degree_val = 180 - diff
        return degree_val

    def discretize_state(self, observation):
        cartX, cartXdot, cartTheta, cartThetadot = observation        
        cartTheta = self.convert_degrees(cartTheta)
        cartX_state = int(np.digitize(cartX, self.cart_position))
        cartXdot_state = int(np.digitize(cartXdot, self.cart_velocity))
        cartTheta_state = int(np.digitize(cartTheta, self.pole_angle))
        cartThetadot_state = int(np.digitize(cartThetadot, self.pole_angular_velocity))
        danger_state = 0        
        if cartTheta < -20 or cartTheta > 20:
            danger_state = 1
        return (cartX_state, cartXdot_state, cartTheta_state, cartThetadot_state), danger_state, cartTheta
    

    def get_action(self, state, episode, env, mlp=None):
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
            
            if mlp:
                prediction = mlp.predict_data({'states': state, 'actions': action})
                if prediction == 1:
                    if action == 1: action = 0
                    else: action = 1
        return action

    def update(self, current_state, action, reward, next_state, next_action):
        '''
        Actualiza la tabla Q
        '''
        self.Q[current_state, action] = self.Q[current_state, action] + self.alpha * \
                (reward + self.gamma *
                 self.Q[next_state, next_action] - self.Q[current_state, action])
        return self.Q
