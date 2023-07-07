import numpy as np
from sklearn.utils import shuffle
from sklearn.neural_network import MLPRegressor, MLPClassifier


class MLP():
    def __init__(self):
        self.model = None

    def parse_data_train(self, data):
        '''
        Convertir a una matriz de NumPy, estas variables son ingresadas como entradas a la red.
        Contiene los datos de entrada, las que se usaran para predecir
        '''
        X = np.stack([data['states'], data['actions'],
                     data['x_locations'], data['y_locations']])
        X = X.transpose()
        Y = data['danger_state']
        X, Y = shuffle(X, Y, random_state=1)
        return X, Y

    def get_number_of_neurons(self, number, Ne, Ns):
        Nw = number/10
        Nc = (Nw - Ns) / (Ne + Ns + 1)
        return int(Nc)

    def train(self, data):
        hidden_layers = self.get_number_of_neurons(len(data['states']), 4, 1)
        regr = MLPClassifier(hidden_layer_sizes=190, activation='logistic',
                            random_state=1, max_iter=5000, learning_rate_init=0.01)

        X_train, Y_train = self.parse_data_train(data)
        regr.fit(X_train, Y_train)
        self.model = regr
        return True

    def predict_data(self, data):
        X = np.stack([data['states'], data['actions'],
                     data['x_locations'], data['y_locations']])
        # X = X.transpose()
        Y_pred = self.model.predict([X])
        return Y_pred
