import numpy as np
from sklearn.utils import shuffle
from sklearn.neural_network import MLPRegressor

class MLP():
    def __init__(self):
        self.model = None
    
    # def get_percentage(self, number, pct):
    #     return int(number * pct / 100)

    def parse_data_train(self, data):
        '''
        Convertir a una matriz de NumPy, estas variables son ingresadas como entradas a la red.
        Contiene los datos de entrada, las que se usaran para predecir
        '''
        X = np.stack([data['states'],data['actions'],data['x_locations'], data['y_locations']]) 
        # Vector de variable de salida, la cual se desea predecir.
        X = X.transpose()
        Y = data['danger_state']
        X, Y = shuffle(X, Y, random_state=0)
        return X, Y
        
        # # Tomamos todos los datos excepto los ultimos 20, usualmente es 80.
        # data_parsed = len(data) - self.get_percentage(len(data), 50)
        # X_train = X[:-data]
        # # Tomamos los ultimos 20 datos, usualmente son los 20 ultimos.
        # X_test = X[-data:]

        # Y_train = Y[:-data]
        # Y_test = Y[-data:]

        # return X_train, X_test, Y_train, Y_test
    
    def get_number_of_neurons(self, number, Ne, Ns):
        Nw = number/10
        Nc = (Nw - Ns) / (Ne + Ns + 1)
        return int(Nc)
    
    def train(self, data):
        #from sklearn.metrics import mean_squared_error, r2_score
        #Creamos el objeto de regresion con MLP.
        regr = MLPRegressor(hidden_layer_sizes=30, activation='logistic',random_state=None, max_iter=5000, learning_rate_init=0.01)
        
        X_train, Y_train = self.parse_data_train(data)
        #Entrenamiento del modelo.
        regr.fit(X_train, Y_train)
        # print('ccc')
        # Hacemos las predicciones que en definitiva una línea (en este caso, al ser 2D)
        # y_pred = regr.predict(X_train)
        # print("Error Cuadrático Medio: %.2f" % mean_squared_error(Y_train, y_pred))
        # Puntaje de Varianza. El mejor puntaje es un 1.0
        # print('Coeficiente de Determinación (Varianza del modelo): %.2f' % r2_score(Y_train, y_pred))        
        self.model = regr
        return True
    
    def predict_data(self, data):
        X = np.stack([data['states'],data['actions'],data['x_locations'], data['y_locations']]) 
        #X = X.transpose()
        Y_pred = self.model.predict([X])
        if Y_pred < 0.5: Y_pred = 0
        else: Y_pred = 1
        return Y_pred
        