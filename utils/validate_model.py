import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

def validate_model_cart_pole(MLP, data_validate_model):
    dic = {}
    predicted_data = []
    original_data = []
    for data in data_validate_model:
        cartX, cartXdot, cartTheta, cartThetadot = zip(data['states'])
        X = np.stack([
            data['pole_theta'], data['actions'],
            cartTheta[0], cartThetadot[0]
        ])
        Y_pred = MLP.model.predict([X])
        predicted_data.append(Y_pred)    
        original_data.append(data['danger_state'])
    accuracy = accuracy_score(original_data, predicted_data)
    matrix = confusion_matrix(original_data, predicted_data)
    precision = precision_score(original_data, predicted_data)
    recall = recall_score(original_data, predicted_data)
    f1 = f1_score(original_data, predicted_data)   
    variance_predicted = np.var(predicted_data)
    variance_original = np.var(original_data) 
    
    dic['accuracy'] = accuracy
    dic['matrix'] = matrix
    dic['precision'] = precision
    dic['recall'] = recall
    dic['f1'] = f1
    dic['variance_predicted'] = variance_predicted
    dic['variance_original'] = variance_original
    
    print('#'*10)
    print('Accuracy: ', accuracy)
    print('Matrix: ', matrix)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('F1: ', f1)
    print('Variance predicted: ', variance_predicted)
    print('Variance original: ', variance_original)
    print('#'*10)
    return True

def validate_model_cliff_walking(MLP, data_validate_model):
    dic = {}
    predicted_data = []
    original_data = []
    for data in data_validate_model:
        X = np.stack([data['states'], data['actions'],
                    data['x_locations'], data['y_locations']])
        Y_pred = MLP.model.predict([X])
        predicted_data.append(Y_pred)    
        original_data.append(data['danger_state'])
    accuracy = accuracy_score(original_data, predicted_data)
    matrix = confusion_matrix(original_data, predicted_data)
    precision = precision_score(original_data, predicted_data)
    recall = recall_score(original_data, predicted_data)
    f1 = f1_score(original_data, predicted_data)
    variance_predicted = np.var(predicted_data)
    variance_original = np.var(original_data)
    
    dic['accuracy'] = accuracy
    dic['matrix'] = matrix
    dic['precision'] = precision
    dic['recall'] = recall
    dic['f1'] = f1
    dic['variance_predicted'] = variance_predicted
    dic['variance_original'] = variance_original
    
    print('#'*10)
    print('Accuracy: ', accuracy)
    print('Matrix: ', matrix)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('F1: ', f1)
    print('Variance predicted: ', variance_predicted)
    print('Variance original: ', variance_original)
    print('#'*10)
    return True