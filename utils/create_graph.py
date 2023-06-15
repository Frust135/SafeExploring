import matplotlib.pyplot as plt
import numpy as np
def get_average_per_array(array):
    averages = []
    transpose_data = zip(*array)

    for data in transpose_data:
        addition = sum(data)
        average = round(addition / len(array), 1)
        averages.append(average)
    return averages

def get_min_per_array(array):
    mins = []
    transpose_data = zip(*array)

    for data in transpose_data:
        min_data = min(data)
        mins.append(min_data)
    return mins

def get_max_per_array(array):
    maxs = []
    transpose_data = zip(*array)

    for data in transpose_data:
        max_data = max(data)
        maxs.append(max_data)
    return maxs

def get_dic_per_array(array):
    dic = {}
    dic['max'] = get_max_per_array(array)
    dic['average'] = get_average_per_array(array)
    dic['min'] = get_min_per_array(array)    
    return dic

def create_fits(x, data):    
    # fit_reward = np.polyfit(x, data, 2)
    # return np.poly1d(fit_reward)(x)
    return data

def create_graph_with_average(
        array_data_with_mlp, array_data_without_mlp, 
        array_data_danger_with_mlp, array_data_danger_without_mlp
    ):    
    data_with_mlp = get_dic_per_array(array_data_with_mlp)
    data_without_mlp = get_dic_per_array(array_data_without_mlp)
    data_danger_with_mlp = get_dic_per_array(array_data_danger_with_mlp)
    data_danger_without_mlp = get_dic_per_array(array_data_danger_without_mlp)
    x = np.arange(len(data_with_mlp['average']))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    

    ax1.plot(x, create_fits(x, data_with_mlp['average']), color='blue', label='Promedio con MLP')
    ax1.fill_between(x, 
        create_fits(x, data_with_mlp['min']), 
        create_fits(x, data_with_mlp['max']), 
        color='lightblue', alpha=0.2)    

    
    ax1.plot(x, create_fits(x, data_without_mlp['average']), color='salmon', label='Promedio sin MLP')
    ax1.fill_between(x, 
        create_fits(x, data_without_mlp['min']), 
        create_fits(x, data_without_mlp['max']), 
        color='lightsalmon', alpha=0.2)
    
    ax1.set_title('Relación entre Episodio y Recompensa')
    ax1.set_xlabel('Episodio')
    ax1.set_ylabel('Recompensa')
    ax1.legend()
    # ax1.set_yticks(range(0, 201, 50))
    ax1.grid(True)

    ax2.plot(x, create_fits(x, data_danger_with_mlp['average']), color='blue', label='Promedio con MLP')
    ax2.fill_between(x, 
        create_fits(x, data_danger_with_mlp['min']), 
        create_fits(x, data_danger_with_mlp['max']), 
        color='lightblue', alpha=0.2)
    
    ax2.plot(x, create_fits(x, data_danger_without_mlp['average']), color='salmon', label='Promedio sin MLP')
    ax2.fill_between(x, 
        create_fits(x, data_danger_without_mlp['min']), 
        create_fits(x, data_danger_without_mlp['max']), 
        color='lightsalmon', alpha=0.2)
    
    ax2.set_title('Relación entre Episodio y Estado Peligroso')
    ax2.set_xlabel('Episodio')
    ax2.set_ylabel('Estado Peligroso')
    ax2.legend()
    ax2.grid(True)    
        
    # plt.yticks(range(0, 201, 50))  # Establecer los ticks en rangos de 50
    plt.show()

    

    