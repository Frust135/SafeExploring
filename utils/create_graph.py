import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

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

def convolve_date(data, filter_size):
    output = np.zeros(len(data))
    for i in range(len(data)):
        ventana = data[max(0, i-filter_size+1):i+1]
        output[i] = np.mean(ventana)
    return output

def create_graph_with_average(
        name_problem,
        array_data_with_mlp, array_data_without_mlp, 
        array_data_danger_with_mlp, array_data_danger_without_mlp,        
    ):    
    data_with_mlp = get_dic_per_array(array_data_with_mlp)
    data_without_mlp = get_dic_per_array(array_data_without_mlp)
    data_danger_with_mlp = get_dic_per_array(array_data_danger_with_mlp)
    data_danger_without_mlp = get_dic_per_array(array_data_danger_without_mlp)    

    date_time = datetime.today()
    if name_problem == 'CliffWalking':
        name_file = 'figures/cliffwalking/reward-{}-{}.pdf'.format(date_time, name_problem)
        name_file2 = 'figures/cliffwalking/danger-{}-{}.pdf'.format(date_time, name_problem)
        filter_size = 10
    elif name_problem == 'CartPole':
        name_file = 'figures/cart_pole/reward-{}-{}.pdf'.format(date_time, name_problem)
        name_file2 = 'figures/cart_pole/danger-{}-{}.pdf'.format(date_time, name_problem)
        filter_size = 30

    fig, ax1 = plt.subplots(figsize=(8, 8))
    data_with_mlp_average = convolve_date(data_with_mlp['average'], filter_size)
    data_with_mlp_min = convolve_date(data_with_mlp['min'], filter_size)
    data_with_mlp_max = convolve_date(data_with_mlp['max'], filter_size)
    
    x = np.arange(len(data_with_mlp_average))

    ax1.plot(data_with_mlp_average, color='royalblue', label='Promedio de recompensas con CA')
    ax1.fill_between(x, 
        data_with_mlp_min, 
        data_with_mlp_max, 
        color='lightblue', alpha=0.2)

    data_without_mlp_average = convolve_date(data_without_mlp['average'], filter_size)
    data_without_mlp_min = convolve_date(data_without_mlp['min'], filter_size)
    data_without_mlp_max = convolve_date(data_without_mlp['max'], filter_size)

    ax1.plot(data_without_mlp_average, color='salmon', label='Promedio de recompensas')
    ax1.fill_between(x, 
        data_without_mlp_min, 
        data_without_mlp_max, 
        color='lightsalmon', alpha=0.2)
    
    ax1.set_title('Recompensa acumulada por episodio')
    ax1.set_xlabel('Episodio')
    ax1.set_ylabel('Recompensa')
    ax1.legend()
    plt.xlim(0, len(data_with_mlp['average']))
    ax1.grid(True)
    plt.savefig(name_file, format="pdf", bbox_inches="tight")

    fig, ax2 = plt.subplots(figsize=(8, 8))
    data_danger_with_mlp_average = convolve_date(data_danger_with_mlp['average'], filter_size)
    data_danger_with_mlp_min = convolve_date(data_danger_with_mlp['min'], filter_size)
    data_danger_with_mlp_max = convolve_date(data_danger_with_mlp['max'], filter_size)

    ax2.plot(x, data_danger_with_mlp_average, color='royalblue', label='Promedio de cantidad de estados peligrosos con CA')
    ax2.fill_between(x, 
        data_danger_with_mlp_min, 
        data_danger_with_mlp_max, 
        color='lightblue', alpha=0.2)
    
    data_danger_without_mlp_average = convolve_date(data_danger_without_mlp['average'], filter_size)
    data_danger_without_mlp_min = convolve_date(data_danger_without_mlp['min'], filter_size)
    data_danger_without_mlp_max = convolve_date(data_danger_without_mlp['max'], filter_size)
    
    ax2.plot(x, data_danger_without_mlp_average, color='salmon', label='Promedio de cantidad de estados peligrosos')
    ax2.fill_between(x, 
        data_danger_without_mlp_min, 
        data_danger_without_mlp_max, 
        color='lightsalmon', alpha=0.2)
    
    ax2.set_title('Cantidad de estados peligrosos acumulados por episodio')
    ax2.set_xlabel('Episodio')
    ax2.set_ylabel('Cantidad estados peligrosos')
    ax2.legend()
    ax2.grid(True)
    
    plt.xlim(0, len(data_with_mlp['average']))
    plt.savefig(name_file2, format="pdf", bbox_inches="tight")
    plt.show()