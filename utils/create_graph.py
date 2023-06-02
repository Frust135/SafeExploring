def create_graph(data_with_mlp, data_without_mlp, data_danger_with_mlp, data_danger_without_mlp):
    import matplotlib.pyplot as plt
    import numpy as np
    x_mlp_reward, y_mlp_reward = zip(*data_with_mlp)
    x_reward, y_reward = zip(*data_without_mlp)

    x_mlp_danger, y_mlp_danger = zip(*data_danger_with_mlp)
    x_danger, y_danger = zip(*data_danger_without_mlp)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Gráfico 1    
    # ax1.plot(x_reward, y_reward, color='green', label='sin MLP')
    fit_reward = np.polyfit(x_reward, y_reward, 2)
    ax1.plot(x_reward, np.poly1d(fit_reward)(x_reward), color='green', linestyle="dashed", label='sin MLP')

    # ax1.plot(x_mlp_reward, y_mlp_reward, color='blue', label='con MLP')
    fit_mlp_reward = np.polyfit(x_mlp_reward, y_mlp_reward, 2)
    ax1.plot(x_mlp_reward, np.poly1d(fit_mlp_reward)(x_mlp_reward), color='blue', linestyle="dashed" ,label='con MLP')
    ax1.set_title('Relación entre Episodio y Recompensa')
    ax1.set_xlabel('Episodio')
    ax1.set_ylabel('Recompensa')
    ax1.legend()

    # Gráfico 2    
    # ax2.plot(x_danger, y_danger, color='green', label='sin MLP')
    fit_danger = np.polyfit(x_danger, y_danger, 2)
    ax2.plot(x_danger, np.poly1d(fit_danger)(x_danger), color='green', linestyle="dashed", label='sin MLP')

    # ax2.plot(x_mlp_danger, y_mlp_danger, color='blue', label='con MLP')
    fit_mlp_danger = np.polyfit(x_mlp_danger, y_mlp_danger, 2)
    ax2.plot(x_mlp_danger, np.poly1d(fit_mlp_danger)(x_mlp_danger), color='blue', linestyle="dashed", label='con MLP')
    
    ax2.set_title('Relación entre Episodio y Estado peligroso')
    ax2.set_xlabel('Episodio')
    ax2.set_ylabel('Estado Peligroso')
    ax2.legend()

    plt.show()