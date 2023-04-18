import tkinter as tk
from threading import Thread

def cliff_walking_normal():
    """
    Crea el entorno de CliffWalking con el agente en la posición (0, 3), y una meta en la posición (11, 3),
    y estados peligrosos desde la posición (1, 3) hasta la (10, 3)
    """
    from environment.cliffwalking import cliffwalking
    from models.cliffwalking import model

    # Escenario
    red_flags = [
        [1, 3], [2, 3], [3, 3], [4, 3], [5, 3],
        [6, 3], [7, 3], [8, 3], [9, 3], [10, 3],
    ]

    green_flags = [[11, 3]]    

    # Sarsa
    n_states = 48
    n_actions = 4
    initial_state = 37
    goal_state = 48
    col_environment = 12
    row_environment = 4
    range_danger = [38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
    model_sarsa = model.SarsaModel(col_environment, row_environment, range_danger,  n_states, n_actions, initial_state, goal_state)
    for episode in range(150):
        actions = []
        env = cliffwalking.CliffwalkingEnviorment(
            px_width=768,
            px_height=256,
            player_position=[0, 3],
            red_flags=red_flags,
            green_flags=green_flags,            
        )
        scores, actions = model_sarsa.run(episode)
        env.run(actions)
        text = 'Episodio: {0} - Recompensa acumulada: {1}'.format(episode, sum(scores))
        print(text)

def cliff_walking_controlled():
    """
    Crea el entorno de controlado de CliffWalking
    """
    from environment.cliffwalking import cliffwalking
    from models.cliffwalking import model

    # Escenario
    red_flags = [
        [1, 3], [2, 3], [3, 3], [4, 3]
    ]

    green_flags = [[5, 3]]

    # Sarsa
    n_states = 24
    n_actions = 4
    initial_state = 19
    goal_state = 24
    col_environment = 6
    row_environment = 4
    range_danger = [20, 21, 22, 23]
    model_sarsa = model.SarsaModel(col_environment, row_environment, range_danger, n_states, n_actions, initial_state, goal_state)
    for episode in range(150):
        actions = []
        env = cliffwalking.CliffwalkingEnviorment(
            px_width=384,
            px_height=256,
            player_position=[0, 3],
            red_flags=red_flags,
            green_flags=green_flags,
        )
        scores, actions = model_sarsa.run(episode)
        env.run(actions)
        text = 'Episodio: {0} - Recompensa acumulada: {1}'.format(episode, sum(scores))
        print(text)

def cliff_option():
    if cliff_checkbox.get() == False:
        cliff_walking_normal()
    else:
        cliff_walking_controlled()
    

def option2():
    print('#'*20)
    return True

if __name__ == "__main__":    
    # crear la ventana principal
    root = tk.Tk()
    root.title("Seleccione el escenario")
    root.geometry("500x300")

    cliff_checkbox = tk.BooleanVar()

    # crear un marco para las opciones
    options_frame = tk.Frame(root)
    options_frame.pack()

    # crear las etiquetas y botones para cada opción
    cliff_option_label = tk.Label(options_frame, text="Cliffwalking")    
    cliff_option_button = tk.Button(options_frame, text="Seleccionar", command=cliff_option)
    cliff_controlled_checkbox = tk.Checkbutton(options_frame, text="Controlado", variable=cliff_checkbox)

    option2_label = tk.Label(options_frame, text="Opción 2")
    option2_button = tk.Button(options_frame, text="Seleccionar", command=option2)
    option2_controlled_checkbox = tk.Checkbutton(options_frame, text="Controlado")

    # colocar las etiquetas y botones en el marco
    cliff_option_label.grid(row=0, column=0, padx=10, pady=50)
    cliff_option_button.grid(row=0, column=1, padx=10, pady=50)
    cliff_controlled_checkbox.grid(row=0, column=2, padx=10, pady=50)
    option2_label.grid(row=2, column=0, padx=10, pady=50)
    option2_button.grid(row=2, column=1, padx=10, pady=50)
    option2_controlled_checkbox.grid(row=2, column=2, padx=10, pady=50)

    # iniciar el bucle principal de la ventana
    root.mainloop()