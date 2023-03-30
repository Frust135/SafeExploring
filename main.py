import tkinter as tk
from enviorments.cliffwalking import cliffwalking

def cliff_option():
    """
    Crea el entorno de CliffWalking con el agente en la posición (0, 3), y una meta en la posición (11, 3),
    y estados peligrosos desde la posición (1, 3) hasta la (10, 3)
    """
    red_flags = [
        [1, 3], [2, 3], [3, 3], [4, 3], [5, 3],
        [6, 3], [7, 3], [8, 3], [9, 3], [10, 3],
    ]

    yellow_flags = []

    green_flags = [[11, 3]]

    env = cliffwalking.CliffwalkingEnviorment(
        player_position=[0, 3],
        red_flags=red_flags,
        yellow_flags=yellow_flags,
        green_flags=green_flags,
    )    
    reward = env.run()
    print(reward)

def option2():
    return True

if __name__ == "__main__":
    # crear la ventana principal
    root = tk.Tk()
    root.title("Seleccione el escenario")
    root.geometry("500x300")

    # crear un marco para las opciones
    options_frame = tk.Frame(root)
    options_frame.pack()

    # crear las etiquetas y botones para cada opción
    cliff_option_label = tk.Label(options_frame, text="Cliffwalking")
    cliff_option_button = tk.Button(options_frame, text="Seleccionar", command=cliff_option)
    option2_label = tk.Label(options_frame, text="Opción 2")
    option2_button = tk.Button(options_frame, text="Seleccionar", command=option2)

    # colocar las etiquetas y botones en el marco
    cliff_option_label.grid(row=0, column=0, padx=10, pady=50)
    cliff_option_button.grid(row=0, column=1, padx=10, pady=50)
    option2_label.grid(row=2, column=0, padx=10, pady=50)
    option2_button.grid(row=2, column=1, padx=10, pady=50)

    # iniciar el bucle principal de la ventana
    root.mainloop()