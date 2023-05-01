import tkinter as tk


def cliff_option():
    from environment.cliffwalking.create_environment import cliff_walking_normal, cliff_walking_controlled
    if cliff_checkbox.get() == False:
        cliff_walking_normal()
    else:
        cliff_walking_controlled()
    return True


def cart_pole_option():
    from environment.cart_pole.create_environment import cart_pole_normal, cart_pole_controlled
    if cart_pole_checkbox.get() == False:
        cart_pole_normal()
    else:
        cart_pole_controlled()    
    return True


if __name__ == "__main__":
    # crear la ventana principal
    root = tk.Tk()
    root.title("Seleccione el escenario")
    root.geometry("500x300")

    cliff_checkbox = tk.BooleanVar()
    cart_pole_checkbox = tk.BooleanVar()

    # crear un marco para las opciones
    options_frame = tk.Frame(root)
    options_frame.pack()

    # crear las etiquetas y botones para cada opción
    cliff_option_label = tk.Label(options_frame, text="Cliffwalking")
    cliff_option_button = tk.Button(
        options_frame, text="Seleccionar", command=cliff_option)
    cliff_controlled_checkbox = tk.Checkbutton(
        options_frame, text="Controlado", variable=cliff_checkbox)

    cart_pole_label = tk.Label(options_frame, text="Cart Pole")
    cart_pole_button = tk.Button(
        options_frame, text="Seleccionar", command=cart_pole_option)
    cart_pole_controlled_checkbox = tk.Checkbutton(
        options_frame, text="Controlado", variable=cart_pole_checkbox)

    # colocar las etiquetas y botones en el marco
    cliff_option_label.grid(row=0, column=0, padx=10, pady=50)
    cliff_option_button.grid(row=0, column=1, padx=10, pady=50)
    cliff_controlled_checkbox.grid(row=0, column=2, padx=10, pady=50)
    cart_pole_label.grid(row=2, column=0, padx=10, pady=50)
    cart_pole_button.grid(row=2, column=1, padx=10, pady=50)
    cart_pole_controlled_checkbox.grid(row=2, column=2, padx=10, pady=50)

    # iniciar el bucle principal de la ventana
    root.mainloop()
