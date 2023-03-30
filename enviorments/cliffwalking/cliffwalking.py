import pygame

class CliffwalkingEnviorment():
    px_width = 768
    px_height = 256
    px_cell_size = 64
    background_color = (255, 255, 255)
    player_color = (0, 0, 255)

    def __init__(self, player_position=[0, 0], red_flags=[], yellow_flags=[], green_flags=[]):
        
        # Inicializar juego y la pantalla
        pygame.init()
        self.screen = pygame.display.set_mode((self.px_width, self.px_height))
        pygame.display.set_caption("Cliffwalking")
        self.clock = pygame.time.Clock()
        
        # Posición inicial del personaje
        self.player_position = player_position

        # Posición de las banderas
        self.red_flags = red_flags
        self.yellow_flags = yellow_flags
        self.green_flags = green_flags        

    def place_red_flags(self, red_flags):
        for red_flag in red_flags:
            pygame.draw.rect(self.screen, (255, 0, 0), (red_flag[0] * self.px_cell_size, red_flag[1] * self.px_cell_size, self.px_cell_size, self.px_cell_size), 3)

    def place_yellow_flags(self, yellow_flags):
        for yellow_flag in yellow_flags:
            pygame.draw.rect(self.screen, (204, 204, 0), (yellow_flag[0] * self.px_cell_size, yellow_flag[1] * self.px_cell_size, self.px_cell_size, self.px_cell_size), 3)

    def place_green_flags(self, green_flags):
        for green_flag in green_flags:
            pygame.draw.rect(self.screen, (0, 255, 0), (green_flag[0] * self.px_cell_size, green_flag[1] * self.px_cell_size, self.px_cell_size, self.px_cell_size), 3)
    
    def run(self):
        while True:
            for evento in pygame.event.get():
                if evento.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                elif evento.type == pygame.KEYDOWN:
                    
                    # Mover el personaje
                    if evento.key == pygame.K_LEFT and self.player_position[0] > 0:
                        self.player_position[0] -= 1
                    elif evento.key == pygame.K_RIGHT and self.player_position[0] < self.px_width // self.px_cell_size - 1:
                        self.player_position[0] += 1
                    elif evento.key == pygame.K_UP and self.player_position[1] > 0:
                        self.player_position[1] -= 1
                    elif evento.key == pygame.K_DOWN and self.player_position[1] < self.px_height // self.px_cell_size - 1:
                        self.player_position[1] += 1

            # Dibujar la grilla
            self.screen.fill(self.background_color)
            for fila in range(self.px_height // self.px_cell_size):
                for columna in range(self.px_width // self.px_cell_size):
                    pygame.draw.rect(self.screen, (0, 0, 0), (columna * self.px_cell_size, fila * self.px_cell_size, self.px_cell_size, self.px_cell_size), 1)

            # Dibujar el personaje
            pygame.draw.rect(self.screen, self.player_color, (self.player_position[0] * self.px_cell_size, self.player_position[1] * self.px_cell_size, self.px_cell_size, self.px_cell_size))

            # Dibujar banderas
            self.place_red_flags(self.red_flags)
            self.place_yellow_flags(self.yellow_flags)
            self.place_green_flags(self.green_flags)

            # Actualizar la pantalla
            pygame.display.update()


if __name__ == "__main__":
    red_flags = [
        [1, 3], [2, 3], [3, 3], [4, 3], [5, 3],
        [6, 3], [7, 3], [8, 3], [9, 3], [10, 3],
    ]

    yellow_flags = []

    green_flags = [[11, 3]]

    env = CliffwalkingEnviorment(
        player_position=[0, 3],
        red_flags=red_flags,
        yellow_flags=yellow_flags,
        green_flags=green_flags,
    )
    
    env.run()