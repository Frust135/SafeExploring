import pygame

class CliffwalkingEnviorment():
    px_width = 768
    px_height = 256
    px_cell_size = 64
    background_color = (255, 255, 255)
    player_color = (0, 0, 255)
    rewards = 0

    def __init__(self, player_position=[0, 0], red_flags=[], yellow_flags=[], green_flags=[]):
        
        # Inicializar juego y la pantalla
        pygame.init()
        self.screen = pygame.display.set_mode((self.px_width, self.px_height))
        pygame.display.set_caption("Cliffwalking")
        self.clock = pygame.time.Clock()
        
        # Posición inicial del personaje
        self.player_position = player_position

        # Fuente
        self.font = pygame.font.Font(None, 20)
        # Posición de las banderas
        self.red_flags = red_flags
        self.yellow_flags = yellow_flags
        self.green_flags = green_flags        

    def place_red_flags(self, red_flags):
        for red_flag in red_flags:
            pygame.draw.rect(self.screen, (255, 0, 0), (red_flag[0] * self.px_cell_size, red_flag[1] * self.px_cell_size, self.px_cell_size, self.px_cell_size), 3)
            text_surface = self.font.render("-100", True, (255, 0, 0))
            self.screen.blit(text_surface, (red_flag[0] * self.px_cell_size + 18, red_flag[1] * self.px_cell_size + 24))
        return True

    def place_yellow_flags(self, yellow_flags):
        for yellow_flag in yellow_flags:
            pygame.draw.rect(self.screen, (204, 204, 0), (yellow_flag[0] * self.px_cell_size, yellow_flag[1] * self.px_cell_size, self.px_cell_size, self.px_cell_size), 3)
            # text_surface = self.font.render("-100", True, (204, 204, 0))
            # self.screen.blit(text_surface, (yellow_flag[0] * self.px_cell_size + 18, yellow_flag[1] * self.px_cell_size + 24))
        return True

    def place_green_flags(self, green_flags):
        for green_flag in green_flags:
            pygame.draw.rect(self.screen, (0, 255, 0), (green_flag[0] * self.px_cell_size, green_flag[1] * self.px_cell_size, self.px_cell_size, self.px_cell_size), 3)
            # text_surface = self.font.render("-100", True, (0, 255, 0))
            # self.screen.blit(text_surface, (green_flag[0] * self.px_cell_size + 18, green_flag[1] * self.px_cell_size + 24))
        return True

    def update_reward(self):
        """
        Actualiza la recompensa del agente en función del estado al cuál transicionó
        """
        if self.player_position in self.red_flags: self.rewards+=-100
        elif self.player_position in self.yellow_flags: self.rewards+=-100
        else: self.rewards+=-1
        text = 'Recompensa Acumulada: {0}'.format(self.rewards)
        print(text)
        return True
    
    def run(self):
        finished = False
        while not finished:
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
                    if self.player_position in self.green_flags:
                        finished=True
                    else:
                        self.update_reward()
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
        text = "Recompensa Final: {0}".format(self.rewards)
        print('#### Finalizado ###')
        print(text)
        pygame.quit()
        return self.rewards