import pygame
from copy import copy


class CliffwalkingEnviorment():
    px_cell_size = 64
    background_color = (255, 255, 255)
    player_color = (0, 0, 255)
    rewards = 0

    def __init__(self, px_width, px_height, player_position=[0, 0], red_flags=[], green_flags=[]):
        self.px_width = px_width
        self.px_height = px_height
        # Inicializar juego y la pantalla
        pygame.init()
        self.screen = pygame.display.set_mode((self.px_width, self.px_height))
        pygame.display.set_caption("Cliffwalking")
        self.clock = pygame.time.Clock()

        # Posición inicial del personaje
        self.initial_position = copy(player_position)
        self.player_position = copy(player_position)

        # Fuente
        self.font = pygame.font.Font(None, 20)
        # Posición de las banderas
        self.red_flags = red_flags
        self.green_flags = green_flags

    def place_red_flags(self, red_flags):
        for red_flag in red_flags:
            pygame.draw.rect(self.screen, (255, 0, 0), (
                red_flag[0] * self.px_cell_size, red_flag[1] * self.px_cell_size, self.px_cell_size, self.px_cell_size), 3)
            text_surface = self.font.render("-100", True, (255, 0, 0))
            self.screen.blit(
                text_surface, (red_flag[0] * self.px_cell_size + 18, red_flag[1] * self.px_cell_size + 24))
        return True

    def place_green_flags(self, green_flags):
        for green_flag in green_flags:
            pygame.draw.rect(self.screen, (0, 255, 0), (
                green_flag[0] * self.px_cell_size, green_flag[1] * self.px_cell_size, self.px_cell_size, self.px_cell_size), 3)
        return True

    def update_reward(self):
        """
        Actualiza la recompensa del agente en función del estado al cuál transicionó, en caso de caer
        en una posición peligrosa, el agente es enviado al inicio
        """
        if self.player_position in self.red_flags:
            self.rewards += -100
            self.player_position = copy(self.initial_position)
        else:
            self.rewards += -1
        # text = 'Recompensa Acumulada: {0}'.format(self.rewards)
        # print(text)
        return True

    def run(self, actions):
        from time import sleep
        for action in actions:
            sleep(.05)
            # Izquierda
            if action == 0:
                self.player_position[0] -= 1
            # Arriba
            elif action == 1:
                self.player_position[1] -= 1
            # Derecha
            elif action == 2:
                self.player_position[0] += 1
            # Abajo
            elif action == 3:
                self.player_position[1] += 1

            self.update_reward()
            # Dibujar la grilla
            self.screen.fill(self.background_color)
            for fila in range(self.px_height // self.px_cell_size):
                for columna in range(self.px_width // self.px_cell_size):
                    pygame.draw.rect(self.screen, (0, 0, 0), (columna * self.px_cell_size,
                                     fila * self.px_cell_size, self.px_cell_size, self.px_cell_size), 1)

            # Dibujar el personaje
            pygame.draw.rect(self.screen, self.player_color, (
                self.player_position[0] * self.px_cell_size, self.player_position[1] * self.px_cell_size, self.px_cell_size, self.px_cell_size))

            # Dibujar banderas
            self.place_red_flags(self.red_flags)
            self.place_green_flags(self.green_flags)

            # Actualizar la pantalla
            pygame.display.update()
        pygame.quit()
        return True
