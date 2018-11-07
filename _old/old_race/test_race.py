
import pygame
from pygame.locals import *
from task_race.EnvironmentRaceS import EnvironmentRaceS


FPS = 60
pause = False

if __name__ == '__main__':

    environment = EnvironmentRaceS(render=True, fixed_ur_timestep=False)

    # -------------------- Main Game Loop ----------------------

    timeCount = 0
    stepCount = 0
    running = True
    pause = False
    while running:
        # Check the event queue
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                # The user closed the window or pressed escape
                running = False
            if event.type == KEYDOWN and event.key == K_p:
                pause = not pause  # Pause the game

        # Pause the game
        if pause:
            continue

        environment.update()

    pygame.quit()
    print('Done!')
