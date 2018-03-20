import pygame
from pygame.locals import *
import math

def slowFunction():
    for i in xrange(50000):
        a = math.sqrt(9123456)

if __name__ == '__main__':

    # -------------------- Pygame Setup ----------------------

    pygame.init()
    clock = pygame.time.Clock()

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
            continue  # go back to loop entry

        slowFunction()

        deltaTime = clock.tick() / 1000.0
        fps = clock.get_fps()

        # Show FPS
        print('FPS : ' + str('{:3.2f}').format(fps))

    pygame.quit()
    print('Done!')
