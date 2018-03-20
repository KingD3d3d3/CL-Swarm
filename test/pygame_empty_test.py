import pygame
from pygame.locals import *

SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
FPS = 60

if __name__ == '__main__':

    render = True
    deltaTime = FPS / 1000.0
    fps = 1.0 / deltaTime

    accumulator = 0
    interpolated = False

    # -------------------- Pygame Setup ----------------------

    pygame.init()
    screen = None
    if render:
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
        pygame.display.set_caption('Homing Task Testbed')
    clock = pygame.time.Clock()

    myfont = pygame.font.SysFont("monospace", 30)
    def PrintScreen(text, color=(255, 0, 0, 255), pos=(0,0)):
        """
        Draw some text at the top status lines
        and advance to the next line.
        """
        screen.blit(myfont.render(
            text, True, color), pos)

    fpsFont = pygame.font.SysFont("monospace", 15)
    def PrintFPS(text, color=(255, 0, 0, 255)):
        """
        Draw some text at the top status lines
        and advance to the next line.
        """
        screen.blit(fpsFont.render(
            text, True, color), (10, 3))
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
                pause = not pause # Pause the game

        screen.fill((0, 0, 0, 0))

        PrintScreen('HELLO WORLD', pos=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2))

        deltaTime = clock.tick() / 1000.0
        fps = clock.get_fps()

        # Show FPS
        PrintFPS('FPS : ' + str('{:3.2f}').format(fps))

        pygame.display.flip()

    pygame.quit()
    print('Done!')
