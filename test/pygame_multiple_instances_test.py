from __future__ import division
import pygame
from pygame.locals import *
import math


def slowFunction():
    for i in xrange(1000000):
        a = math.sqrt(9123456)


class Simulation(object):
    def __init__(self, simulation_id):
        self.simulation_id = simulation_id
        print("Simulation: {}".format(simulation_id))

        pygame.init()
        self.clock = pygame.time.Clock()

        self.running = True

    def run(self):
        for j in xrange(2):
            # while self.running:
            slowFunction()

            deltaTime = self.clock.tick() / 1000.0

            # Show FPS
            print('deltaTime: {}'.format(deltaTime))
            # self.running = False

    def end(self):
        pygame.quit()
        print('Done! simulation: {}'.format(self.simulation_id))


if __name__ == '__main__':
    for i in xrange(100):
        print('****************************************')
        simulation = Simulation(i)
        simulation.run()
        simulation.end()

    print("All simulation finished")
