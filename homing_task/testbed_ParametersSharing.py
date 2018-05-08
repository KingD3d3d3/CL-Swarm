import random
import argparse
import pygame
# Box2D.b2 maps Box2D.b2Vec2 to vec2 (and so on)
from Box2D.b2 import (world)
from pygame.locals import *
import csv

try:
    # Running in PyCharm
    from AgentHoming import AgentHoming
    import res.colors as Color
    from Border import Border
    from Circle import StaticCircle
    from Box import StaticBox
    from MyContactListener import MyContactListener
    from Setup import *
    from Util import *
    from experimental.EntityManager import EntityManager
    from experimental.Entity import Entity
    import homing_global
except:
    # Running in command line
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info('Running from command line -> Import libraries as package')
    from .AgentHoming import AgentHoming
    from ..Border import Border
    from ..Circle import StaticCircle
    from ..res import colors as Color
    from .MyContactListener import MyContactListener
    from ..Setup import *
    from ..Util import *
    import homing_global


class TestbedParametersSharing(object):
    def __init__(self, screen_width, screen_height, target_fps, ppm, physics_timestep, vel_iters, pos_iters):

        self.screen_width = screen_width
        self.screen_height = screen_height
        self.target_fps = target_fps
        self.ppm = ppm
        self.physics_timestep = physics_timestep
        self.vel_iters = vel_iters
        self.pos_iters = pos_iters

        parser = argparse.ArgumentParser(description='Testbed Parameters Sharing')
        parser.add_argument('--render', help='render the simulation', default='True')
        parser.add_argument('--print_fps', help='print fps', default='False')
        parser.add_argument('--debug', help='print simulation log', default='True')
        parser.add_argument('--record', help='record simulation log in file', default='False')
        parser.add_argument('--fixed_ur_timestep', help='fixed your timestep', default='True')
        args = parser.parse_args()
        self.render = args.render == 'True'
        self.print_fps = args.print_fps == 'True'
        homing_global.debug = args.debug == 'True'
        homing_global.record = args.record == 'True'
        self.fixed_ur_timestep = args.fixed_ur_timestep == 'True'
        self.deltaTime = 1.0 / target_fps
        self.fps = 1.0 / self.deltaTime
        self.accumulator = 0

        # -------------------- Pygame Setup ----------------------

        pygame.init()
        self.screen = None
        if self.render:
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), 0, 32)
            pygame.display.set_caption('Testbed Parameters Sharing')
        self.clock = pygame.time.Clock()
        self.myfont = pygame.font.SysFont("monospace", 15)

        self.running = True
        self.pause = False
        self.recording = False

        # -------------------- PyBox2d World Setup ----------------------

        # Create the world
        self.world = world(gravity=(0, 0), doSleep=True, contactListener=MyContactListener())  # gravity = (0, -10)

        # Border of the map
        self.border = Border(screen=self.screen, world=self.world)

        # Agents
        self.numAgents = 1  # total numbers of agents in the simulation
        self.agents = []
        for i in xrange(self.numAgents):
            randX = random.randint(2, self.screen_width / self.ppm - 2)
            randY = random.randint(2, self.screen_height / self.ppm - 2)
            randAngle = degToRad(random.randint(0, 360))
            a = AgentHoming(screen=self.screen, world=self.world, x=randX, y=randY, angle=randAngle,
                            radius=1.5, id=i, numAgents=self.numAgents)
            self.agents.append(a)

        # Obstacles
        self.obstacles = []
        circle1 = StaticCircle(screen=self.screen, world=self.world, x=12.75, y=24, radius=2)
        circle1.id = 1
        #circle2 = StaticCircle(screen=self.screen, world=self.world, x=26, y=30, radius=2)
        circle2 = StaticCircle(screen=self.screen, world=self.world, x=26, y=28, radius=2)
        circle2.id = 2
        #circle3 = StaticCircle(screen=self.screen, world=self.world, x=19.75, y=12.25, radius=2)
        circle3 = StaticCircle(screen=self.screen, world=self.world, x=14.75, y=10, radius=2)
        circle3.id = 3
        circle4 = StaticCircle(screen=self.screen, world=self.world, x=32, y=18, radius=2)
        circle4.id = 4
        #circle5 = StaticCircle(screen=self.screen, world=self.world, x=44.25, y=23.75, radius=2)
        circle5 = StaticCircle(screen=self.screen, world=self.world, x=49.25, y=26, radius=2)
        circle5.id = 5
        #circle6 = StaticCircle(screen=self.screen, world=self.world, x=38, y=6, radius=2)
        circle6 = StaticCircle(screen=self.screen, world=self.world, x=38, y=8, radius=2)
        circle6.id = 6
        circle7 = StaticCircle(screen=self.screen, world=self.world, x=51.25, y=12, radius=2)
        circle7.id = 7
        self.obstacles.extend((circle1, circle2, circle3, circle4, circle5, circle6, circle7))
        self.numObstacles = len(self.obstacles)

    def draw(self):
        """
            Rendering part
        """
        if not self.render:
            return

        # Reset screen's pixels
        self.screen.fill((0, 0, 0, 0))

        # Draw goals
        goalFont = pygame.font.SysFont("monospace", 25)
        goal1Pos = 100

        # Goal 1
        pygame.draw.circle(self.screen, Color.Red, (goal1Pos, goal1Pos), 20)
        self.screen.blit(goalFont.render('1', True, Color.White), (goal1Pos - 8, goal1Pos - 12))

        # Goal 2
        pygame.draw.circle(self.screen, Color.Red, (self.screen_width - goal1Pos, self.screen_height - goal1Pos), 20)
        self.screen.blit(goalFont.render('2', True, Color.White),
                         (self.screen_width - goal1Pos - 8, self.screen_height - goal1Pos - 12))

        # Moving Objects
        for i in xrange(self.numAgents):
            self.agents[i].draw()

        # Obstacles
        for i in xrange(self.numObstacles):
            self.obstacles[i].draw()

        # Boundary
        self.border.draw()

        # Show FPS
        PrintFPS(self.screen, self.myfont, 'FPS : ' + str('{:3.2f}').format(self.fps))

        #pygame.draw.rect(self.screen, Color.Cyan, (1260, 0, 20, 720))
        #pygame.draw.circle(self.screen, Color.Cyan, (0, 720), 20)

        # Flip the screen
        pygame.display.flip()


    def handle_events(self):
        """
            Check and handle the event queue
        """
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                # The user closed the window or pressed escape
                self.running = False
            if event.type == KEYDOWN and event.key == K_p:
                self.pause = not self.pause  # Pause the game
            if event.type == KEYDOWN and event.key == K_r:
                if not homing_global.record:  # Record simulation
                    print("start recording")
                    homing_global.record = True
                    filename = homing_global.fileCreate()
                    homing_global.fo = open(filename, 'a')
                    homing_global.writer = csv.writer(homing_global.fo)
                else:  # Stop recording
                    print("stop recording")
                    homing_global.record = False
                    homing_global.fo.close()
            if event.type == KEYDOWN and event.key == K_b:
                print("Save Agent's brain and Memory")
                self.agents[0].save_brain()
                self.agents[0].save_memory()
            if event.type == KEYDOWN and event.key == K_l:
                print("Load brain model")
                self.agents[0].load_weights()

    def update(self):
        """
            Update game logic
        """
        # Update the agents
        for i in xrange(self.numAgents):
            self.agents[i].update()

    def fps_physic_step(self):
        """
            FPS and Physic's Step Part
        """
        if self.render:
            self.deltaTime = self.clock.tick(self.target_fps) / 1000.0
            self.fps = self.clock.get_fps()

            if self.fixed_ur_timestep:
                # "Fixed your timestep" technique
                # Physics is stepped by a fixed amount i.e. 1/60s.
                # Faster machine i.e. render at 120fps -> step the physics one every 2 frames
                # Slower machine i.e. render at 30fps -> run the physics twice
                self.accumulator += self.deltaTime
                while self.accumulator >= self.physics_timestep:
                    # Physic step
                    self.world.Step(self.physics_timestep, self.vel_iters, self.pos_iters)
                    self.world.ClearForces()

                    self.accumulator -= self.physics_timestep
            else:
                # Basic Physic step
                self.world.Step(self.physics_timestep, self.vel_iters, self.pos_iters)
                self.world.ClearForces()

        elif not self.render:
            self.deltaTime = self.clock.tick() / 1000.0
            self.fps = self.clock.get_fps()

            if self.deltaTime >= self.target_fps:  # Frame is faster than target (60fps) -> simulation run faster
                self.accumulator = 0

                # Physic step
                self.world.Step(self.physics_timestep, self.vel_iters, self.pos_iters)
                self.world.ClearForces()
            else:
                self.accumulator += self.deltaTime

                while self.accumulator >= self.physics_timestep:
                    # Physic step
                    self.world.Step(self.physics_timestep, self.vel_iters, self.pos_iters)
                    self.world.ClearForces()

                    self.accumulator -= self.physics_timestep

    def run(self):
        """
            Main Game Loop
        """
        while self.running:

            self.handle_events()

            # Pause the game
            if self.pause:
                # clock.tick compute how many milliseconds have passed since the previous call.
                # If we don't call that during pause, clock.tick will compute time spend during pause
                # thus timer is counting during simulation pause -> we want to avoid that !
                if self.render:
                    self.deltaTime = self.clock.tick(self.target_fps) / 1000.0
                elif not self.render:
                    self.deltaTime = self.clock.tick() / 1000.0
                continue  # go back to loop entry

            self.update()
            self.fps_physic_step()
            self.draw()

            if self.print_fps:
                print('FPS : ' + str('{:3.2f}').format(self.fps))

            # Step counter
            homing_global.timestep += 1

    @staticmethod
    def end():
        """
            Last function called before leaving the application
        """
        pygame.quit()
        print('Pygame Exit')

        if homing_global.record:
            homing_global.record = False
            homing_global.fo.close()
            print("stop recording")


if __name__ == '__main__':
    simulation = TestbedParametersSharing(SCREEN_WIDTH, SCREEN_HEIGHT, TARGET_FPS, PPM,
                                          PHYSICS_TIME_STEP, VEL_ITERS, POS_ITERS)
    simulation.run()
    simulation.end()
