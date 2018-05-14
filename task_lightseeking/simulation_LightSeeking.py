import random
import argparse
import pygame
# Box2D.b2 maps Box2D.b2Vec2 to vec2 (and so on)
from Box2D.b2 import (world)
from pygame.locals import *
import csv
import matplotlib.pyplot as plt

try:
    # Running in PyCharm
    from AgentLightSeeking import AgentLightSeeking
    import res.colors as Color
    from Border import Border
    from Circle import StaticCircle
    from Box import StaticBox
    from MyContactListener import MyContactListener
    from Setup import *
    from Util import *
    import lightseeking_debug
    from res.print_colors import printColor
    import lightseeking_global
except:
    # Running in command line
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info('Running from command line -> Import libraries as package')
    from .AgentLightSeeking import AgentLightSeeking
    from ..Border import Border
    from ..Circle import StaticCircle
    from ..res import colors as Color
    from .MyContactListener import MyContactListener
    from ..Setup import *
    from ..Util import *
    import lightseeking_debug
    from ..res.print_colors import printColor
    import lightseeking_global


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
        parser.add_argument('--training', help='if yes should train agent', default='True')
        args = parser.parse_args()
        self.render = args.render == 'True'
        self.print_fps = args.print_fps == 'True'
        lightseeking_global.debug = args.debug == 'True'
        lightseeking_global.record = args.record == 'True'
        self.fixed_ur_timestep = args.fixed_ur_timestep == 'True'
        self.training = args.training == 'True'
        self.deltaTime = 1.0 / target_fps # 0.016666
        self.fps = 1.0 / self.deltaTime
        self.accumulator = 0

        self.learning_scores = []  # initializing the mean score curve (sliding window of the rewards) with respect to timestep

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
            a = AgentLightSeeking(screen=self.screen, world=self.world, x=randX, y=randY, angle=randAngle,
                                  radius=1.5, id=i, numAgents=self.numAgents, training=self.training)
            self.agents.append(a)

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

        # Boundary
        self.border.draw()

        # Show FPS
        PrintFPS(self.screen, self.myfont, 'FPS : ' + str('{:3.2f}').format(self.fps))

        # pygame.draw.rect(self.screen, Color.Cyan, (1260, 0, 20, 720))
        # pygame.draw.circle(self.screen, Color.Cyan, (0, 720), 20)

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
                if not lightseeking_global.record:  # Record simulation
                    lightseeking_debug.xprint(msg="Start recording")
                    lightseeking_global.record = True
                    filename = lightseeking_global.fileCreate()
                    lightseeking_global.fo = open(filename, 'a')
                    lightseeking_global.writer = csv.writer(lightseeking_global.fo)
                else:  # Stop recording
                    lightseeking_debug.xprint(msg="Stop recording")
                    lightseeking_global.record = False
                    lightseeking_global.fo.close()
            if event.type == KEYDOWN and event.key == K_s:
                lightseeking_debug.xprint(msg="Save Agent's brain and Memory")
                self.agents[0].save_brain()
                self.agents[0].save_memory()
            if event.type == KEYDOWN and event.key == K_l:
                lightseeking_debug.xprint(msg="Load full model weights")
                self.agents[0].load_weights()
                self.agents[0].stop_training()
            if event.type == KEYDOWN and event.key == K_w:
                lightseeking_debug.xprint(msg="Load lower layer weights")
                self.agents[0].load_lower_layers_weights()
                self.agents[0].stop_training()
            if event.type == KEYDOWN and event.key == K_p:  # plot Agent's learning scores
                self.plot_learning_scores()

    def update(self):
        """
            Update game logic
        """
        # Update the agents
        for i in xrange(self.numAgents):
            self.agents[i].update()
        self.learning_scores.append(self.agents[0].learning_score())  # appending the learning score
        if self.agents[0].learning_score() > self.agents[0].maxReward:
            self.running = False
            printColor(msg="achieved maximum reward")

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

            if self.deltaTime <= self.target_fps:  # Frame is faster than target (60fps) -> simulation run faster
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
                printColor(msg='FPS : ' + str('{:3.2f}').format(self.fps))

            # Step counter
            lightseeking_global.timestep += 1

            # Time counter
            lightseeking_global.timer += self.deltaTime

    def end(self):
        """
            Last function called before leaving the application
        """
        pygame.quit()
        lightseeking_debug.xprint(msg='Pygame Exit')

        if lightseeking_global.record:
            lightseeking_global.record = False
            lightseeking_global.fo.close()
            lightseeking_debug.xprint(msg="Stop recording")

        self.plot_learning_scores()

    def plot_learning_scores(self):
        plt.plot(self.learning_scores)
        plt.xlabel('Timestep')
        plt.ylabel('Learning Score')
        plt.title('Agent\'s Learning Score over Timestep')
        plt.grid(True)
        plt.show(block=False)


if __name__ == '__main__':
    simulation = TestbedParametersSharing(SCREEN_WIDTH, SCREEN_HEIGHT, TARGET_FPS, PPM,
                                          PHYSICS_TIME_STEP, VEL_ITERS, POS_ITERS)
    simulation.run()
    simulation.end()
