

import pygame
# Box2D.b2 maps Box2D.b2Vec2 to vec2 (and so on)
from Box2D.b2 import (world, vec2)
from pygame.locals import *
try:
    # Running in PyCharm
    import res.colors as Color
    from objects.Border import Border
    from objects.Circle import StaticCircle
    from objects.Box import StaticBox
    from Setup import *
    from Util import world_to_pixels, pixels_to_world
    import Util
    from task_race.AgentRace import AgentRace
    from task_race.RaceCircle import RaceCircle
    from task_race.RaceCircle_v2 import RaceCircleV2
    from task_race.RaceContactListener import RaceContactListener
except NameError as err:
    print(err, "--> our error message")
    # Running in command line
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Running from command line -> Import libraries as package")
    from ..objects.Border import Border
    from ..objects.Circle import StaticCircle
    from ..res import colors as Color
    from ..Setup import *
    from ..Util import world_to_pixels, pixels_to_world
    from .. import Util
    from ..res.print_colors import *
    from ..res import print_colors as PrintColor
    from task_race.AgentRace import AgentRace
    from task_race.RaceCircle import RaceCircle
    from task_race.RaceContactListener import RaceContactListener

class EnvironmentRace(object):

    def __init__(self, render=False, fixed_ur_timestep=False, num_agents=1, solved_score=100000, seed=None):

        self.render = render
        self.fixed_ur_timestep = fixed_ur_timestep

        # -------------------- Pygame Setup ----------------------

        pygame.init()

        self.delta_time = 1.0 / TARGET_FPS  # 0.016666
        self.fps = TARGET_FPS
        self.accumulator = 0

        self.screen = None
        if self.render:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
            pygame.display.set_caption('Testbed Homing Simple')
        self.clock = pygame.time.Clock()
        self.myfont = pygame.font.SysFont("monospace", 15)

        # -------------------- Environment and PyBox2d World Setup ----------------------

        # Create the world
        self.world = world(gravity=(0, 0), doSleep=True, contactListener=RaceContactListener())

        env = 'RaceCircle_v2' # 'RaceCircle_v2' # 'RaceCircle'
        if env == 'RaceCircle':
            self.race = RaceCircle(screen=self.screen, world=self.world)
        elif env == 'RaceCircle_v2':
            self.race = RaceCircleV2(screen=self.screen, world=self.world)

        self.agent = AgentRace(screen=self.screen, world=self.world,
                               x=30, y=30, radius=1.25, solved_score=solved_score, env=env, seed=seed)

    def draw(self):
        """
            Rendering part
        """
        if not self.render:
            return

        # Reset screen's pixels
        self.screen.fill((0, 0, 0, 0))

        # Boundary
        self.race.draw()

        self.agent.draw()

        # Show FPS
        Util.print_fps(self.screen, self.myfont, 'FPS : ' + str('{:3.2f}').format(self.fps))

        # Flip the screen
        pygame.display.flip()

    def update(self):
        """
            Environment main
        """
        self.draw()
        self.before_step()
        self.fps_physic_step()
        self.after_step()

    def before_step(self):
        """
            Call agent's before_step
        """
        self.agent.before_step()

    def after_step(self):
        """
            Call agent's after_step
        """
        self.agent.after_step()

    def fps_physic_step(self):
        """
            FPS and Physic's Step Part
        """
        # With rendering
        if self.render:
            self.delta_time = self.clock.tick(TARGET_FPS) / 1000.0
            self.fps = self.clock.get_fps()

            # Fixed your timestep
            if self.fixed_ur_timestep:
                """
                    'Fixed your timestep' technique.
                    Physics is stepped by a fixed amount e.g. 1/60s.
                    Faster machine e.g. render at 120fps -> step the physics one every 2 frames
                    Slower machine e.g. render at 30fps -> run the physics twice
                """
                self.accumulator += self.delta_time

                while self.accumulator >= PHYSICS_TIME_STEP:
                    # Physic step
                    self.world.Step(PHYSICS_TIME_STEP, VEL_ITERS, POS_ITERS)
                    self.world.ClearForces()

                    self.accumulator -= PHYSICS_TIME_STEP
                return

            # Frame dependent
            else:
                # Physic step
                self.world.Step(PHYSICS_TIME_STEP, VEL_ITERS, POS_ITERS)
                self.world.ClearForces()
                return

        # No rendering
        elif not self.render:
            self.delta_time = self.clock.tick() / 1000.0
            # self.fps = self.clock.get_fps()

            # Frame dependent -- Physic step
            self.world.Step(PHYSICS_TIME_STEP, VEL_ITERS, POS_ITERS)
            self.world.ClearForces()
            return