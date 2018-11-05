
from Box2D import b2RayCastCallback, b2Vec2
from task_race.RaceCircle import Quadrilateral
class RayCastCallback(b2RayCastCallback):
    """
    This class captures the closest hit shape.
    """

    def __init__(self, **kwargs):
        b2RayCastCallback.__init__(self)
        self.fixture = None
        self.hit = False

    # Called for each fixture found in the query. You control how the ray proceeds
    # by returning a float that indicates the fractional length of the ray. By returning
    # 0, you set the ray length to zero. By returning the current fraction, you proceed
    # to find the closest point. By returning 1, you continue with the original ray
    # clipping.
    def ReportFixture(self, fixture, point, normal, fraction):
        self.fixture = fixture
        self.point = b2Vec2(point)
        self.normal = b2Vec2(normal)
        self.hit = True # flag to inform raycast hit an object
        self.fraction = fraction
        # You will get this error: "TypeError: Swig director type mismatch in output value of type 'float32'"
        # without returning a value

        return fraction
