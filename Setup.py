

SCREEN_WIDTH, SCREEN_HEIGHT = 720, 720 # 1440, 1280
# 1280 / PPM , 720 / PPM -> 64 , 36

# Box2D deals with meters, but we want to display pixels,
# so define a conversion factor:
PPM = 20.0  # pixels per meter

TARGET_FPS = 60
PHYSICS_TIME_STEP = 1.0 / TARGET_FPS
VEL_ITERS, POS_ITERS = 10, 10  # 6, 2