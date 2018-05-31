from __future__ import division
import math
import time

start_time = time.time()
def get_time():
    elapsed_time = time.time() - start_time

    m, s = divmod(elapsed_time, 60)
    h, m = divmod(m, 60)
    formated = "%dh %02dm %02ds" % (h, m, s)
    return formated

timestep = 1  # 0 # timesteps passed since beginning of simulation