
import time

# Time and timesteps passed since the beginning of the program
timesteps = 0
start_time = time.time()
def get_time():
    """
        Return the time passed since the beginning of the program
    """
    elapsed_time = time.time() - start_time

    m, s = divmod(elapsed_time, 60)
    h, m = divmod(m, 60)
    formated = "%dh %02dm %02ds" % (h, m, s)
    return formated

# Time and timesteps passed since beginning of a simulation
sim_timesteps = 0
sim_start_time = time.time()
def get_sim_time():
    """
        Return the time passed since beginning of a simulation
    """
    elapsed_time = time.time() - sim_start_time

    m, s = divmod(elapsed_time, 60)
    h, m = divmod(m, 60)
    formated = "%dh %02dm %02ds" % (h, m, s)
    return formated
def get_sim_time_in_seconds():
    """
        Return the time in seconds passed since the beginning of a simulation
    """
    elapsed_time = time.time() - sim_start_time
    return elapsed_time