import Global
from gymplayground.testbed_gym import TestbedGym
import gymplayground.simulation_parameters_gym as sim_param_gym
import time

def evaluate(t_bed):
    """
        Evaluation method based on the environment of the testbed
    """
    print('evaluate environment {}'.format(t_bed.env_name))
    if t_bed.env_name == 'LunarLander-v2':
        success = t_bed.agents[0].env.env.sucessful_landing_count # number of successful landing (between the 2 flags)
        t_bed.agents[0].env.env.sucessful_landing_count = 0 # reset successful landing counter
        print('successful landing: {}'.format(success))
        msg = 'successful landing: {}'.format(success)
        return msg, success

    elif t_bed.env_name == 'MountainCar-v0': # average timesteps over the last 100 episodes
        score = t_bed.agents[0].scores
        avg_tmstp = sum(score) / len(score)
        print('average minus timestep: {}'.format(avg_tmstp))
        msg = 'average minus timestep: {}'.format(avg_tmstp)
        return msg, score

    elif t_bed.env_name == 'CartPole-v0': # average timesteps over the last 100 episodes
        score = t_bed.agents[0].scores
        avg_tmstp = sum(score) / len(score)
        print('average timestep: {}'.format(avg_tmstp))
        msg = 'average timestep: {}'.format(avg_tmstp)
        return msg, score
    else:
        return None

if __name__ == '__main__':

    # Import simulation parameters
    param = sim_param_gym.args

    param.debug = True
    param.render = False
    param.training = False
    param.max_ep = 100
    param.load_all_weights = True
    param.collect_experiences = False
    param.record = True
    param.suffix = 'confirm_master'
    param.solved_score = 100000 # just a high unreachable number so that the agent will play specified nums of episodes

    nn_file = param.file_to_load # Input nn file
    # ----------------------------------------------------------------------

    # Create Testbed
    testbed = TestbedGym(sim_param=param)

    # Simulation lifecycle
    testbed.setup_simulations()
    testbed.run_simulation()
    testbed.end_simulation()

    # Evaluation
    message, evalu = evaluate(testbed)

    # Eval file
    timestr = time.strftime('%Y%m%d_%H%M%S')
    file = open(testbed.sim_dir + 'eval_master.txt', 'w')

    file.write("*** Performance of a master agent. Run 100 episodes (with different init) during testing. ***\n")
    file.write(message)
    file.close()

    # -------------------------------------------------------

    print("\n_______________________")
    print("All simulation finished\n"
          "Number of simulations: {}\n".format(testbed.sim_count) +
          "Total simulations time: {}\n".format(Global.get_time()) +
          "Total simulations timesteps: {}".format(Global.timesteps))

    testbed.save_summary()

