from __future__ import division
import argparse
import time
import sys

# -------------------- Simulation Parameters ----------------------

parser = argparse.ArgumentParser(description='Testbed Parameters Sharing')
parser.add_argument('--render', help='render the simulation', default='True')
parser.add_argument('--debug', help='print simulation log', default='True')
parser.add_argument('--record', help='record simulation log in file', default='False')
parser.add_argument('--fixed_ur_timestep', help='fixed your timestep', default='False')


parser.add_argument('--num_agents', help='number of agents in the simulation', default='1')

parser.add_argument('--training', help='train agent', default='True')
parser.add_argument('--exploration', help='agent takes random action at the beginning (exploration)',
                    default='True')
parser.add_argument('--collect_experiences', help='append a new experience to memory at each timestep',
                    default='True')
parser.add_argument('--max_timesteps', help='maximum number of timesteps for 1 simulation', default='-1')
parser.add_argument('--max_training_it', help='maximum number of training iterations for 1 simulation',
                    default='-1')
parser.add_argument('--multi_simulation', help='multiple successive simulations', default='1')
parser.add_argument('--random_agent', help='agent is taking random action', default='False')

parser.add_argument('--load_model',
                    help='load model to agent', default='False')
parser.add_argument('--load_full_weights',
                    help='load full weights of neural networks from master to learning agent', default='False')
parser.add_argument('--load_h1h2_weights',
                    help='load hidden layer 1 and 2 weights of neural networks from master to learning agent',
                    default='False')
parser.add_argument('--load_h1_weights',
                    help='load hidden layer 1 weights of neural networks from master to learning agent',
                    default='False')

parser.add_argument('--load_h2_weights',
                    help='load h2 weights of neural networks from master to learning agent',
                    default='False')
parser.add_argument('--load_out_weights',
                    help='load output weights of neural networks from master to learning agent',
                    default='False')
parser.add_argument('--load_h2out_weights',
                    help='load h2 output weights of neural networks from master to learning agent',
                    default='False')
parser.add_argument('--load_h1out_weights',
                    help='load h1 output weights of neural networks from master to learning agent',
                    default='False')

parser.add_argument('--load_memory', help='load defined number of experiences to agent', default='-1')
parser.add_argument('--file_to_load', help='name of the file to load NN weights or memory', default='')

parser.add_argument('--save_network_freq', help='save neural networks model every defined timesteps', default='-1')
parser.add_argument('--save_network_freq_training_it',
                    help='save neural networks model every defined training iterations', default='-1')
parser.add_argument('--save_memory_freq', help='save memory every defined timesteps', default='-1')
parser.add_argument('--start_save_nn_from_it', help='start saving neural networks model from defined training iterations', default='0')


parser.add_argument('--wait_learning_score_and_save_model',
                    help='wait agent to reach specified learning score before to close application', default='-1')
parser.add_argument('--suffix', help='custom suffix to add', default='')

parser.add_argument('--record_ls', help='record learning score of agent', default='False')

parser.add_argument('--dir_name', help='directory name to load NN files (to run in parallel universe)', default="")

# Parameter object
args = parser.parse_args()

def simulation_suffix(sim_param):
    """
        Simulation suffix name given simulation parameters
    """

    # Suffix
    suffix = ""
    if sim_param.load_full_weights == 'True':
        suffix = "loadfull"
    elif sim_param.load_h1h2_weights == 'True':
        suffix = "loadh1h2"
    elif sim_param.load_h1_weights == 'True':
        suffix = "loadh1"
    elif sim_param.load_model == 'True':
        suffix = "loadmodel"
    elif sim_param.load_h2_weights == 'True':
        suffix = "loadh2"
    elif sim_param.load_out_weights == 'True':
        suffix = "loadout"
    elif sim_param.load_h2out_weights == 'True':
        suffix = "loadh2out"
    elif sim_param.load_h1out_weights == 'True':
        suffix = "loadh1out"

    # Load memory
    if sim_param.load_memory != '-1':
        suffix += "load" + sim_param.load_memory + "xp"

    # Normal case
    if suffix == "":
        suffix = "normal"

    # No exploration
    if sim_param.exploration == 'False':
        suffix += "_noexplore"

    # Random agent
    if sim_param.random_agent == 'True':
        suffix = "random"

    # Number of agents
    if sim_param.num_agents != '1':
        suffix += "_" + sim_param.num_agents + "agents"

    # General purpose suffix
    if sim_param.suffix != '' and sim_param.suffix != "":
        suffix += '_' + sim_param.suffix

    return suffix

def simulation_dir(sim_param):
    """
        Simulation directory name given simulation parameters
    """

    multi_simulation = int(sim_param.multi_simulation)

    # Number of trials
    max_timesteps = int(sim_param.max_timesteps)
    max_training_it = int(sim_param.max_training_it)
    timestr = time.strftime("%Y%m%d_%H%M%S")

    # Handling illegal case
    if sim_param.random_agent == 'True' and max_training_it != -1:
        print("cannot be random and have max training_it")
        sys.exit()
    if max_training_it != -1 and max_timesteps != -1:
        print("both max_training_it and max_timesteps != -1, need only one")
        sys.exit()

    if max_training_it != -1:
        max_t = str(max_training_it) + "it_"
    elif max_timesteps != -1:
        max_t = str(max_timesteps) + "tmstp_"
    else:
        max_t = ""

    suffix = simulation_suffix(sim_param)
    sim_dir = "./simulation_data/" + suffix + "_" + max_t + str(multi_simulation) + "sim_" + timestr + "/"

    return sim_dir