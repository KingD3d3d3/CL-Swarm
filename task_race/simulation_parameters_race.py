
import argparse
import time
import res.Util as Util

# -------------------- Simulation Parameters ----------------------

parser = argparse.ArgumentParser(description='Testbed Race')
parser.add_argument('--display', help='display the simulation', default=True, type=Util.str2bool)
parser.add_argument('--debug', help='print simulation log', default=True, type=Util.str2bool)
parser.add_argument('--manual', help='control manually the car', default=False, type=Util.str2bool)
parser.add_argument('--record', help='record simulation log in file', default=False, type=Util.str2bool)
parser.add_argument('--save_model', help='save agent model in file', default=False, type=Util.str2bool)
parser.add_argument('--save_mem', help='save agent experiences in file', default=False, type=Util.str2bool)
parser.add_argument('--num_agents', help='number of agents in the simulation', default=1, type=int)

parser.add_argument('--training', help='train agent', default=True, type=Util.str2bool)
parser.add_argument('--exploration', help='agent takes random action at the beginning (exploration)', default=True, type=Util.str2bool)
parser.add_argument('--collect_experiences', help='append a new experience to memory', default=True, type=Util.str2bool)
parser.add_argument('--max_ep', help='maximum number of episodes for 1 simulation', default=0, type=int)

parser.add_argument('--solved_timesteps', help='average score agent needs to reach to consider the problem solved', default=0, type=int)
parser.add_argument('--multi_sim', help='multiple successive simulations', default=1, type=int)
parser.add_argument('--random_agent', help='agent is taking random action', default=False, type=Util.str2bool)

# File to load
parser.add_argument('--file_to_load', help='name of the file to load NN weights or memory', default='')
parser.add_argument('--file_to_load2', help='name of the file to load NN weights or memory', default='')
# Load weights from file
parser.add_argument('--load_model',  help='load model to agent', default=False, type=Util.str2bool)
parser.add_argument('--load_all_weights', help='load full weights of neural networks', default=False, type=Util.str2bool)
parser.add_argument('--load_h1h2_weights', help='load hidden layer 1 and 2 weights of neural networks', default=False, type=Util.str2bool)
parser.add_argument('--load_h1_weights', help='load hidden layer 1 weights of neural networks', default=False, type=Util.str2bool)
parser.add_argument('--load_h2_weights', help='load h2 weights of neural networks', default=False, type=Util.str2bool)
parser.add_argument('--load_out_weights', help='load output weights of neural networks', default=False, type=Util.str2bool)
parser.add_argument('--load_h2out_weights', help='load h2 output weights of neural networks', default=False, type=Util.str2bool)
parser.add_argument('--load_h1out_weights', help='load h1 output weights of neural networks', default=False, type=Util.str2bool)
# Load experiences from file
parser.add_argument('--load_mem', help='load defined number of experiences to agent', default=0, type=int)
parser.add_argument('--load_mem2', help='load defined number of experiences to agent', default=0, type=int)

parser.add_argument('--seed', help='seed tuples', default='None', type=Util.str_to_intlist)
parser.add_argument('--save_seed', help='save simulation seed', default=True, type=Util.str2bool)
parser.add_argument('--save_record_rpu', help='save record simulation in run parallel universe case', default=False, type=Util.str2bool)

# Saving
parser.add_argument('--save_model_freq_ep', help='save neural networks model every defined episodes', default=0, type=int)
parser.add_argument('--save_mem_freq_ep', help='save memory every defined episodes', default=0, type=int) # TODO : not implemented, edit: not sure if really useful


parser.add_argument('--suffix', help='custom suffix to add', default='')
parser.add_argument('--dir_name', help='directory name to load NN files (to run in parallel universe)', default='')

# Collaborative Learning
parser.add_argument('--give_exp', help='CL agent 1 and agent 2 gives experiences to agent 0', default=False, type=Util.str2bool)
parser.add_argument('--tts', help='Time before agent 1 and agent 2 (RaceCircle Left/Right) stop giving experiences to agent 0 (RaceCombined)', default=2000, type=int)

# Race problem to solve
parser.add_argument('--cfg', help='game environment and agent\'s hyperparameters config file', required=True)
parser.add_argument('--cfg2', help='2nd agent environment and hyperparameters config file', default='')
parser.add_argument('--cfg3', help='3rd agent environment and hyperparameters config file', default='')

# Parameter object
args = parser.parse_args()

def sim_suffix():
    """
        Simulation suffix (for directory name, file name) given the simulation parameters
        :return: suffix
    """
    # Suffix
    suffix = ''
    if args.load_all_weights:
        suffix = 'loadall'
    elif args.load_h1h2_weights:
        suffix = 'loadh1h2'
    elif args.load_h1_weights:
        suffix = 'loadh1'
    elif args.load_model:
        suffix = 'loadmodel'
    elif args.load_h2_weights:
        suffix = 'loadh2'
    elif args.load_out_weights:
        suffix = 'loadout'
    elif args.load_h2out_weights:
        suffix = 'loadh2out'
    elif args.load_h1out_weights:
        suffix = 'loadh1out'

    # Load memory
    if args.load_mem:
        suffix += 'load' + str(args.load_mem) + 'xp'

        # Load memory 2
        if args.load_mem2:
            suffix += '_load_mem2_' + str(args.load_mem2) + 'xp'

    # Normal case
    if not suffix:
        suffix = 'normal'

    # No exploration
    if not args.exploration:
        suffix += '_noexplore'

    # Random agent
    if args.random_agent:
        suffix = 'random'

    # Number of agents
    if args.num_agents > 1:
        suffix += '_' + str(args.num_agents) + 'agents'

    # General purpose suffix
    if args.suffix:
        suffix += '_' + args.suffix

    return suffix

def sim_dir():
    """
        Simulation directory name given simulation parameters
        :return: directory name
    """
    multi_sim = args.multi_sim

    # Environment
    env = args.cfg.split('_')[0]
    if args.cfg2:
        env += '_' + args.cfg2.split('_')[0]
    if args.cfg3:
        env += '_' + args.cfg3.split('_')[0]

    # Number of trials
    max_ep = args.max_ep
    timestr = time.strftime('%Y%m%d_%H%M%S')

    if max_ep:
        max_ep_str = str(max_ep) + 'ep_'
    else:
        max_ep_str = ''

    suffix = sim_suffix()
    dir = './simulation_data/' + env + '_' + suffix + '_' + max_ep_str + str(multi_sim) + 'sim_' + timestr + '/'

    return dir