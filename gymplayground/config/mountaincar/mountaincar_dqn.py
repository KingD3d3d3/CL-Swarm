# Environment type: Gym
# Algorithm: DQN

environment = {
    'env_name': 'MountainCar-v0',
    'solved_score': -110.0,
    'max_ep': 2000
}

hyperparams = {
    'eps_start': 1.,
    'eps_end': 0.01,                    # optimized
    'exploration_steps': 10000,         # optimized
    'gamma': 0.999,                     # optimized
    'mem_capacity': 100000,             # optimized
    'layers': (64, 64),                 # optimized
    'use_double_dqn': True,
    'update_target_steps': 1000,
    'batch_size': 32,
    'lr': 0.001
}