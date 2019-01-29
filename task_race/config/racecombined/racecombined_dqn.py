# Environment type: Custom Race
# Algorithm: DQN

environment = {
    'env_name': 'RaceCombined',
    'solved_timesteps': 250,
    'max_ep': 2000
}

hyperparams = {
    'eps_start': 1.,
    'eps_end': 0.1,                     # optimized
    'exploration_steps': 10000,         # optimized
    'gamma': 0.99,                      # optimized
    'mem_capacity': 100000,             # optimized
    'layers': (64, 64),                 # optimized
    'use_double_dqn': True,
    'update_target_steps': 1000,
    'batch_size': 32,
    'lr': 0.001
}