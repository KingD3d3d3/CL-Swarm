# Environment type: Gym
# Algorithm: DQN

environment = {
    'env_name': 'MountainCar-v0',
    'max_ep': 2000
}

hyperparams = {
    'layers': (64, 64),
    'mem_capacity': 100000,
    'batch_size': 32,
    'eps_start': 1.,
    'eps_end': 0.1,
    'exploration_steps': 1000,
    'gamma': 0.99,
    'lr': 0.001,
    'update_target_steps': 1000,
    'use_double_dqn': True,
    'use_prioritized_experience_replay': False
}
