# Environment type: Gym
# Algorithm: DQN

environment = {
    'env_name': 'MountainCar-v0',
    'solved_score': -110.0,
    'max_ep': 5000
}

hyperparams = {
    'layers': (32, 32),
    'mem_capacity': 100000,
    'batch_size': 32,
    'eps_start': 1.,
    'eps_end': 0.01,
    'exploration_steps': 10000,
    'gamma': 0.99,
    'lr': 0.001,
    'update_target_steps': 1000,
    'use_double_dqn': True,
    'use_prioritized_experience_replay': False
}