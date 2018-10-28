# Environment type: Gym
# Algorithm: DQN

environment = {
    'env_name': 'LunarLanderNoShaping-v2',
    'solved_score': 90.0,
    'max_ep': 100000
}

hyperparams = {
    'layers': (64, 64),
    'mem_capacity': 100000,
    'batch_size': 32,
    'eps_start': 1.,
    'eps_end': 0.05,
    'exploration_steps': 10000,
    'gamma': 0.99,
    'lr': 0.001,
    'update_target_steps': 1000,
    'use_double_dqn': True,
    'use_prioritized_experience_replay': False
}