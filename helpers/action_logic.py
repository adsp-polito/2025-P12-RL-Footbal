import numpy as np

# RANDOM STRATEGIES
def random_passing(env, frame):
    actions = [0] * 22
    possessor_id = env.ball.owner_id

    if possessor_id is not None:
        receiver_id = np.random.choice([i for i in range(22) if i != possessor_id])
        actions[possessor_id] = ("pass", receiver_id)

    return actions

def random_action(env, frame):
    """
    Randomly selects an action for each player in the environment.
    """
    actions = np.random.randint(0, 9, size=22).tolist()  # Random action from 0 to 8
    possessor_id = env.ball.owner_id

    if possessor_id is not None:
        receiver_id = np.random.choice([i for i in range(22) if i != possessor_id])
        actions[possessor_id] = ("pass", receiver_id)

    return actions


# STRATEGY REGISTRY
ACTIONS = {
    "random": random_action,
    "random_passing": random_passing
}