import numpy as np

def random_passing(env, frame):
    actions = [0] * 22
    possessor_id = env.ball.owner_id

    if possessor_id is not None:
        receiver_id = np.random.choice([i for i in range(22) if i != possessor_id])
        actions[possessor_id] = ("pass", receiver_id)

    return actions