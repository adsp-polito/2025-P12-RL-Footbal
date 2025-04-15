import numpy as np

def random_strategy(env):
    """
    Returns a random action for each player.
    Players move in random directions; the possessor may pass or shoot.
    """
    # Initialize each player with a random movement action (0-8)
    actions = [("move", np.random.randint(0, 9)) for _ in range(22)]
    
    # Check if any player currently has possession of the ball
    possessor_id = env.ball.owner_id

    # If a player has possession, decide randomly whether to pass or shoot
    if possessor_id is not None:
        decision = np.random.rand()
        # Select a random teammate to receive the pass
        if decision < 0.5:
            receiver_id = np.random.choice([i for i in range(22) if i != possessor_id])
            # Assign pass or shoot action to the possessor based on decision
            actions[possessor_id] = ("pass", receiver_id)
            print(f"Player {possessor_id} passes to Player {receiver_id}")
        else:
            # Assign pass or shoot action to the possessor based on decision
            actions[possessor_id] = ("shoot", None)
            print(f"Player {possessor_id} shoots")

    return actions

def pass_strategy(env):
    """
    Returns an action where all players are idle except the possessor who passes to a random teammate.
    """
    # Initialize each player with an idle action (0)
    actions = [("move", 0) for _ in range(22)]
    
    # Check if any player currently has possession of the ball
    possessor_id = env.ball.owner_id

    # If a player has possession, decide randomly whether to pass or shoot
    if possessor_id is not None:
        # Select a random teammate to receive the pass
        receiver_id = np.random.choice([i for i in range(22) if i != possessor_id])
        # Assign pass action to the possessor and idle action to others
        actions[possessor_id] = ("pass", receiver_id)
        print(f"Player {possessor_id} passes to Player {receiver_id}")

    return actions