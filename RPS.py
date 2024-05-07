import random
import numpy as np

# Q-learning parameters
learning_rate = 0.15      # Learning rate (alpha)
discount_factor = 0.8     # Discount factor (gamma)
exploration_rate = 0.1    # Exploration rate (epsilon)

# History configuration
history_length = 3  # Number of previous moves to consider

# Initialize the Q-table
q_table = {}
agent_moves = []
previous_state = None

def player(opponent_last_move, opponent_moves=[]):
    """
    Determine the next move to make using Q-learning.

    Args:
        opponent_last_move (str): The opponent's previous move ("R", "P", or "S").
        opponent_moves (list of str): The history of the opponent's previous moves.

    Returns:
        str: The next move to make ("R", "P", or "S").
    """
    global agent_moves, q_table, previous_state

    if opponent_last_move == "":
        move = random.choice(["R", "P", "S"])
        agent_moves.append(move)
        return move

    opponent_moves.append(opponent_last_move)
    opponent_moves = opponent_moves[-history_length:]
    agent_moves = agent_moves[-history_length:]

    current_state = create_state(opponent_moves, agent_moves)

    if len(opponent_moves) < history_length or len(agent_moves) < history_length:
        move = random.choice(["R", "P", "S"])
        agent_moves.append(move)
        previous_state = current_state
        return move

    if previous_state is not None:
        update_q_table(previous_state, agent_moves[-1], calculate_reward(opponent_last_move, agent_moves[-1]))

    if random.uniform(0, 1) < exploration_rate:
        move = random.choice(["R", "P", "S"])
    else:
        q_table.setdefault(current_state, {"R": 0, "P": 0, "S": 0})
        best_move = max(q_table[current_state], key=q_table[current_state].get)
        move = best_move

    agent_moves.append(move)
    previous_state = current_state
    return move

def create_state(opponent_moves, agent_moves):
    """
    Create a state string from the move histories.

    Args:
        opponent_moves (list of str): The history of the opponent's previous moves.
        agent_moves (list of str): The history of the agent's previous moves.

    Returns:
        str: The concatenated state string.
    """
    return "".join(opponent_moves) + "".join(agent_moves)

def update_q_table(state, move, reward):
    """
    Update the Q-table with the new reward for the given state and move.

    Args:
        state (str): The current state string.
        move (str): The move made ("R", "P", or "S").
        reward (int): The reward received for the move.
    """
    global q_table

    q_table.setdefault(state, {"R": 0, "P": 0, "S": 0})
    current_q = q_table[state][move]
    max_future_q = max(q_table[state].values())
    updated_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount_factor * max_future_q)
    q_table[state][move] = updated_q

def calculate_reward(opponent_move, agent_move):
    """
    Calculate the reward based on the moves of the opponent and the agent.

    Args:
        opponent_move (str): The opponent's move ("R", "P", or "S").
        agent_move (str): The agent's move ("R", "P", or "S").

    Returns:
        int: The reward received (1 for win, 0 for tie, -1 for loss).
    """
    if agent_move == opponent_move:
        return 0
    elif (agent_move == "R" and opponent_move == "S") or \
            (agent_move == "P" and opponent_move == "R") or \
            (agent_move == "S" and opponent_move == "P"):
        return 1
    else:
        return -1
