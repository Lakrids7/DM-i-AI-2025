import pygame
import random
from src.game.core import initialize_game_state, game_loop, DQNAgent, NUM_SENSORS, NUM_ACTIONS
import torch
import numpy as np

# --- 1. Initialize the Agent and Load the Trained Model ---

# Define the configuration your agent was trained with.
# Most of these values do not matter for inference, but the agent expects them.
config = {
    "BATCH_SIZE": 128,
    "GAMMA": 0.99,
    "EPS_START": 0.0,  # Epsilon starts at 0 for pure exploitation
    "EPS_END": 0.0,    # Epsilon ends at 0
    "EPS_DECAY": 100000,
    "TAU": 0.005,
    "LR": 1e-4,
    "MEMORY_CAPACITY": 10000,
}

# Initialize the agent
agent = DQNAgent(n_observations=NUM_SENSORS, n_actions=NUM_ACTIONS, config=config)

TRAINED_MODEL_PATH = "training_runs/2025-09-03_15-49-52/dqn_model.pth"
agent.load_model(TRAINED_MODEL_PATH)
agent.policy_net.eval()  # Set the model to evaluation mode

print("--- Trained model loaded successfully. Ready for inference. ---")


def return_action(state):
    """
    Uses the trained DQN model to return the best action for the given state.

    Args:
        state (np.ndarray): The current state of the game from the sensors.

    Returns:
        str: The action selected by the model.
    """
    # Action space mapping from integer output of the model to game action strings
    action_map = {
        0: 'STEER_LEFT',
        1: 'STEER_RIGHT',
        2: 'NOTHING',
        3: 'ACCELERATE',
        4: 'DECELERATE'
    }

    state_tensor = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0)

    # The agent will automatically use its policy network to predict the action
    # with the highest Q-value because epsilon is 0.
    action_tensor = agent.select_action(state_tensor)


    action_index = action_tensor.item()

    return action_map[action_index]






if __name__ == '__main__':
    seed_value = None
    pygame.init()
    initialize_game_state("http://example.com/api/predict", seed_value)
    game_loop(agent=agent, verbose=True) # For pygame window
    pygame.quit()