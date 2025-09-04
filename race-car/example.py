# example.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os
from collections import namedtuple, deque

# --- NECESSARY CLASSES COPIED FROM core.py ---
# These classes are needed for the DQNAgent to be loaded and used.

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self, n_observations, n_actions, config):
        self.batch_size = config['BATCH_SIZE']
        self.gamma = config['GAMMA']
        self.eps_start = config['EPS_START']
        self.eps_end = config['EPS_END']
        self.eps_decay = config['EPS_DECAY']
        self.tau = config['TAU']
        self.lr = config['LR']
        self.memory = ReplayMemory(config['MEMORY_CAPACITY'])

        self.n_actions = n_actions
        self.device = torch.device("cpu") # Forcing CPU since the server environment is CPU-only
        print(f"Using device: {self.device}")

        self.policy_net = DQN(n_observations, n_actions).to(self.device)
        self.target_net = DQN(n_observations, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.steps_done = 0

    def load_model(self, path="dqn_model.pth"):
        if os.path.exists(path):
            # Load model onto the CPU, regardless of where it was trained
            self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.policy_net.eval()
            self.target_net.eval()
            print(f"Model loaded from {path}")
        else:
            print(f"No model found at {path}, starting from scratch.")

# --- CONSTANTS AND MODEL INITIALIZATION ---

# These must match the parameters used during training
NUM_SENSORS = 16
NUM_ACTIONS = 3  # The model was trained to output one of 3 actions

# The action map must match the output of the neural network
ACTION_MAP = {0: 'STEER_LEFT', 1: 'STEER_RIGHT', 2: 'NOTHING'}

# Define the configuration your agent was trained with.
config = {
    "BATCH_SIZE": 128,
    "GAMMA": 0.99,
    "EPS_START": 0.0,
    "EPS_END": 0.0,
    "EPS_DECAY": 100000,
    "TAU": 0.005,
    "LR": 1e-4,
    "MEMORY_CAPACITY": 10000,
}

# Initialize the agent globally. This ensures the model is loaded only ONCE when the server starts.
agent = DQNAgent(n_observations=NUM_SENSORS, n_actions=NUM_ACTIONS, config=config)

# --- IMPORTANT: Update this path to point to your actual trained model file ---
TRAINED_MODEL_PATH = "training_runs/2025-09-03_15-49-52/dqn_model.pth"
agent.load_model(TRAINED_MODEL_PATH)

print("--- Trained model loaded successfully. Ready for inference. ---")


def return_action(state: dict):
    """
    Processes the raw game state dictionary from the API request and returns an action.

    Args:
        state (dict): The current state of the game, received as a dictionary.

    Returns:
        dict: A dictionary containing the action chosen by the model.
    """
    # --- 1. PREPROCESS THE INPUT ---
    # This logic is based on the 'get_raw_state' function from your core.py
    max_sensor_range = 1000.0  # Use the same normalization value as during training

    # Extract sensor readings from the input dictionary.
    # The key 'sensors' might be different; check your project's DTOs if this fails.
    sensor_data = state.get('sensors', [])

    # Create a clean, normalized list of sensor readings
    readings = [
        (sensor['reading'] if sensor.get('reading') is not None else max_sensor_range) / max_sensor_range
        for sensor in sensor_data
    ]

    # --- 2. CONVERT TO TENSOR ---
    # Create the tensor from the clean list of numbers
    state_tensor = torch.tensor(readings, dtype=torch.float32, device=agent.device).unsqueeze(0)

    # --- 3. GET ACTION FROM MODEL ---
    # Use torch.no_grad() for faster inference as we don't need to calculate gradients
    with torch.no_grad():
        # Get the Q-values from the policy network
        action_q_values = agent.policy_net(state_tensor)
        # Select the action with the highest Q-value
        action_tensor = action_q_values.max(1)[1].view(1, 1)

    action_index = action_tensor.item()
    action_string = ACTION_MAP.get(action_index, 'NOTHING') # Default to 'NOTHING' if index is out of bounds

    # --- 4. FORMAT THE OUTPUT FOR THE API ---
    # The API expects a dictionary in a specific format
    return {
        "action_type": "string",
        "actions": [action_string]
    }


# This part below is for local testing with Pygame and will not be run by the API server.
if __name__ == '__main__':
    # This requires the 'src.game.core' module to be available
    try:
        import pygame
        from src.game.core import initialize_game_state, game_loop

        print("\n--- Running local Pygame simulation for testing ---")
        seed_value = None
        pygame.init()
        # Note: The api_url is just a placeholder for local testing
        initialize_game_state("http://localhost/api", seed_value)
        # The game_loop will need to be adapted to use the agent defined in this file
        game_loop(agent=agent, verbose=True)
        pygame.quit()
    except ImportError:
        print("\nCould not import Pygame or src.game.core.")
        print("This is normal if you are running this script as part of the API server.")
        print("To run the local simulation, ensure Pygame is installed and the project structure is correct.")