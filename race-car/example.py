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
        self.device = torch.device("cpu")  # Forcing CPU since the server environment is CPU-only
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
    Processes the raw game state dictionary, gets a model prediction,
    and occasionally overrides 'NOTHING' with 'ACCELERATE'.

    Args:
        state (dict): The current state of the game, received as a dictionary.

    Returns:
        list: A list containing the final action string for the API.
    """
    max_sensor_range = 1000.0
    sensor_data = state.get('sensors', [])

    # Robustly parse sensor data
    readings = []
    for reading in sensor_data:
        try:
            if reading is None:
                normalized_value = max_sensor_range / max_sensor_range
            else:
                normalized_value = float(reading) / max_sensor_range
            readings.append(normalized_value)
        except (ValueError, TypeError):
            readings.append(max_sensor_range / max_sensor_range)

    # Ensure the list is the correct size
    while len(readings) < NUM_SENSORS:
        readings.append(max_sensor_range / max_sensor_range)

    readings = readings[:NUM_SENSORS]

    # Convert to tensor and get model prediction
    state_tensor = torch.tensor(readings, dtype=torch.float32, device=agent.device).unsqueeze(0)

    with torch.no_grad():
        action_q_values = agent.policy_net(state_tensor)
        action_tensor = action_q_values.max(1)[1].view(1, 1)

    action_index = action_tensor.item()
    action_string = ACTION_MAP.get(action_index, 'NOTHING')

    # --- NEW LOGIC: Occasionally accelerate ---
    # If the model thinks it's safe to do nothing, there's a chance we can accelerate instead.
    if action_string == 'NOTHING':
        # Accelerate 50% of the time when the model says "NOTHING".
        # You can change 0.5 to a higher or lower value to make it more or less aggressive.
        if random.random() < 0.5:
            action_string = 'ACCELERATE'

    # Return the final action in the required list format
    return [action_string]


# This part below is for local testing with Pygame and will not be run by the API server.
if __name__ == '__main__':
    try:
        import pygame
        from src.game.core import initialize_game_state, game_loop

        print("\n--- Running local Pygame simulation for testing ---")
        seed_value = None
        pygame.init()
        initialize_game_state("http://localhost/api", seed_value)
        game_loop(agent=agent, verbose=True)
        pygame.quit()
    except ImportError:
        print("\nCould not import Pygame or src.game.core.")
        print("This is normal if you are running this script as part of the API server.")
        print("To run the local simulation, ensure Pygame is installed and the project structure is correct.")