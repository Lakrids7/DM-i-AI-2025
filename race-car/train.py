#example.py

import pygame
import csv
from collections import deque
import os
import datetime
import json
import numpy as np # <-- Make sure numpy is imported

from src.game.core import (
    initialize_game_state,
    game_loop,
    DQNAgent,
    NUM_SENSORS,
    NUM_ACTIONS
)

if __name__ == '__main__':
    # --- Centralized Configuration ---
    # All hyperparameters and settings are defined in this dictionary.
    # This dictionary will be saved as a JSON file for every run.
    config = {
        # Training settings
        "NUM_EPISODES": 15000,
        "SAVE_INTERVAL": 50,
        "LOG_INTERVAL": 10,

        # Agent Hyperparameters
        "BATCH_SIZE": 128,
        "GAMMA": 0.99,
        "EPS_START": 0.9,
        "EPS_END": 0.05,
        "EPS_DECAY": 1500000, # Slower decay for more exploration
        "TAU": 0.005,
        "LR": 1e-4,
        "MEMORY_CAPACITY": 10000,

        # Optional: Specify a model to load to continue training
        # Set to None to start a fresh training session
        "LOAD_MODEL_PATH": None
    }

    # --- 1. Create a Unique Directory for this Training Run ---
    run_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join("training_runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"--- Starting new training run: {run_name} ---")
    print(f"Logs and models will be saved in: {run_dir}")

    # Define file paths within the new directory
    MODEL_SAVE_PATH = os.path.join(run_dir, "dqn_model.pth")
    LOG_FILE = os.path.join(run_dir, 'training_log.csv')
    CONFIG_FILE = os.path.join(run_dir, 'config.json')

    # --- 2. Save the Configuration File ---
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to {CONFIG_FILE}")

    # --- 3. Setup CSV Logging ---
    recent_distances = deque(maxlen=50)
    recent_durations = deque(maxlen=50)

    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'distance', 'duration_ticks', 'crashed', 'epsilon'])

    # --- 4. Initialize Pygame and the DQN Agent ---
    pygame.init()

    # The agent is now initialized with the config dictionary
    agent = DQNAgent(n_observations=NUM_SENSORS, n_actions=NUM_ACTIONS, config=config)

    if config["LOAD_MODEL_PATH"]:
        agent.load_model(config["LOAD_MODEL_PATH"])

    # --- 5. Main Training Loop ---
    for episode in range(1, config["NUM_EPISODES"] + 1):
        initialize_game_state(seed_value=None)

        # Render every 20th episode to check progress visually
        show_game = (episode % 20 == 0) or (episode == 1)

        final_state = game_loop(agent=agent, verbose=show_game)

        # --- Process and Log Episode Results ---
        distance = final_state.distance
        duration = final_state.ticks
        crashed = final_state.crashed
        recent_distances.append(distance)
        recent_durations.append(duration)

        current_epsilon = agent.eps_end + (agent.eps_start - agent.eps_end) * \
            np.exp(-1. * agent.steps_done / agent.eps_decay)

        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, f"{distance:.2f}", duration, crashed, f"{current_epsilon:.4f}"])

        if episode % config["LOG_INTERVAL"] == 0:
            avg_dist = sum(recent_distances) / len(recent_distances)
            avg_dur = sum(recent_durations) / len(recent_durations)
            print(
                f"Ep: {episode}/{config['NUM_EPISODES']} | "
                f"Avg Dist (Last {len(recent_distances)}): {avg_dist:.2f} | "
                f"Avg Ticks: {avg_dur:.1f} | "
                f"Epsilon: {current_epsilon:.4f}"
            )

        # Save the model periodically to the run-specific directory
        if episode % config["SAVE_INTERVAL"] == 0:
            agent.save_model(MODEL_SAVE_PATH)

    # Final Save
    print("--- Training finished. Saving final model. ---")
    agent.save_model(MODEL_SAVE_PATH)

    pygame.quit()

    pygame.quit()