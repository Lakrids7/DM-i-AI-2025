import pygame
import random
from collections import deque
import csv
from src.game import core
from src.game.core import (
    initialize_game_state,
    game_loop,
    save_q_table,
    load_q_table,
    STATE,
    q_table,
)
import numpy as np

'''
Set seed_value to None for random seed.
Within game_loop, change get_action() to your custom models prediction for local testing and training.
'''

def return_action(state):
    # Returns a list of actions

    actions = []
    action_choices = ['ACCELERATE', 'DECELERATE', 'STEER_LEFT', 'STEER_RIGHT', 'NOTHING']
    for _ in range(10):
        actions.append(random.choice(action_choices))
    return actions

if __name__ == '__main__':
    # --- Training Parameters ---
    NUM_EPISODES = 100000
    SAVE_INTERVAL = 5
    LOG_INTERVAL = 10  # How often to print the detailed log to the console
    Q_TABLE_FILENAME = "q_table_2.json"

    # --- NEW: Logging Setup ---
    LOG_FILE = 'training_log.csv'
    # Use a deque for an efficient rolling average of the last 50 episodes
    recent_distances = deque(maxlen=50)
    recent_durations = deque(maxlen=50)

    # Create the CSV log file and write the header if it doesn't exist
    try:
        with open(LOG_FILE, 'x', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'distance', 'duration_ticks', 'crashed', 'epsilon', 'q_table_size'])
        print(f"Created new log file: {LOG_FILE}")
    except FileExistsError:
        print(f"Appending to existing log file: {LOG_FILE}")
    # --- END Logging Setup ---

    pygame.init()
    seed_value = None

    # Load the agent's previous knowledge
    load_q_table(Q_TABLE_FILENAME)

    # --- Training Loop ---
    for episode in range(1, NUM_EPISODES + 1):

        # Initialize the game state for a new episode
        initialize_game_state("http://example.com/api/predict", seed_value)

        # Determine if this episode should be rendered
        show_game = (episode % 20 == 0) or (episode == 1)
        #show_game = True

        # Run the game loop and capture the final state returned by the function
        final_state = game_loop(verbose=show_game)

        # --- NEW: Process and Log Episode Results ---
        distance = final_state.distance
        duration = final_state.ticks
        crashed = final_state.crashed

        # Add current scores to the deques for rolling average calculation
        recent_distances.append(distance)
        recent_durations.append(duration)

        # Append the results of this episode to the CSV file
        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, f"{distance:.2f}", duration, crashed, f"{core.EPSILON:.4f}", len(q_table)])

        # --- NEW: Improved Console Logging ---
        if episode % LOG_INTERVAL == 0:
            avg_dist = sum(recent_distances) / len(recent_distances)
            avg_dur = sum(recent_durations) / len(recent_durations)

            print(
                f"Ep: {episode}/{NUM_EPISODES} | "
                f"Avg Dist (Last {len(recent_distances)}): {avg_dist:.2f} | "
                f"Avg Ticks: {avg_dur:.1f} | "
                f"Epsilon: {core.EPSILON:.4f} | "  # <-- TO THIS
                f"Q-States: {len(q_table)}"
            )

        # Save the Q-table periodically
        if episode % SAVE_INTERVAL == 0:
            print(f"--- Saving Q-table with {len(q_table)} states... ---")
            # --- FIXED TYPO: .jso -> .json ---
            save_q_table(Q_TABLE_FILENAME)

    # Final Save
    print("--- Training finished. Saving final Q-table. ---")
    save_q_table(Q_TABLE_FILENAME)

    pygame.quit()