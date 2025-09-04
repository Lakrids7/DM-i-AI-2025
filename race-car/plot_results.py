import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


csv_file_path = 'training_runs/2025-09-03_15-49-52/training_log.csv'

# --- Step 2: Load the data from the CSV file ---
try:
    df = pd.read_csv(csv_file_path)
except FileNotFoundError:
    print(f"Error: The file '{csv_file_path}' was not found.")
    print("Please make sure the file name and path are correct.")
    exit()

# --- Step 3: Check if the required columns exist ---
required_columns = ['episode', 'distance', 'duration_ticks']
if not all(col in df.columns for col in required_columns):
    print("Error: The CSV file must contain the columns: 'episode', 'distance', and 'duration_ticks'.")
    exit()

# --- Step 4: Create the 2D plots ---
print("Generating 2D plots...")

# Create a figure with 2 subplots, arranged in 1 row and 2 columns.
# figsize is set to have a nice aspect ratio for the two plots.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))

# --- Plot 1: Distance vs. Episode ---
ax1.plot(df['episode'], df['distance'], marker='.', linestyle='-', color='b', label='Data')
ax1.set_title('Distance over Episodes')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Distance')
ax1.grid(True)

# --- Add regression line to Plot 1 ---
# Calculate the coefficients of the regression line (degree 1 polynomial)
m1, b1 = np.polyfit(df['episode'], df['distance'], 1)
# Add the regression line to the plot
ax1.plot(df['episode'], m1*df['episode'] + b1, color='red', linestyle='--', label=f'Regression Line: y={m1:.2f}x+{b1:.2f}')
ax1.legend()


# --- Plot 2: Duration Ticks vs. Episode ---
ax2.plot(df['episode'], df['duration_ticks'], marker='.', linestyle='-', color='g', label='Data')
ax2.set_title('Duration over Episodes')
ax2.set_xlabel('Episode')
ax2.set_ylabel('Duration Ticks')
ax2.grid(True)

# --- Add regression line to Plot 2 ---
# Calculate the coefficients of the regression line
m2, b2 = np.polyfit(df['episode'], df['duration_ticks'], 1)
# Add the regression line to the plot
ax2.plot(df['episode'], m2*df['episode'] + b2, color='purple', linestyle='--', label=f'Regression Line: y={m2:.2f}x+{b2:.2f}')
ax2.legend()


# Add a main title for the entire figure
fig.suptitle('Training Performance Analysis with Regression', fontsize=16)

# Adjust layout to prevent titles and labels from overlapping
plt.tight_layout(rect=[0, 0, 1, 0.96]) # rect leaves space for suptitle

# --- Step 5: Display the plots ---
plt.show()

print("Plot window closed.")