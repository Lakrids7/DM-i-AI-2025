import pandas as pd
import matplotlib.pyplot as plt

# --- Step 1: Specify the path to your CSV file ---
# Replace 'training_log.csv' with the actual name of your file.
# If the file is not in the same directory as the script, provide the full path.
csv_file_path = 'training_log.csv'

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

# Create a figure with 3 subplots, arranged in 1 row and 3 columns.
# figsize is set to have a nice aspect ratio for the three plots.
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# --- Plot 1: Distance vs. Episode ---
ax1.plot(df['episode'], df['distance'], marker='.', linestyle='-', color='b')
ax1.set_title('Distance over Episodes')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Distance')
ax1.grid(True)

# --- Plot 2: Duration Ticks vs. Episode ---
ax2.plot(df['episode'], df['duration_ticks'], marker='.', linestyle='-', color='g')
ax2.set_title('Duration over Episodes')
ax2.set_xlabel('Episode')
ax2.set_ylabel('Duration Ticks')
ax2.grid(True)

# --- Plot 3: Duration Ticks vs. Distance ---
# A scatter plot is better here to see correlation
ax3.scatter(df['distance'], df['duration_ticks'], alpha=0.6, color='r')
ax3.set_title('Duration vs. Distance')
ax3.set_xlabel('Distance')
ax3.set_ylabel('Duration Ticks')
ax3.grid(True)

# Add a main title for the entire figure
fig.suptitle('Training Performance Analysis', fontsize=16)

# Adjust layout to prevent titles and labels from overlapping
plt.tight_layout(rect=[0, 0, 1, 0.96]) # rect leaves space for suptitle

# --- Step 5: Display the plots ---
plt.show()

print("Plot window closed.")