# core.py

import pygame
from time import sleep
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import namedtuple, deque
import os

from ..mathematics.randomizer import seed, random_choice, random_number
from ..elements.car import Car
from ..elements.road import Road
from ..elements.sensor import Sensor
from ..mathematics.vector import Vector

# Define constants
SCREEN_WIDTH = 1600
SCREEN_HEIGHT = 1200
LANE_COUNT = 5
MAX_TICKS = 60 * 60  # 60 seconds @ 60 fps
MAX_MS = 60 * 1000

# --- NEW: DQN Setup ---
# Define the structure of an experience replay tuple
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Define the action space for the DQN agent
# NOTE: We now use integer indices for actions to work with PyTorch
ACTION_MAP = {0: 'STEER_LEFT', 1: 'STEER_RIGHT', 2: 'NOTHING'}
NUM_ACTIONS = len(ACTION_MAP)


# Define the DQN model architecture
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        # A simple multi-layer perceptron (MLP)
        # Input layer: Number of sensor readings (n_observations)
        # Hidden Layer 1: 128 neurons, with ReLU activation
        # Hidden Layer 2: 128 neurons, with ReLU activation
        # Output Layer: Q-values for each possible action (n_actions)
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        """The forward pass of the network."""
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


# Define the Replay Memory for storing experiences
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Sample a random batch of transitions for training"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# The main Agent class that encapsulates the model and learning logic
class DQNAgent:
    def __init__(self, n_observations, n_actions, config):
        """
        Initializes the agent with a configuration dictionary.
        """
        # Hyperparameters are now sourced from the config dictionary
        self.batch_size = config['BATCH_SIZE']
        self.gamma = config['GAMMA']
        self.eps_start = config['EPS_START']
        self.eps_end = config['EPS_END']
        self.eps_decay = config['EPS_DECAY']
        self.tau = config['TAU']
        self.lr = config['LR']
        self.memory = ReplayMemory(config['MEMORY_CAPACITY'])

        self.n_actions = n_actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.policy_net = DQN(n_observations, n_actions).to(self.device)
        self.target_net = DQN(n_observations, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.steps_done = 0

    def select_action(self, state):
        """Selects an action using an epsilon-greedy policy."""
        sample = random.random()
        # Calculate the current epsilon value based on decay
        epsilon = self.eps_end + (self.eps_start - self.eps_end) * \
                  np.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1

        if sample > epsilon:
            # --- EXPLOIT: Choose the best action from the policy network ---
            with torch.no_grad():
                # state.max(1) returns the largest Q-value and its index for each batch item
                # We use .view(1, 1) to create a tensor of the correct shape for an action
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            # --- EXPLORE: Choose a random action ---
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        """Performs one step of optimization on the policy network."""
        if len(self.memory) < self.batch_size:
            return  # Don't train until we have enough samples

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))  # Converts batch-array of Transitions to Transition of batch-arrays.

        # Create batches of states, actions, and rewards
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device,
                                      dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken.
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]

        # Compute the expected Q values: R + γ * max_a' Q_target(s', a')
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)  # Gradient clipping
        self.optimizer.step()

    def update_target_net(self):
        """Soft update of the target network's weights."""
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (
                        1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

    def save_model(self, path="dqn_model.pth"):
        torch.save(self.policy_net.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path="dqn_model.pth"):
        if os.path.exists(path):
            self.policy_net.load_state_dict(torch.load(path))
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.policy_net.eval()  # Set model to evaluation mode
            self.target_net.eval()
            print(f"Model loaded from {path}")
        else:
            print(f"No model found at {path}, starting from scratch.")


def get_raw_state(game_state):
    """
    Extracts raw sensor readings and normalizes them to be the state.
    Returns a PyTorch tensor.
    """
    max_sensor_range = 1000.0  # A reasonable max value for normalization

    # Get all sensor readings, use max_range if a sensor sees nothing (None)
    readings = [
        (sensor.reading if sensor.reading is not None else max_sensor_range) / max_sensor_range
        for sensor in game_state.sensors
    ]

    # Ensure we always have a fixed size state (e.g., if sensors can be removed)
    # This example assumes a fixed number of sensors. If not, padding would be needed.

    # Convert to a PyTorch tensor and add a batch dimension
    state_tensor = torch.tensor(readings, dtype=torch.float32,
                                device="cuda" if torch.cuda.is_available() else "cpu").unsqueeze(0)
    return state_tensor


# Define game state
class GameState:
    def __init__(self, api_url: str):
        self.api_url = api_url
        self.ego = None
        self.cars = []
        self.car_bucket = []
        self.sensors = []
        self.road = None
        self.crashed = False
        self.distance = 0
        self.ticks = 0
        self.sensor_readings = []
        self.sensors_enabled = True


STATE = None


def intersects(rect1, rect2):
    return rect1.colliderect(rect2)


def handle_action(action: str):
    if action == "ACCELERATE":
        STATE.ego.speed_up()
    elif action == "DECELERATE":
        STATE.ego.slow_down()
    elif action == "STEER_LEFT":
        STATE.ego.turn(-0.1)
    elif action == "STEER_RIGHT":
        STATE.ego.turn(0.1)
    else:  # NOTHING
        pass


def update_cars():
    for car in STATE.cars:
        car.update(STATE.ego)


def remove_passed_cars():
    min_distance = -1000
    max_distance = SCREEN_WIDTH + 1000
    cars_to_keep = [car for car in STATE.cars if min_distance < car.x < max_distance]
    cars_to_retire = [car for car in STATE.cars if car not in cars_to_keep]

    for car in cars_to_retire:
        STATE.car_bucket.append(car)
        car.lane = None

    STATE.cars = cars_to_keep


def place_car():
    if len(STATE.cars) > LANE_COUNT:
        return

    open_lanes = [lane for lane in STATE.road.lanes if not any(c.lane == lane for c in STATE.cars if c != STATE.ego)]
    if not open_lanes: return

    lane = random_choice(open_lanes)
    x_offset = random_choice([-0.5, 1.5])  # Behind or in front
    speed_diff = random_number() * 5

    car = STATE.car_bucket.pop() if STATE.car_bucket else None
    if not car: return

    velocity_x = STATE.ego.velocity.x + speed_diff if x_offset == -0.5 else STATE.ego.velocity.x - speed_diff
    car.velocity = Vector(max(2, velocity_x), 0)  # Ensure cars don't stand still
    STATE.cars.append(car)

    car_sprite = car.sprite
    car.x = (SCREEN_WIDTH * x_offset) - (car_sprite.get_width() // 2)
    car.y = int((lane.y_start + lane.y_end) / 2 - car_sprite.get_height() / 2)
    car.lane = lane


# This now assumes a fixed number of sensors for the DQN input layer
NUM_SENSORS = 16


def initialize_game_state(api_url: str, seed_value: str = None, sensor_removal = 0):
    """
    Initializes the entire game environment for a new episode.
    """
    # --- FIX: Call the seed function UNCONDITIONALLY ---
    # The seed() function should handle a `None` value to ensure the
    # random number generator is always initialized.
    seed(seed_value)

    global STATE
    STATE = GameState(api_url)

    STATE.road = Road(SCREEN_WIDTH, SCREEN_HEIGHT, LANE_COUNT)
    middle_lane = STATE.road.middle_lane()
    lane_height = STATE.road.get_lane_height()

    ego_velocity = Vector(10, 0)
    STATE.ego = Car("yellow", ego_velocity, lane=middle_lane, target_height=int(lane_height * 0.8))
    # Start the car a bit to the left to give it more room initially
    STATE.ego.x = (SCREEN_WIDTH // 4) - (STATE.ego.sprite.get_width() // 2)
    STATE.ego.y = int((middle_lane.y_start + middle_lane.y_end) / 2 - STATE.ego.sprite.get_height() / 2)

    # Define a fixed set of sensors, distributed evenly in a circle
    sensor_options = [
        (angle, f"sensor_{i}") for i, angle in enumerate(np.linspace(0, 360, NUM_SENSORS, endpoint=False))
    ]

    STATE.sensors = [
        Sensor(STATE.ego, angle, name, STATE)
        for angle, name in sensor_options
    ]
    STATE.sensor_readings = [0.0] * len(STATE.sensors)

    # Create a bucket of cars to be placed on the road during the game
    for i in range(LANE_COUNT):
        car = Car(random_choice(["blue", "red"]), Vector(8, 0), target_height=int(lane_height * 0.8))
        STATE.car_bucket.append(car)

    # Start the game with only the ego car on the road
    STATE.cars = [STATE.ego]


def game_loop(agent: DQNAgent, verbose: bool = True):
    """Main game loop for a single episode, now driven by the DQN agent."""
    global STATE
    clock = pygame.time.Clock()
    screen = None
    if verbose:
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("DQN Race Car")

    while True:
        clock.tick(60)
        STATE.ticks += 1

        # 1. Observe the current state
        current_state_tensor = get_raw_state(STATE)

        # 2. Select and perform an action
        action_tensor = agent.select_action(current_state_tensor)
        action_index = action_tensor.item()
        action_string = ACTION_MAP[action_index]
        handle_action(action_string)

        # 3. Update the game simulation
        STATE.distance += STATE.ego.velocity.x
        update_cars()
        remove_passed_cars()
        place_car()
        for sensor in STATE.sensors:
            sensor.update()

        # 4. Check for collisions (done state)
        STATE.crashed = False
        for car in STATE.cars:
            if car != STATE.ego and intersects(STATE.ego.rect, car.rect):
                STATE.crashed = True
        for wall in STATE.road.walls:
            if intersects(STATE.ego.rect, wall.rect):
                STATE.crashed = True

        done = STATE.crashed or STATE.ticks > MAX_TICKS

        # 5. Define the reward
        reward = 0
        if STATE.crashed:
            reward = -100  # Severe penalty for crashing
        else:
            # --- NEW REWARD STRUCTURE ---
            # 1. Reward for forward speed (as before)
            velocity_reward = STATE.ego.velocity.x / 10.0

            # 2. Reward for staying alive
            survival_reward = 0.1  # A small, constant reward for every tick

            # 3. Penalty for being too close to the side walls
            # This encourages staying centered in the lane.
            lane_center = STATE.road.middle_lane().y_start + (STATE.road.get_lane_height() / 2)
            car_center_y = STATE.ego.y + (STATE.ego.get_bounds().height / 2)
            # Penalize based on the square of the distance from the center
            centering_penalty = -0.5 * ((car_center_y - lane_center) / (SCREEN_HEIGHT / 2)) ** 2

            reward = velocity_reward + survival_reward + centering_penalty

            # Small penalty for steering to encourage smooth driving
            if action_string != 'NOTHING':
                reward -= 0.05

        reward_tensor = torch.tensor([reward], device=agent.device)

        # 6. Observe the new state
        if not done:
            next_state_tensor = get_raw_state(STATE)
        else:
            next_state_tensor = None

        # 7. Store the transition in memory
        agent.memory.push(current_state_tensor, action_tensor, next_state_tensor, reward_tensor)

        # 8. Perform one step of the optimization (on the policy network)
        agent.optimize_model()

        # 9. Soft update of the target network's weights
        agent.update_target_net()

        # --- Rendering ---
        if verbose:
            screen.fill((0, 0, 0))
            screen.blit(STATE.road.surface, (0, 0))
            for car in STATE.cars:
                screen.blit(car.sprite, (car.x, car.y))
                bounds = car.get_bounds()
                color = (255, 0, 0) if car == STATE.ego else (0, 255, 0)
                pygame.draw.rect(screen, color, bounds, width=2)
            for sensor in STATE.sensors:
                sensor.draw(screen)
            pygame.display.flip()

        if done:
            break

    return STATE