#core.py

import pygame
from time import sleep
#import requests
#from typing import List, Optional
from ..mathematics.randomizer import seed, random_choice, random_number
from ..elements.car import Car
from ..elements.road import Road
from ..elements.sensor import Sensor
from ..mathematics.vector import Vector
import json
import numpy as np

# Define constants
SCREEN_WIDTH = 1600
SCREEN_HEIGHT = 1200
LANE_COUNT = 5
CAR_COLORS = ['yellow', 'blue', 'red']
MAX_TICKS = 60 * 60  # 60 seconds @ 60 fps
MAX_MS = 60 * 1000600   # 60 seconds flat

# Q-learning parameters
env_actions = ['STEER_LEFT', 'STEER_RIGHT', 'NOTHING']
#env_states = ['LEFT_SIDE_CLOSE', 'RIGHT_SIDE_CLOSE', 'FRONT_SIDE_CLOSE', 'BACK_SIDE_CLOSE']


# Hyperparameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95
EPSILON = 0.9
EPSILON_DECAY = 0.999
MIN_EPSILON = 0.05
q_table = {}

#TODO: Add delay for actions, so it doesn't perform an action every tick?


def save_q_table(filename="q_table.json"):
    """Saves the Q-table dictionary to a JSON file."""
    try:
        with open(filename, 'w') as f:
            # Convert tuple keys to strings because JSON does not support tuple keys
            string_keyed_q_table = {str(k): v for k, v in q_table.items()}
            json.dump(string_keyed_q_table, f, indent=4)
        print(f"Q-table successfully saved to {filename}")
    except Exception as e:
        print(f"Error saving Q-table: {e}")

def load_q_table(filename="q_table.json"):
    """Loads a Q-table from a JSON file."""
    global q_table
    try:
        with open(filename, 'r') as f:
            string_keyed_q_table = json.load(f)
            q_table = {eval(k): v for k, v in string_keyed_q_table.items()}
        print(f"Q-table successfully loaded from {filename}")
    except FileNotFoundError:
        print(f"No Q-table file found at {filename}. Starting with a new empty table.")
    except Exception as e:
        print(f"Error loading Q-table: {e}")


# In core.py

def get_simplified_state(state):
    """
    Converts sensor readings into a binary state for 8 directional sectors.
    The state for each sector is either 1 (Close) or 0 (Far).

    Returns a tuple representing the 8 sectors in clockwise order starting from the front.
    """
    # --- FINAL SIMPLIFIED LOGIC ---

    # This is the ONLY threshold that matters. If an object is closer than this,
    # the state is 1 (Close). Otherwise, it's 0 (Far).
    CLOSE_THRESHOLD = 300

    # This is NOT a threshold. It is just a default value for when sensors see nothing.
    DEFAULT_FALLBACK_DISTANCE = 1000

    # The eight directional sensor groups with 3 sensors each.
    # Some sensors are shared between adjacent groups for overlap.
    SENSOR_GROUPS = {
        'front': ['front_left_front', 'front', 'front_right_front'],
        'front_right': ['front_right_front', 'right_front', 'right_side_front'],
        'right': ['right_side_front', 'right_side', 'right_side_back'],
        'back_right': ['right_side_back', 'right_back', 'back_right_back'],
        'back': ['back_right_back', 'back', 'back_left_back'],
        'back_left': ['back_left_back', 'left_back', 'left_side_back'],
        'left': ['left_side_back', 'left_side', 'left_side_front'],
        'front_left': ['left_side_front', 'left_front', 'front_left_front']
    }

    sensor_readings = {sensor.name: sensor.reading for sensor in state.sensors}

    # Helper function to get the minimum valid distance for a group
    def get_min_distance_for_group(group_name):
        valid_distances = [
            sensor_readings.get(sensor_name)
            for sensor_name in SENSOR_GROUPS[group_name]
            if sensor_readings.get(sensor_name) is not None
        ]
        if valid_distances:
            return min(valid_distances)
        else:
            return DEFAULT_FALLBACK_DISTANCE

    # Get the nearest detected object distance for each sector
    min_front = get_min_distance_for_group('front')
    min_front_right = get_min_distance_for_group('front_right')
    min_right = get_min_distance_for_group('right')
    min_back_right = get_min_distance_for_group('back_right')
    min_back = get_min_distance_for_group('back')
    min_back_left = get_min_distance_for_group('back_left')
    min_left = get_min_distance_for_group('left')
    min_front_left = get_min_distance_for_group('front_left')


    # --- The Binary State Calculation ---
    # Convert each distance to a simple 0 or 1.
    front_state = 1 if min_front < CLOSE_THRESHOLD else 0
    front_right_state = 1 if min_front_right < CLOSE_THRESHOLD else 0
    right_state = 1 if min_right < CLOSE_THRESHOLD else 0
    back_right_state = 1 if min_back_right < CLOSE_THRESHOLD else 0
    back_state = 1 if min_back < CLOSE_THRESHOLD else 0
    back_left_state = 1 if min_back_left < CLOSE_THRESHOLD else 0
    left_state = 1 if min_left < CLOSE_THRESHOLD else 0
    front_left_state = 1 if min_front_left < CLOSE_THRESHOLD else 0


    # Return the final state tuple.
    return (front_state, front_right_state, right_state, back_right_state,
            back_state, back_left_state, left_state, front_left_state)




# Define game state
class GameState:
    def __init__(self, api_url: str):
        self.ego = None
        self.cars = []
        self.car_bucket = []
        self.sensors = []
        self.road = None
        self.statistics = None
        self.sensors_enabled = True
        self.api_url = api_url
        self.crashed = False
        self.elapsed_game_time = 0
        self.distance = 0
        self.latest_action = "NOTHING"
        self.ticks = 0

STATE = None


def intersects(rect1, rect2):
    return rect1.colliderect(rect2)

# Game logic
def handle_action(action: str):
    if action == "ACCELERATE":
        STATE.ego.speed_up()
    elif action == "DECELERATE":
        STATE.ego.slow_down()
    elif action == "STEER_LEFT":
        STATE.ego.turn(-0.1)
    elif action == "STEER_RIGHT":
        STATE.ego.turn(0.1)
    else:
        pass

def update_cars():
    for car in STATE.cars:
        car.update(STATE.ego)


def remove_passed_cars():
    min_distance = -1000
    max_distance = SCREEN_WIDTH + 1000
    cars_to_keep = []
    cars_to_retire = []

    for car in STATE.cars:
        if car.x < min_distance or car.x > max_distance:
            cars_to_retire.append(car)
        else:
            cars_to_keep.append(car)

    for car in cars_to_retire:
        STATE.car_bucket.append(car)
        car.lane = None

    STATE.cars = cars_to_keep

def place_car():
    if len(STATE.cars) > LANE_COUNT:
        return

    speed_coeff_modifier = 5
    x_offset_behind = -0.5
    x_offset_in_front = 1.5

    open_lanes = [lane for lane in STATE.road.lanes if not any(c.lane == lane for c in STATE.cars if c != STATE.ego)]
    lane = random_choice(open_lanes)
    x_offset = random_choice([x_offset_behind, x_offset_in_front])
    horizontal_velocity_coefficient = random_number() * speed_coeff_modifier

    car = STATE.car_bucket.pop() if STATE.car_bucket else None
    if not car:
        return

    velocity_x = STATE.ego.velocity.x + horizontal_velocity_coefficient if x_offset == x_offset_behind else STATE.ego.velocity.x - horizontal_velocity_coefficient
    car.velocity = Vector(velocity_x, 0)
    STATE.cars.append(car)

    car_sprite = car.sprite
    car.x = (SCREEN_WIDTH * x_offset) - (car_sprite.get_width() // 2)
    car.y = int((lane.y_start + lane.y_end) / 2 - car_sprite.get_height() / 2)
    car.lane = lane


def get_action_json():
    """
    Get action depending on tick from the actions_log.json.
    Finds the action for the current STATE.ticks.
    """
    try:
        with open("actions_log.json", "r") as f:
            actions = json.load(f)
            for entry in actions:
                if entry.get("tick") == STATE.ticks:
                    return entry.get("action", "NOTHING")
            return "NOTHING"
    except FileNotFoundError:
        return "NOTHING"


def initialize_game_state( api_url: str, seed_value: str, sensor_removal = 0):
    seed(seed_value)
    global STATE
    STATE = GameState(api_url)

    # Create environment
    STATE.road = Road(SCREEN_WIDTH, SCREEN_HEIGHT, LANE_COUNT)
    middle_lane = STATE.road.middle_lane()
    lane_height = STATE.road.get_lane_height()

    # Create ego car
    ego_velocity = Vector(10, 0)
    STATE.ego = Car("yellow", ego_velocity, lane=middle_lane, target_height=int(lane_height * 0.8))
    ego_sprite = STATE.ego.sprite
    STATE.ego.x = (SCREEN_WIDTH // 2) - (ego_sprite.get_width() // 2)
    STATE.ego.y = int((middle_lane.y_start + middle_lane.y_end) / 2 - ego_sprite.get_height() / 2)
    sensor_options = [
            #Front group
            (90, "front"),
            (67.5, "front_left_front"), #IGNORE
            (112.5, "front_right_front"), #IGNORE

            #Back group
            (270, "back"),
            (247.5, "back_right_back"), #IGNORE
            (292.5, "back_left_back"), #IGNORE

            #Right group
            (157.5, "right_side_front"),
            (202.5, "right_side_back"),
            (135, "right_front"),
            (180, "right_side"),
            (225, "right_back"),

            #Left group
            (315, "left_back"),
            (0, "left_side"),
            (45, "left_front"),
            (22.5, "left_side_front"),
            (337.5, "left_side_back"),

        ]

    for _ in range(sensor_removal): # Removes random sensors
        random_sensor = random_choice(sensor_options)
        sensor_options.remove(random_sensor)
    STATE.sensors = [
        Sensor(STATE.ego, angle, name, STATE)
        for angle, name in sensor_options
    ]

    # Create other cars and add to car bucket
    for i in range(0, LANE_COUNT - 1):
        car_colors = ["blue", "red"]
        color = random_choice(car_colors)
        car = Car(color, Vector(8, 0), target_height=int(lane_height * 0.8))
        STATE.car_bucket.append(car)

    STATE.cars = [STATE.ego]

def update_game(current_action: str):
    handle_action(current_action)
    STATE.distance += STATE.ego.velocity.x
    update_cars()
    remove_passed_cars()
    place_car()
    for sensor in STATE.sensors:
        sensor.update()

    return STATE

def get_action_manual(state):
    """
    Reads pygame events and returns an action string based on arrow keys or spacebar.
    Up: ACCELERATE, Down: DECELERATE, Left: STEER_LEFT, Right: STEER_RIGHT, Space: NOTHING
    """

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()


    # Holding down keys
    keys = pygame.key.get_pressed()

    # Priority: accelerate, decelerate, steer left, steer right, nothing
    if keys[pygame.K_RIGHT]:
        return ["ACCELERATE"]
    if keys[pygame.K_LEFT]:
        return ["DECELERATE"]
    if keys[pygame.K_UP]:
        return ["STEER_LEFT"]
    if keys[pygame.K_DOWN]:
        return ["STEER_RIGHT"]
    if keys[pygame.K_SPACE]:
        return ["NOTHING"]

    # Just clicking once and it keeps doing it until a new press
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT:
                return ["ACCELERATE"]
            elif event.key == pygame.K_LEFT:
                return ["DECELERATE"]
            elif event.key == pygame.K_UP:
                return ["STEER_LEFT"]
            elif event.key == pygame.K_DOWN:
                return ["STEER_RIGHT"]
            elif event.key == pygame.K_SPACE:
                return ["NOTHING"]
    print(f"Velocity is {state.ego.velocity.x}, {state.ego.velocity.y}")

    # If no relevant key is pressed, repeat last action or do nothing
    #return STATE.latest_action if hasattr(STATE, "latest_action") else "NOTHING"
    return "NOTHING"

def get_action_rule_based(state):
    """
    An improved rule-based agent that adjusts its behavior based on its current speed.
    1. It uses DYNAMIC safety zones: the faster it goes, the more cautious it becomes.
    2. It has a TARGET cruising speed to avoid accelerating uncontrollably.
    3. It includes a corrective "nudge" to avoid side-swipes.
    """
    # =================================================================================
    # --- TUNABLE PARAMETERS ---
    # You can adjust these values to change the car's "personality".
    #
    # The desired speed when the road is clear.
    TARGET_CRUISING_SPEED = 12
    # How much the safety zones should increase per unit of speed.
    # Higher value = more cautious at high speeds.
    SPEED_SENSITIVITY_FACTOR = 18
    # The base distance for making a lane change.
    BASE_WARNING_ZONE = 200
    # The base distance for slamming the brakes.
    BASE_DANGER_ZONE = 100
    # The minimum side clearance needed to consider a lane change safe.
    SIDE_CLEARANCE = 300
    # **NEW**: The distance at which a car on the side is considered an immediate threat.
    SIDE_DANGER_ZONE = 400
    # =================================================================================

    # Define Sensor Groups (as before)
    SENSOR_GROUPS = {
        'front': [
            'front'
        ],
        'right': [
            'right_side_front',
            'right_side_back',
            'right_front',
            'right_side',
            'right_back',
            'front_right_front'
        ],
        'left': [
            'left_back',
            'left_side',
            'left_front',
            'left_side_front',
            'left_side_back',
            'front_left_front'
        ]
    }
    sensor_readings = {sensor.name: sensor.reading for sensor in state.sensors}
    DEFAULT_CLEAR_DISTANCE = 1000

    # Aggregate Sensor Readings (as before)
    grouped_distances = {}
    for group_name, sensor_list in SENSOR_GROUPS.items():
        min_distance = DEFAULT_CLEAR_DISTANCE
        for sensor_name in sensor_list:
            reading = sensor_readings.get(sensor_name)
            if reading is not None and reading < min_distance:
                min_distance = reading
        grouped_distances[group_name] = min_distance

    front_dist = grouped_distances.get('front', DEFAULT_CLEAR_DISTANCE)
    left_dist = grouped_distances.get('left', DEFAULT_CLEAR_DISTANCE)
    right_dist = grouped_distances.get('right', DEFAULT_CLEAR_DISTANCE)

    # Get Current State Information ---
    current_speed = state.ego.velocity.x

    # The warning and danger zones now grow based on the car's speed.
    # This makes the car look further ahead when it's moving faster.
    dynamic_warning_zone = BASE_WARNING_ZONE + (current_speed * SPEED_SENSITIVITY_FACTOR)
    dynamic_danger_zone = BASE_DANGER_ZONE + (
                current_speed * SPEED_SENSITIVITY_FACTOR / 2)  # Danger zone grows less aggressively


    # LOWEST): CLEAR ROAD (SPEED MANAGEMENT)

    if current_speed < TARGET_CRUISING_SPEED:
        action = "NOTHING"
    else:
        action = "NOTHING"

    # --- PRIORITY 3: POTENTIAL DANGER AHEAD (PROACTIVE LANE CHANGE) ---
    # This overrides the default action if an object is in the warning zone.
    if front_dist < dynamic_warning_zone:
        if left_dist > right_dist and left_dist > SIDE_CLEARANCE:
            action = "STEER_LEFT"
        elif right_dist > SIDE_CLEARANCE:
            action = "STEER_RIGHT"
        else:
            # Both lanes are blocked, so we must slow down.
            action = "DECELERATE"

    # --- PRIORITY 2: IMMEDIATE SIDE DANGER (CORRECTIVE NUDGE) ---
    # **NEW LOGIC**: This overrides any previous decision if a car is too close on the side.
    if left_dist < SIDE_DANGER_ZONE:
        action = "STEER_RIGHT"
    elif right_dist < SIDE_DANGER_ZONE:
        action = "STEER_LEFT"

    # --- PRIORITY 1 (HIGHEST): IMMEDIATE FRONT DANGER (EMERGENCY BRAKE) ---
    # This check has the final say and will override ALL other decisions.
    if front_dist < dynamic_danger_zone:
        action = "DECELERATE"

    # For debugging: print what the agent is "thinking"
    debug_msg = (
        f"Speed: {current_speed:.1f} | "
        f"Dist(F/L/R): {int(front_dist)}/{int(left_dist)}/{int(right_dist)} | "
        f"Zones(W/D): {int(dynamic_warning_zone)}/{int(dynamic_danger_zone)} | "
        f"ACTION: {action}"
    )
    print(debug_msg)

    return [action]


def get_action_Q_learning(state):
    # First, get the current simplified state from our helper function.
    current_state = get_simplified_state(state)

    # --- NEW Q-LEARNING CODE: STEP 2 ---

    # Before choosing an action, we must ensure this state is in our Q-table.
    # If we've never encountered this state before, we initialize it with a Q-value of 0 for every possible action.
    if current_state not in q_table:
        q_table[current_state] = {action: 0 for action in env_actions}
        print(f"New state discovered! Initializing Q-values for: {current_state}")  # For debugging

    # This is the Epsilon-Greedy strategy.
    # We generate a random number between 0 and 1.
    # If it's less than Epsilon, we choose a random action (Explore).
    # Otherwise, we choose the best-known action from our Q-table (Exploit).
    if random_number() < EPSILON:
        # --- EXPLORE ---
        # Choose a completely random action from the list of available actions.
        action = random_choice(env_actions)
        print(f"State: {current_state} -> Exploring: Chose random action '{action}'")  # For debugging
    else:
        # --- EXPLOIT ---
        # Look up the Q-values for the current state.
        state_q_values = q_table[current_state]
        # Find the action that has the highest Q-value.
        # The `max()` function with a key is a clean way to find the key in a dictionary corresponding to the maximum value.
        action = max(state_q_values, key=state_q_values.get)
        print(
            f"State: {current_state} -> Exploiting: Chose best action '{action}' with Q-value: {state_q_values[action]:.2f}")  # For debugging

    # The game loop expects a list of actions, so we wrap our chosen action in a list.
    return [action]

# Main game loop
ACTION_LOG = []

# Main game loop
ACTION_LOG = []


def game_loop(verbose: bool = True, log_actions: bool = True, log_path: str = "actions_log.json"):
    global STATE, EPSILON
    clock = pygame.time.Clock()
    screen = None
    actions = []
    if verbose:
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Race Car Game")

    while True:
        delta = clock.tick(60)
        STATE.elapsed_game_time += delta
        STATE.ticks += 1

        # Observe the state BEFORE taking an action.
        old_state = get_simplified_state(STATE)
        # --- NEW CODE: Record distance before the step ---
        old_distance = STATE.distance

        if not actions:
            action_list = get_action_Q_learning(STATE)
            for act in action_list:
                actions.append(act)
        action = actions.pop()

        if log_actions:
            ACTION_LOG.append({"tick": STATE.ticks, "action": action})

        # --- Game Simulation ---
        handle_action(action)
        # The line below updates the total distance
        STATE.distance += STATE.ego.velocity.x
        update_cars()
        remove_passed_cars()
        place_car()

        for sensor in STATE.sensors:
            sensor.update()

        for car in STATE.cars:
            if car != STATE.ego and intersects(STATE.ego.rect, car.rect):
                STATE.crashed = True

        for wall in STATE.road.walls:
            if intersects(STATE.ego.rect, wall.rect):
                STATE.crashed = True
        # --- End of Simulation ---

        # Get the new state and calculate the reward.
        new_state = get_simplified_state(STATE)
        reward = 0
        if STATE.crashed:
            reward = -10
        else:
            # --- UPDATED REWARD LOGIC ---
            # Reward is the actual distance traveled in this single step.
            #distance_this_tick = STATE.distance - old_distance
            #reward = distance_this_tick

            #TEST: Change reward to just staying alive
            #TODO: Monitor whether this works
            reward = 0.1

        # Update the Q-table (this logic remains the same)
        if new_state not in q_table:
            q_table[new_state] = {act: 0 for act in env_actions}

        old_q_value = q_table[old_state][action]
        max_future_q = max(q_table[new_state].values())
        new_q_value = old_q_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_future_q - old_q_value)
        q_table[old_state][action] = new_q_value

        print(
            f"Tick: {STATE.ticks} | State: {old_state} | Action: {action} | Reward: {reward:.2f} | New State: {new_state}")

        # Game over check (remains the same)
        if STATE.crashed or STATE.ticks > MAX_TICKS or STATE.elapsed_game_time > MAX_MS:
            print(f"Game over: Crashed: {STATE.crashed}, Ticks: {STATE.ticks}, Distance: {STATE.distance}")
            print(f"Final Epsilon: {EPSILON}")
            break

        # --- NEW CODE: Update sensor colors based on proximity ---
        # This assumes your Sensor class has a 'color' attribute that its 'draw' method uses.
        CLOSE_THRESHOLD = 300  # Should match the value in get_simplified_state
        for sensor in STATE.sensors:
            if sensor.reading is not None and sensor.reading < CLOSE_THRESHOLD:
                sensor.color = (0, 255, 0)  # Green for "close"
            else:
                sensor.color = (255, 0, 0)  # Red for "far" or no reading

        # Rendering
        if verbose:
            screen.fill((0, 0, 0))
            screen.blit(STATE.road.surface, (0, 0))
            for wall in STATE.road.walls: wall.draw(screen)
            for car in STATE.cars:
                if car.sprite:
                    screen.blit(car.sprite, (car.x, car.y))
                    bounds = car.get_bounds()
                    color = (255, 0, 0) if car == STATE.ego else (0, 255, 0)
                    pygame.draw.rect(screen, color, bounds, width=2)
                else:
                    pygame.draw.rect(screen, (255, 255, 0) if car == STATE.ego else (0, 0, 255), car.rect)
            if STATE.sensors_enabled:
                for sensor in STATE.sensors: sensor.draw(screen) # The draw call now uses the updated color
            pygame.display.flip()


    # Epsilon decay (remains the same)
    if EPSILON > MIN_EPSILON:
        EPSILON *= EPSILON_DECAY

    return STATE


    # # Save actions to file after game ends
    # import os
    # if log_actions:
    #     log_dir = os.path.dirname(log_path)
    #     if log_dir and not os.path.exists(log_dir):
    #         os.makedirs(log_dir, exist_ok=True)
    #     with open(log_path, "w") as f:
    #         json.dump(ACTION_LOG, f, indent=2)

# Initialization - not used
def init(api_url: str):
    global STATE
    STATE = GameState(api_url)
    print(f"Game initialized with API URL: {api_url}")


# Entry point
if __name__ == "__main__":
    seed_value = "static_seed"
    pygame.init()
    initialize_game_state("http://example.com/api/predict", seed_value)  # Replace with actual API URL
    game_loop(verbose=True)  # Change to verbose=False for headless mode
    pygame.quit()