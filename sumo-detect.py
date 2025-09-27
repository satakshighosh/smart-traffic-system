import traci
import time
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from collections import deque

# --- Define the LSTM model class and load the pre-trained weights
# This class is identical to the one you provided in your example.
class ForecastNet(nn.Module):
    def __init__(self, window_size, hidden_size=64):
        super(ForecastNet, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: batch_size x window_size
        x = x.unsqueeze(-1)  # batch_size x window_size x 1
        out, (hn, cn) = self.lstm(x)
        h_last = hn[-1]
        pred = self.fc(h_last)
        # ensure non-negative
        return torch.relu(pred)

# --- Initialize the model and load the trained state
WINDOW_SIZE = 24  # This must match the window_size used for training the model
try:
    forecast_model = ForecastNet(window_size=WINDOW_SIZE)
    # The map_location="cpu" ensures it runs on any machine, even without a GPU.
    forecast_model.load_state_dict(torch.load("forecast_model.pth", map_location="cpu"))
    forecast_model.eval()
    print("LSTM model loaded successfully.")
except FileNotFoundError:
    print("Warning: forecast_model.pth not found. Using current vehicle counts instead of forecast.")
    forecast_model = None

# --- Traffic simulation setup
sumoBinary = "sumo-gui"  # change to "sumo" for headless
sumoConfig = "sumo_sim.sumocfg"

# --- Traffic light parameters
base_green = 10         # minimum green seconds
extra_per_vehicle = 1   # extra seconds per vehicle
min_green = 5           # minimum green to prevent too short
max_green = 25          # cap maximum green time

current_green_phase = 1
green_timer = time.time()

# Define lane groups at the junction
lane_group_1 = ["E5_1", "E4_1"]
lane_group_2 = ["E6_1", "E1_1"]
lane_ids = lane_group_1 + lane_group_2
junction_id = "J4"

# --- Data storage for LSTM input
# We use deque to store a rolling window of past vehicle counts,
# which serves as the input sequence for our LSTM model.
history_group_1 = deque(maxlen=WINDOW_SIZE)
history_group_2 = deque(maxlen=WINDOW_SIZE)

# --- Function to make a forecast
def forecast_next(history_data):
    """
    Makes a single-step forecast using the loaded LSTM model.
    This replaces the dummy 'forecast_next' function from your example.
    """
    if forecast_model is None or len(history_data) < WINDOW_SIZE:
        # If model is not loaded or history is too short, return 0
        return 0

    # Convert the deque of historical counts to a PyTorch tensor
    # The shape must be (batch_size, sequence_length), which is (1, WINDOW_SIZE)
    input_tensor = torch.tensor(list(history_data), dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        # Pass the data through the LSTM model to get the prediction
        predicted_count = forecast_model(input_tensor).item()
    
    return predicted_count

# --- Initialize queue data and live plot
queue_data = {lane: [] for lane in lane_ids}
plt.ion()  # interactive mode for live plotting
plot_interval = 10  # update plot every N simulation steps
step = 0

# --- Start SUMO simulation
try:
    traci.start([sumoBinary, "-c", sumoConfig])
    
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        step += 1
    
        # 1. Vehicle Counting (simulating YOLO)
        # Use SUMO's built-in functions to count vehicles in each lane group,
        # which acts as our real-time input data.
        current_count_1 = sum([traci.lane.getLastStepVehicleNumber(l) for l in lane_group_1])
        current_count_2 = sum([traci.lane.getLastStepVehicleNumber(l) for l in lane_group_2])
    
        # 2. Update historical data buffer
        history_group_1.append(current_count_1)
        history_group_2.append(current_count_2)
    
        # 3. Forecast future traffic and calculate dynamic green times
        # The system now uses the LSTM to make a PROACTIVE decision.
        if len(history_group_1) == WINDOW_SIZE:
            predicted_count_1 = forecast_next(history_group_1)
            predicted_count_2 = forecast_next(history_group_2)
            
            # We use the *predicted* counts to calculate the green time
            # This allows the system to prepare for future traffic surges.
            green1 = min(max(base_green + predicted_count_1 * extra_per_vehicle, min_green), max_green)
            green2 = min(max(base_green + predicted_count_2 * extra_per_vehicle, min_green), max_green)
            print(f"Step {step} | Forecasted counts -> Lane 1: {predicted_count_1:.2f}, Lane 2: {predicted_count_2:.2f}")
        else:
            # Fallback to current counts until enough historical data is collected
            green1 = min(max(base_green + current_count_1 * extra_per_vehicle, min_green), max_green)
            green2 = min(max(base_green + current_count_2 * extra_per_vehicle, min_green), max_green)
            print(f"Step {step} | Current counts -> Lane 1: {current_count_1}, Lane 2: {current_count_2}")
    
        # 4. Timer logic to switch traffic light phases
        elapsed = time.time() - green_timer
        lane_time = green1 if current_green_phase == 1 else green2
        time_left = max(int(lane_time - elapsed), 0)
    
        # Switch phase if time is up
        if time_left == 0:
            current_green_phase = 2 if current_green_phase == 1 else 1
            green_timer = time.time()
            lane_time = green1 if current_green_phase == 1 else green2
    
        # Set traffic light state
        if current_green_phase == 1:
            traci.trafficlight.setRedYellowGreenState(junction_id, "GGGrrr")
        else:
            traci.trafficlight.setRedYellowGreenState(junction_id, "rrrGGG")
    
        # 5. Record queue data for plotting
        for lane in lane_ids:
            queue_data[lane].append(traci.lane.getLastStepVehicleNumber(lane))
    
        # Live plotting every 'plot_interval' steps
        if step % plot_interval == 0:
            plt.clf()
            for lane in lane_ids:
                plt.plot(queue_data[lane], label=lane)
            plt.xlabel("Simulation Step")
            plt.ylabel("Number of Vehicles")
            plt.title("LSTM-Powered Dynamic Signal Timing - Live Queue")
            plt.legend()
            plt.pause(0.01)
    
except traci.exceptions.FatalTraCIError as e:
    print(f"Error: SUMO simulation failed. Please ensure 'sumo_sim.sumocfg' is in the correct path. Details: {e}")
finally:
    traci.close()
    plt.ioff()
    plt.show()
