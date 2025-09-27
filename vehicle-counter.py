import ultralytics
import torch
import torch.nn as nn
import cv2
import time
from collections import deque

# --- Model definition
class ForecastNet(nn.Module):
    def __init__(self, window_size, hidden_size=64):
        super(ForecastNet, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)  # batch_size x window_size x 1
        out, (hn, cn) = self.lstm(x)
        h_last = hn[-1]
        pred = self.fc(h_last)
        return torch.relu(pred)  # ensure non-negative


yolo_model = ultralytics.YOLO("yolov8n.pt")  # nano model for faster processing

forecast_model = ForecastNet(window_size=24)
try:
    forecast_model.load_state_dict(torch.load("forecast_model.pth", map_location="cpu"))
    forecast_model.eval()
    print("PCNN model loaded successfully.")
except FileNotFoundError:
    print("Warning: forecast_model.pth not found. The model will use random data for demonstration.")
    forecast_model = None

cap1 = cv2.VideoCapture("highwaylane.mp4")   # or 0 for webcam
cap2 = cv2.VideoCapture("lane_light.mp4")    # or 1 for webcam

vehicle_classes = ["car", "truck", "bus", "motorbike"]

# Traffic light timing parameters
base_green = 10         # base green time (seconds)
extra_per_vehicle = 1   # add 1 sec per detected vehicle
min_green = 5           # minimum green to prevent too short
max_green = 25          # cap maximum green time

# Initialize timers
green_timer = None
current_green_lane = 1    # start with lane 1 green

# --- CHANGES: Initialize deque to store a rolling window of past vehicle counts
history_lane1 = deque(maxlen=24)
history_lane2 = deque(maxlen=24)

def count_vehicles(frame):
    """
    Detect vehicles, draw bounding boxes, and return count.
    """
    results = yolo_model(frame, verbose=False)
    count = 0
    for box in results[0].boxes:
        cls = int(box.cls[0])
        name = yolo_model.names[cls]
        if name in vehicle_classes:
            count += 1
            # Draw bounding box
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, name, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return count

# --- CHANGES: Update the forecast function to use real historical data
def forecast_next(history_data):
    """
    Makes a single-step forecast using the loaded PCNN model.
    """
    # Check if the model is loaded and we have enough data to make a forecast
    if forecast_model is None or len(history_data) < 24:
        return None # Return None if a forecast can't be made

    input_tensor = torch.tensor(list(history_data), dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        predicted_count = forecast_model(input_tensor).item()
        
    return predicted_count

# 🔹 ADDED: Create named windows and set fixed size
cv2.namedWindow("Lane 1", cv2.WINDOW_NORMAL)
cv2.namedWindow("Lane 2", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Lane 1", 960, 540)   # width x height
cv2.resizeWindow("Lane 2", 960, 540)   # adjust as needed

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not ret1 or not ret2:
        break

    # Vehicle detection & counting
    count1 = count_vehicles(frame1)
    count2 = count_vehicles(frame2)
    
    # --- CHANGES: Add current counts to historical deque
    history_lane1.append(count1)
    history_lane2.append(count2)

    # --- CHANGES: Call forecast function and use the result for green time
    predicted_count_1 = forecast_next(history_lane1)
    predicted_count_2 = forecast_next(history_lane2)

    # Initialize timer if first frame or lane switch
    if green_timer is None:
        green_timer = time.time()
        # Use forecasted value if available, otherwise use current count
        lane1_final_count = predicted_count_1 if predicted_count_1 is not None else count1
        lane2_final_count = predicted_count_2 if predicted_count_2 is not None else count2
        
        green1 = min(max(base_green + lane1_final_count * extra_per_vehicle, min_green), max_green)
        green2 = min(max(base_green + lane2_final_count * extra_per_vehicle, min_green), max_green)
        
        lane_time = green1 if current_green_lane == 1 else green2
        time_left = int(lane_time)
        
    # Remaining time for current green lane
    elapsed = time.time() - green_timer
    time_left = max(int(lane_time - elapsed), 0)

    # Switch lanes when time is up
    if time_left == 0:
        current_green_lane = 2 if current_green_lane == 1 else 1
        green_timer = time.time()

        # Recalculate green time for the new cycle
        lane1_final_count = predicted_count_1 if predicted_count_1 is not None else count1
        lane2_final_count = predicted_count_2 if predicted_count_2 is not None else count2
        
        green1 = min(max(base_green + lane1_final_count * extra_per_vehicle, min_green), max_green)
        green2 = min(max(base_green + lane2_final_count * extra_per_vehicle, min_green), max_green)
        
        lane_time = green1 if current_green_lane == 1 else green2
        time_left = int(lane_time)

    # Display counts, signals, and forecast
    cv2.putText(frame1, f"Lane 1: {count1} vehicles", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame2, f"Lane 2: {count2} vehicles", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show the predicted value on the screen
    forecast_display_1 = f"Forecast: {predicted_count_1:.2f}" if predicted_count_1 is not None else "Forecasting..."
    forecast_display_2 = f"Forecast: {predicted_count_2:.2f}" if predicted_count_2 is not None else "Forecasting..."

    cv2.putText(frame1, forecast_display_1, (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame2, forecast_display_2, (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    if current_green_lane == 1:
        cv2.putText(frame1, f"GREEN - {time_left}s", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        cv2.putText(frame2, f"RED - {time_left}s", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    else:
        cv2.putText(frame1, f"RED - {time_left}s", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.putText(frame2, f"GREEN - {time_left}s", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # Show results with fixed window size
    cv2.imshow("Lane 1", frame1)
    cv2.imshow("Lane 2", frame2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
