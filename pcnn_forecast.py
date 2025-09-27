import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# -- Dataset class for time series forecasting
class TrafficTS(Dataset):
    def __init__(self, csv_path, window_size=24, horizon=1):
        df = pd.read_csv(csv_path)
        
        df = df.sort_values('DateTime')
        counts = df['Vehicles'].values.astype(np.float32)
        self.window = window_size
        self.horizon = horizon
        self.X = []
        self.y = []
        for i in range(len(counts) - window_size - horizon + 1):
            self.X.append(counts[i : i + window_size])
            self.y.append(counts[i + window_size + horizon - 1])
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -- Model definition
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

# -- Training code
def train(csv_path, window_size=24, horizon=1, lr=0.001, epochs=20, batch_size=32):
    dataset = TrafficTS(csv_path, window_size, horizon)
    n = len(dataset)
    train_len = int(0.8 * n)
    val_len = n - train_len
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = ForecastNet(window_size=window_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_train = total_loss / train_len

        model.eval()
        total_val = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                total_val += loss.item() * xb.size(0)
        avg_val = total_val / val_len

        print(f"Epoch {epoch+1}/{epochs} — Train Loss: {avg_train:.4f}, Val Loss: {avg_val:.4f}")

    torch.save(model.state_dict(), 'forecast_model.pth')
    print("Model saved as forecast_model.pth")

# -- If running this file
if __name__ == "__main__":
    train('traffic.csv', window_size=24, horizon=1, epochs=20)
