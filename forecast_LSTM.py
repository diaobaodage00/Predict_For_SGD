from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def load_series(csv_path):
    df = pd.read_csv(csv_path, dtype=str, header=0, names=["DATE", "SGD"])
    df = df[~df['DATE'].str.upper().eq('DATE')]
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    df['SGD'] = pd.to_numeric(df['SGD'], errors='coerce')
    df = df.dropna(subset=['DATE', 'SGD'])
    df = df.sort_values('DATE').drop_duplicates(subset=['DATE'], keep='last').set_index('DATE').sort_index()
    idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')
    s = df['SGD'].reindex(idx).ffill()
    s.index.name = 'DATE'
    return s


def create_dataset(series_values, look_back=20, target='delta', y_source=None):
    """
    Create sequences X and targets y.
    - series_values: numpy array shaped (N, 1) for model inputs (already scaled/standardized)
    - If target=='delta', y is the difference between next value and last value of the input window (in the same space as series_values).
    - If target=='abs', y is the absolute next value (same space as series_values).
    - If target=='pct', y is the relative change (next/last - 1) computed in the same space as series_values.
    - If target=='logret', y is log-return computed from y_source in ORIGINAL value space: log(S_{t+1})-log(S_t).
      y_source must be the original (unscaled) values with shape (N,1).
    """
    X, y = [], []
    for i in range(len(series_values) - look_back):
        window = series_values[i:(i + look_back), 0]
        X.append(window)
        if target == 'delta':
            nxt = series_values[i + look_back, 0]
            last = window[-1]
            y.append(nxt - last)
        elif target == 'pct':
            nxt = series_values[i + look_back, 0]
            last = window[-1]
            if last == 0:
                y.append(0.0)
            else:
                y.append((nxt / last) - 1.0)
        elif target == 'logret':
            if y_source is None:
                raise ValueError("y_source (original values) is required when target='logret'")
            s_t = float(y_source[i + look_back - 1, 0])
            s_tp1 = float(y_source[i + look_back, 0])
            # numerical safety: avoid non-positive due to any data issue
            s_t = max(s_t, 1e-12)
            s_tp1 = max(s_tp1, 1e-12)
            y.append(np.log(s_tp1) - np.log(s_t))
        else:
            y.append(series_values[i + look_back, 0])
    X = np.array(X).reshape(-1, look_back, 1).astype(np.float32)
    y = np.array(y).astype(np.float32).reshape(-1, 1)
    return X, y


class LSTMForecast(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.act = nn.Tanh()  # bounded output for stability

    def forward(self, x):
        out, _ = self.lstm(x)  # (batch, seq_len, hidden)
        out = out[:, -1, :]    # last step
        out = self.fc(out)
        out = self.act(out)    # keep predictions in [-1, 1]
        return out


def forecast_future(model, x_scaler, last_window, steps, device, clip=True, target='delta', delta_clip=None):
    """
    Recursive multi-step forecast.
    - last_window: 1D numpy array of scaled/standardized values (length == look_back)
    - If target=='logret', model outputs log-returns in ORIGINAL value space; we map to next value as next = last * exp(p).
    - For target=='pct', model outputs fractional changes in scaled space.
    - delta_clip: tuple(min, max) to clip predicted changes; for 'logret' interpreted as absolute log-return bounds.
    - Returns: inverse-transformed predictions in ORIGINAL value space as 1D numpy array.
    """
    model.eval()
    window = last_window.copy().astype(np.float32)
    preds_values = []  # store in original value space
    with torch.no_grad():
        for _ in range(steps):
            inp = torch.from_numpy(window.reshape(1, -1, 1)).to(device)
            out = model(inp)
            p = float(out.cpu().numpy().reshape(-1)[0])
            if target == 'logret':
                # clip log-return
                if delta_clip is not None:
                    p = float(np.clip(p, float(delta_clip[0]), float(delta_clip[1])))
                # last original value
                last_scaled = window[-1]
                last_val = float(x_scaler.inverse_transform(np.array([[last_scaled]], dtype=np.float32)).flatten()[0])
                last_val = max(last_val, 1e-12)
                next_val = last_val * float(np.exp(p))
                preds_values.append(next_val)
                # transform next_val back to scaled space for window update
                next_scaled = float(x_scaler.transform(np.array([[next_val]], dtype=np.float32)).flatten()[0])
                if clip:
                    # prevent extreme standardized values from exploding window
                    next_scaled = float(np.clip(next_scaled, -10.0, 10.0))
                window = np.roll(window, -1)
                window[-1] = next_scaled
            elif target == 'pct':
                if delta_clip is not None:
                    p = float(np.clip(p, float(delta_clip[0]), float(delta_clip[1])))
                last_scaled = window[-1]
                last_val = float(x_scaler.inverse_transform(np.array([[last_scaled]], dtype=np.float32)).flatten()[0])
                next_val = last_val * (1.0 + p)
                preds_values.append(next_val)
                next_scaled = float(x_scaler.transform(np.array([[next_val]], dtype=np.float32)).flatten()[0])
                if clip:
                    next_scaled = float(np.clip(next_scaled, -10.0, 10.0))
                window = np.roll(window, -1)
                window[-1] = next_scaled
            else:
                # absolute value in scaled space; map back
                next_scaled = p
                if clip:
                    next_scaled = float(np.clip(next_scaled, -10.0, 10.0))
                next_val = float(x_scaler.inverse_transform(np.array([[next_scaled]], dtype=np.float32)).flatten()[0])
                preds_values.append(next_val)
                window = np.roll(window, -1)
                window[-1] = next_scaled
    return np.array(preds_values, dtype=np.float32).flatten()


def main(steps=22, look_back=20, epochs=30, batch_size=128, hidden_size=64, lr=1e-3):
    base = Path(__file__).parent
    csv_path = base / "data_subset.csv"
    s = load_series(csv_path)
    values_raw = s.values.reshape(-1, 1).astype(np.float32)

    # Quick sanity check
    print(f"Raw data range: min={float(np.nanmin(values_raw))}, max={float(np.nanmax(values_raw))}, len={len(values_raw)}")

    # Clip extreme outliers using 1st/99th percentiles for input scaling stability
    low_p, high_p = np.percentile(values_raw, [1, 99])
    clipped = values_raw.copy()
    low_count = int(np.sum(clipped < low_p))
    high_count = int(np.sum(clipped > high_p))
    if low_count + high_count > 0:
        clipped = np.clip(clipped, low_p, high_p)
        print(f"Clipped {low_count} below 1st pct and {high_count} above 99th pct: new range {clipped.min():.6f}..{clipped.max():.6f}")

    # Use StandardScaler for inputs (more stable than MinMax in recursive forecasting)
    x_scaler = StandardScaler()
    series_x = x_scaler.fit_transform(clipped).astype(np.float32)
    print(f"Standardized X: mean={float(series_x.mean()):.4f}, std={float(series_x.std()):.4f}")

    if len(series_x) <= look_back:
        raise ValueError("数据太少，无法建立序列。")

    # Build dataset: inputs from standardized price, targets are log-returns in ORIGINAL space
    X, y = create_dataset(series_x, look_back=look_back, target='logret', y_source=values_raw)

    # time-based split
    split = int(len(X) * 0.9)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    # derive robust per-step log-return clip from training targets
    r_low, r_high = np.percentile(y_train, [1, 99])
    # cap bound to ±0.5% in log-return (~±0.5% change) to prevent compounding drift
    fixed_cap = 0.005
    per_step_bound = min(max(abs(float(r_low)), abs(float(r_high))), fixed_cap)
    if per_step_bound <= 0:
        per_step_bound = fixed_cap
    print(f"Log-return clip bound: +/-{per_step_bound:.6g}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Model + optimization
    model = LSTMForecast(input_size=1, hidden_size=hidden_size, num_layers=3, dropout=0.2).to(device)
    criterion = nn.HuberLoss(delta=0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val = float('inf')
    best_state = None
    patience = 20
    no_improve = 0
    best_val = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            # gradient clip for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= max(1, len(val_loader.dataset))

        # early stopping tracking
        if val_loss + 1e-12 < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % max(1, epochs // 5) == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch}/{epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (best val_loss={best_val:.6f})")
            break

    # load best state if available
    if best_state is not None:
        model.load_state_dict(best_state)

    last_window = series_x[-look_back:, 0]
    print("Last standardized window (last 5 values):", last_window[-5:])

    preds = forecast_future(
        model,
        x_scaler,
        last_window,
        steps,
        device,
        clip=True,
        target='logret',
        delta_clip=(-per_step_bound, per_step_bound),
    )

    print(f"Predictions range (original space): min={float(preds.min())}, max={float(preds.max())}")
    print(f"Last history value: {float(values_raw[-1, 0])}")

    future_idx = pd.bdate_range(start=s.index[-1] + pd.Timedelta(days=1), periods=steps)
    df_forecast = pd.DataFrame({'DATE': future_idx, 'forecast_SGD': preds})
    df_forecast = df_forecast.set_index('DATE')
    out_csv = base / "lstm_forecast.csv"
    df_forecast.to_csv(out_csv, float_format='%.6f')

    # Plot last history + forecast
    plt.figure(figsize=(10, 5))
    history_span = 120 if len(s) > 120 else len(s)
    plt.plot(s.index[-history_span:], s.values[-history_span:], label='history')
    plt.plot(df_forecast.index, df_forecast['forecast_SGD'], label='forecast')
    plt.title(f'LSTM (PyTorch) forecast next {steps} business days')
    plt.legend()
    out_png = base / "lstm_forecast.png"
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

    print("输出文件：", out_csv, out_png)
    print(df_forecast.head(steps))


if __name__ == "__main__":
    # 可在此修改参数：steps(预测工作日数)、look_back、epochs
    # Safer defaults
    main(steps=22, look_back=120, epochs=100, batch_size=64, hidden_size=32, lr=10e-6)
