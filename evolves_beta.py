# Requiere: numpy, torch, matplotlib, sklearn
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ---------------------------
# 1) Construir dataset ventana multi-horizon y multi-feature
# ---------------------------
def build_windowed_dataset(series, window_size=64, horizon=5):
    """
    series: np.array shape (n_features, T) OR (T,) -> convert to (T, n_features)
    returns X: (N_samples, window_size, n_features)
            Y: (N_samples, horizon) -> next 'horizon' targets (we use close)
    """
    arr = np.asarray(series)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    # arr shape (T, n_features)
    if arr.shape[0] < window_size + horizon:
        raise ValueError("Serie demasiado corta")
    T, n_feat = arr.shape
    X = []
    Y = []
    for i in range(T - window_size - horizon + 1):
        x = arr[i:i+window_size, :]
        y = arr[i+window_size:i+window_size+horizon, 0]  # target: feature 0 (close)
        X.append(x)
        Y.append(y)
    X = np.stack(X).astype(np.float32)  # (N, window, n_feat)
    Y = np.stack(Y).astype(np.float32)  # (N, horizon)
    return X, Y

# ---------------------------
# 2) Modelo probabilístico simple (CNN/MLP para windows)
# ---------------------------
class ProbNet(nn.Module):
    def __init__(self, n_features, hidden=64, horizon=5):
        super().__init__()
        self.horizon = horizon
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_features * 64, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # salida: para cada horizon predict mean y log_var
        self.out_mean = nn.Linear(hidden, horizon)
        self.out_logvar = nn.Linear(hidden, horizon)

    def forward(self, x):
        # x: (B, window, n_features)
        h = self.net(x)
        mu = self.out_mean(h)
        logvar = self.out_logvar(h)
        # opcional: clamp logvar para estabilidad
        logvar = torch.clamp(logvar, -10.0, 5.0)
        return mu, logvar

# ---------------------------
# 3) Loss: Gaussian NLL
# ---------------------------
def gaussian_nll_loss(mu, logvar, target):
    # target: (B, horizon)
    var = torch.exp(logvar)
    mse_term = (target - mu) ** 2 / var
    loss = 0.5 * (logvar + mse_term).mean()
    return loss

# ---------------------------
# 4) Entrenamiento + evaluación
# ---------------------------
def train_model(model, train_loader, val_loader, lr=1e-3, epochs=10, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            mu, logvar = model(xb)
            loss = gaussian_nll_loss(mu, logvar, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            train_losses.append(loss.item())
        # val
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                mu, logvar = model(xb)
                loss = gaussian_nll_loss(mu, logvar, yb)
                val_losses.append(loss.item())
        print(f"Epoch {ep+1}/{epochs} train={np.mean(train_losses):.6f} val={np.mean(val_losses):.6f}")
    return model

# ---------------------------
# 5) Predicción + plotting
# ---------------------------
def predict_and_plot(model, X_test, y_test, idx=0, device=None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model.to(device).eval()
    xb = torch.from_numpy(X_test).float().to(device)
    with torch.no_grad():
        mu, logvar = model(xb)
    mu = mu.cpu().numpy()
    sigma = np.sqrt(np.exp(logvar.cpu().numpy()))
    # elegir muestra idx
    y_true = y_test[idx]
    y_pred = mu[idx]
    y_sigma = sigma[idx]
    horizon = y_true.shape[0]

    # plot
    plt.figure(figsize=(8,3))
    xs = np.arange(horizon)
    plt.plot(xs, y_true, label="Real", marker='o')
    plt.plot(xs, y_pred, label="Pred mean", marker='o')
    plt.fill_between(xs, y_pred - 1.96*y_sigma, y_pred + 1.96*y_sigma, alpha=0.3, label="95% CI")
    plt.legend()
    plt.title("Predicción multi-horizon con intervalo")
    plt.xlabel("Steps ahead")
    plt.show()

    # métricas simples
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"MSE={mse:.6f}, MAE={mae:.6f}")

# ---------------------------
# 6) Ejemplo de uso (suponer que data contiene close series)
# ---------------------------
if __name__ == "__main__":
    # serie de ejemplo: usa tus datos (normalizar antes)
    # data: np.array shape (T,) or (T, n_features)
    T = 2000
    t = np.linspace(0, 50, T)
    series = np.sin(t) + 0.1*np.random.randn(T)
    # normalize / scale
    series = (series - series.mean()) / (series.std() + 1e-9)
    # convertir a formato (T, n_features)
    series = series.reshape(-1, 1)

    window = 64
    horizon = 8
    X, Y = build_windowed_dataset(series, window, horizon)
    # split
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    Y_train, Y_val = Y[:split], Y[split:]

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train)), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val)), batch_size=64, shuffle=False)

    model = ProbNet(n_features=1, hidden=128, horizon=horizon)
    model = train_model(model, train_loader, val_loader, lr=1e-3, epochs=8)

    predict_and_plot(model, X_val, Y_val, idx=5)
