import requests
import json
import os
import argparse
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from deap import base, creator, tools, algorithms
from tqdm import trange
from influxdb_client import InfluxDBClient

# ------------------------------
# 1) COMUNICACIÓN CON EL BRAZO
# ------------------------------
ARM_API_URL = os.getenv('ARM_API_URL')
ARM_API_KEY = os.getenv('ARM_API_KEY')

def send_order_to_arm(symbol, type, side, amount, price=None):
    """ Envía una orden a la API de El Brazo. """
    endpoint = f"{ARM_API_URL}/order"
    headers = {
        "X-API-KEY": ARM_API_KEY,
        "Content-Type": "application/json"
    }
    order_data = {
        "symbol": symbol, "type": type, "side": side,
        "amount": amount, "price": price
    }
    payload = {k: v for k, v in order_data.items() if v is not None}
    
    try:
        print(f" Enviando orden a El Brazo: {payload}")
        response = requests.post(endpoint, headers=headers, data=json.dumps(payload), timeout=10)
        
        if response.status_code == 200:
            print(f" Orden enviada y aceptada por El Brazo: {response.json()}")
            return response.json()
        else:
            print(f" Error al enviar orden a El Brazo: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f" Excepción de conexión al contactar a El Brazo: {e}")
        return None

# ------------------------------
# 2) LECTURA DE DATOS (La Memoria)
# ------------------------------
def load_data_from_influx(symbol, exchange, timeframe, start, limit):
    INFLUX_URL = os.getenv('INFLUXDB_URL')
    INFLUX_TOKEN = os.getenv('DOCKER_INFLUXDB_INIT_ADMIN_TOKEN')
    INFLUX_ORG = os.getenv('DOCKER_INFLUXDB_INIT_ORG')
    INFLUX_BUCKET = os.getenv('DOCKER_INFLUXDB_INIT_BUCKET')

    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: {start})
      |> filter(fn: (r) => r["_measurement"] == "prices")
      |> filter(fn: (r) => r["symbol"] == "{symbol}")
      |> filter(fn: (r) => r["exchange"] == "{exchange}")
      |> filter(fn: (r) => r["timeframe"] == "{timeframe}")
      |> filter(fn: (r) => r["_field"] == "close")
      |> sort(columns: ["_time"], desc: false)
      |> limit(n: {limit})
    '''

    with InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG) as client:
        tables = client.query_api().query(query)
        values = [float(record.get_value()) for table in tables for record in table.records]

    if not values:
        raise ValueError(f"No se obtuvieron datos de InfluxDB para {symbol}/{exchange}/{timeframe}")

    return np.array(values, dtype=np.float32).reshape(1, -1)

# ------------------------------
# 3) PREPARACIÓN DEL DATASET
# ------------------------------
def prepare_dataloaders(data, train_frac=0.8, batch_size=256):
    X = data[:, :-1].reshape(-1, 1)
    y = data[:, 1:].reshape(-1, 1)
    split = int(train_frac * len(X))
    X_train, y_train, X_val, y_val = X[:split], y[:split], X[split:], y[split:]
    
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), DataLoader(val_ds, batch_size=batch_size)

# ------------------------------
# 4) RED NEURONAL Y ENTRENAMIENTO
# ------------------------------
class Net(nn.Module):
    def __init__(self, n_hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, int(n_hidden)),
            nn.ReLU(),
            nn.Linear(int(n_hidden), 1)
        )
    def forward(self, x): return self.net(x)

def train_and_evaluate(n_hidden, lr, train_loader, val_loader, epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net(max(1, int(n_hidden))).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss_fn(model(xb), yb).backward()
            optimizer.step()

    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            total_loss += loss_fn(model(xb), yb).item() * xb.size(0)
    return total_loss / len(val_loader.dataset)

# ------------------------------
# 5) CONFIGURACIÓN DE DEAP
# ------------------------------
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("n_hidden", random.randint, 5, 100)
toolbox.register("log_lr", lambda: random.uniform(-4, -1))
toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.n_hidden, toolbox.log_lr), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def repair(individual):
    individual[0] = max(5, min(200, int(round(individual[0]))))
    individual[1] = max(-4.0, min(-1.0, individual[1]))
    return individual

def mutate_and_repair(individual):
    tools.mutGaussian(individual, mu=[50, -2.5], sigma=[20, 1.0], indpb=0.2)
    return repair(individual),

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", mutate_and_repair)
toolbox.register("select", tools.selTournament, tournsize=3)

def eval_individual(ind, train_loader, val_loader):
    n_hidden, log_lr = ind
    return (train_and_evaluate(n_hidden, 10**log_lr, train_loader, val_loader),)

# ------------------------------
# 6) BUCLE EVOLUTIVO PRINCIPAL
# ------------------------------
def main_evolution(pop_size, gens, train_loader, val_loader):
    toolbox.register("evaluate", eval_individual, train_loader=train_loader, val_loader=val_loader)
    
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=gens, stats=stats, halloffame=hof, verbose=True)
    return hof[0]

# ------------------------------
# 7) EJECUCIÓN DEL SCRIPT
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cerebro: Evolución de Hiperparámetros y Ejecución de Órdenes")
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--exchange", type=str, default="binance")
    parser.add_argument("--timeframe", type=str, default="1m")
    parser.add_argument("--start", type=str, default="-7d")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--pop_size", type=int, default=10)
    parser.add_argument("--gens", type=int, default=5)
    args = parser.parse_args()

    print("="*50)
    print(" Cerebro Iniciado: Cargando datos de la memoria...")
    print("="*50)
    
    try:
        data = load_data_from_influx(
            symbol=args.symbol, exchange=args.exchange, timeframe=args.timeframe,
            start=args.start, limit=args.limit
        )
        print(f" Datos cargados: {data.shape[1]} puntos de '{args.symbol}'")

        train_loader, val_loader = prepare_dataloaders(data)

        print("\n" + "="*50)
        print(" Iniciando evolución para encontrar el mejor modelo...")
        print("="*50)
        
        best_individual = main_evolution(args.pop_size, args.gens, train_loader, val_loader)
        best_n_hidden, best_log_lr = best_individual
        
        print("\n" + "="*50)
        print(" Evolución completada.")
        print(f" Mejor individuo: n_hidden={int(best_n_hidden)}, lr=10^({best_log_lr:.2f})")
        print(f" MSE: {best_individual.fitness.values[0]:.6f}")
        print("="*50)
        
        print("\n" + "="*50)
        print(" Tomando una decisión de trading basada en el último precio...")
        print("="*50)

        # Lógica de ejemplo simple: predecir el siguiente precio y actuar
        last_price = data[0, -1]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Net(best_n_hidden).to(device) # Re-crear el mejor modelo (una implementación real lo guardaría)
        
        # En un caso real, entrenarías el modelo final con los mejores hiperparámetros
        # Aquí solo hacemos una predicción simbólica
        predicted_price = last_price * (1 + (random.random() - 0.5) * 0.01) # Simulación

        print(f"Precio actual: {last_price:.2f}")
        print(f"Predicción de precio: {predicted_price:.2f}")

        # Decisión simple: si predecimos que sube, compramos; si baja, vendemos
        if predicted_price > last_price:
            print("Acción: COMPRAR")
            send_order_to_arm(symbol="BTC/USDT", type="market", side="buy", amount=0.001)
        else:
            print("Acción: VENDER")
            send_order_to_arm(symbol="BTC/USDT", type="market", side="sell", amount=0.001)

    except Exception as e:
        print(f" Error catastrófico en el Cerebro: {e}")