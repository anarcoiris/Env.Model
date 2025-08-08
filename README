# 🐳 Guía de instalación y ejecución en Windows

## 1️⃣ Requisitos previos

Instalar las dependencias necesarias:

```bash
# Abre el diálogo de ejecutar (Win + R) y escribe: cmd
winget install -e --id Docker.DockerDesktop
winget install -e --id Git.Git
wsl --install && wsl --update   # Comprueba que la virtualización (VT-d) está habilitada en la BIOS y reinicia
2️⃣ Clonar el repositorio y ejecutar el contenedor

mkdir C:\Entorno_de_Trabajo\
cd C:\Entorno_de_Trabajo
git clone https://github.com/anarcoiris/Env.Model/
cd Env*
docker compose up --build
💡 Si es necesario, reinicia manualmente los contenedores de Kafka, producer o consumer.

🧠 Explicación de la arquitectura
Este sistema está compuesto por cinco elementos principales: Los Ojos, La Voz, La Memoria, El Brazo y El Cerebro.

👀 Los Ojos (Kafka - Producer)
El módulo encargado de observar datos en internet y enviarlos al sistema.

Variables de entorno:

KAFKA_BOOTSTRAP_SERVERS=kafka:9092
KAFKA_TOPIC=prices
EXCHANGES=binance,kucoin,bybit,coinbase,kraken
SYMBOLS=BTC/USDT,XMR/USDT,ETH/USDT
FETCH_INTERVAL=10
TIMEFRAMES=1m,30m,4h,1d
🗣 La Voz (Kafka - Consumer)
Repite todo lo que Los Ojos ven, enviando los datos procesados a la base de datos.

Ejemplo de escritura:

point = (
    Point("prices")
        .tag("exchange", data['exchange'])
        .tag("symbol", data['symbol'])
        .tag("timeframe", data.get("timeframe", "unknown"))
        .field("open", float(data['open']))
        .field("high", float(data['high']))
        .field("low", float(data['low']))
        .field("close", float(data['close']))
        .field("volume", float(data['volume']))
        .time(ts, WritePrecision.NS)
)
💾 La Memoria (InfluxDB)
Base de datos donde La Voz almacena lo que Los Ojos captan.

Variables de entorno:

DOCKER_INFLUXDB_INIT_MODE=setup
DOCKER_INFLUXDB_INIT_USERNAME=root
DOCKER_INFLUXDB_INIT_PASSWORD=rootroot
DOCKER_INFLUXDB_INIT_ORG=BreOrganization
DOCKER_INFLUXDB_INIT_BUCKET=prices
DOCKER_INFLUXDB_INIT_ADMIN_TOKEN='J4twWiQSH6QZF33ZyAB9NJLoNMyrHjOlvY6UJGgczJfk-_DC3d5BFEiZzQOYC39ObPYwxF5kZTAZtzIX-Xr40Q=='
🤖 El Brazo (JaviBre)
Ejecuta órdenes en los mercados:

Comprar

Vender

Leer órdenes (desde la base de datos o API del Cerebro)

Reportar estado: órdenes ejecutadas, balances, estado del sistema

🧠 El Cerebro (Evolves2.py)
Utiliza La Memoria para entrenar modelos predictivos y dar órdenes a El Brazo.

📊 Diagrama de flujo

    A[👀 Los Ojos] -->|Datos de mercado| B[🗣 La Voz]
    B -->|Inserta datos| C[💾 La Memoria]
    C -->|Consulta datos| D[🧠 El Cerebro]
    D -->|Órdenes| E[🤖 El Brazo]
    E -->|Ejecuta en| F[(Mercados)]

############################### README OLD ################################################



Cómo lanzar el contenedor en windows:
  instalar dependencias: Git, Docker Desktop, repositorio
Tecla 'Win' + R :  'cmd' 
  winget install -e --id Docker.DockerDesktop
  winget install -e --id Git.Git
  wsl --install && wsl --update   # Comprobar que está habilitada la virtualizacion (vt-d?) en la BIOS, reiniciar.

Creamos un directorio:
mkdir C:\Entorno_de_Trabajo\
cd C:\Entorno_de_trabajo
git clone https://github.com/anarcoiris/Env.Model/
cd Env*
docker compose up --build

Comprobar si hace falta reiniciar los contenedores de kafka y/o producer y consumer





%%%%%%%%%%%%%%%%%%%%   EXPLICACION DEL CODIGO  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
vvvvvvvvvvvvvvvvvvv  ^^^^^^^^^^^^^^^^^^^^^^^^^^^   vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv


Los ojos: El programa encargado de ver todo lo que se mueve y sucede en algun lado de internet (Kafka - producer)
 Variables de entorno propias: timeframes, exchanges...:
      - KAFKA_BOOTSTRAP_SERVERS=kafka:9092
      - KAFKA_TOPIC=prices
      - EXCHANGES=binance,kucoin,bybit,coinbase,kraken
      - SYMBOLS=BTC/USDT,XMR/USDT,ETH/USDT
      - FETCH_INTERVAL=10
      - TIMEFRAMES=1m,30m,4h,1d

La Voz: El programa encargado de repetir todo lo que los ojos ven (Kafka - consumer)

 Habla de:
  point = (
                Point("prices")
                    .tag("exchange", data['exchange'])
                    .tag("symbol", data['symbol'])
                    .tag("timeframe", data.get("timeframe", "unknown"))
                    .field("open", float(data['open']))
                    .field("high", float(data['high']))
                    .field("low", float(data['low']))
                    .field("close", float(data['close']))
                    .field("volume", float(data['volume']))
                    .time(ts, WritePrecision.NS)
            )

La memoria: La base de datos sobre la que La Voz guarda lo que Los Ojos ven (InfluxDB)

   Variables de entorno propias:
      - DOCKER_INFLUXDB_INIT_MODE=setup
      - DOCKER_INFLUXDB_INIT_USERNAME=root
      - DOCKER_INFLUXDB_INIT_PASSWORD=rootroot
      - DOCKER_INFLUXDB_INIT_ORG=BreOrganization
      - DOCKER_INFLUXDB_INIT_BUCKET=prices
      - DOCKER_INFLUXDB_INIT_ADMIN_TOKEN='J4twWiQSH6QZF33ZyAB9NJLoNMyrHjOlvY6UJGgczJfk-_DC3d5BFEiZzQOYC39ObPYwxF5kZTAZtzIX-Xr40Q=='

El Brazo: El programa que sabe ejecutar ordenes en los mercados (JaviBre)
 Funciones: Comprar, Vender, Leer ordenes (probablemente tambien de la base de datos, si no de la API de El Cerebro), Reportar estado (ordenes ejecutadas correctamente, el balance... si esta vivo)

El Cerebro: El modelo que utiliza La Memoria para entrenar modelos predictivos y dar ordenes a El Brazo (Evolves2.py)
