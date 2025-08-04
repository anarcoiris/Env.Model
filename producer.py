import os
import json
import time
from datetime import datetime
from kafka import KafkaProducer
import ccxt
from influxdb_client import InfluxDBClient, Point, WritePrecision

# Configuración desde variables de entorno
BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
TOPIC = os.getenv('KAFKA_TOPIC', 'prices')
EXCHANGES = os.getenv('EXCHANGES', 'binance').split(',')
SYMBOLS = os.getenv('SYMBOLS', 'BTC/USDT').split(',')
FETCH_INTERVAL = int(os.getenv('FETCH_INTERVAL', 86400))  # segundos

# Inicializar productor Kafka
producer = KafkaProducer(
    bootstrap_servers=BOOTSTRAP_SERVERS,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Crear instancias CCXT para cada exchange
exchange_instances = {}
for name in EXCHANGES:
    cls = getattr(ccxt, name.strip())
    exchange_instances[name.strip()] = cls({'enableRateLimit': True})


def fetch_and_publish():
    timestamp = datetime.utcnow().isoformat()
    for exch_name, exch in exchange_instances.items():
        for symbol in SYMBOLS:
            try:
                # Obtener ticker OHLCV más reciente
                ohlcv = exch.fetch_ohlcv(symbol, timeframe='1d', limit=1)
                if not ohlcv:
                    continue
                ts_ms, open_, high, low, close, *_ = ohlcv[0]
                msg = {
                    'exchange': exch_name,
                    'symbol': symbol.replace('/', ''),
                    'ts': datetime.utcfromtimestamp(ts_ms/1000).isoformat() + 'Z',
                    'open': open_,
                    'high': high,
                    'low': low,
                    'close': close
                }
                producer.send(TOPIC, msg)
                print(f"Sent: {msg}")
            except Exception as e:
                print(f"Error fetching {symbol} from {exch_name}: {e}")
    producer.flush()


# Productor en bucle hasta interrupción
def produce_loop():
    try:
        while True:
            fetch_and_publish()
            time.sleep(FETCH_INTERVAL)
    except KeyboardInterrupt:
        print("\nProducer stopped, starting consumer to write to InfluxDB...")

# Consumidor que escribe en InfluxDB
def consume_and_write():
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )
    with InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG) as client:
        write_api = client.write_api()
        for msg in consumer:
            data = msg.value
            point = (
                Point("prices")
                .tag("exchange", data['exchange'])
                .tag("symbol", data['symbol'])
                .field("open", float(data['open']))
                .field("high", float(data['high']))
                .field("low", float(data['low']))
                .field("close", float(data['close']))
                .time(data['ts'], WritePrecision.NS)
            )
            write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)
            print(f"Written to InfluxDB: {data}")

# Punto de entrada
if __name__ == '__main__':
    produce_loop()
    consume_and_write()
