import os
import json
import time
from datetime import datetime, timezone, timedelta
from kafka import KafkaProducer
import ccxt

BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
TOPIC = os.getenv('KAFKA_TOPIC', 'prices')
EXCHANGES = os.getenv('EXCHANGES', 'binance').split(',')
SYMBOLS = os.getenv('SYMBOLS', 'BTC/USDT').split(',')
TIMEFRAMES = os.getenv('TIMEFRAMES', '1m,30m,4h,1d').split(',')
FETCH_INTERVAL = int(os.getenv('FETCH_INTERVAL', 60))

producer = KafkaProducer(
    bootstrap_servers=BOOTSTRAP_SERVERS,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

exchange_instances = {
    name.strip(): getattr(ccxt, name.strip())({'enableRateLimit': True})
    for name in EXCHANGES
}

last_ts = {}

def fetch_historical_and_publish():
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    limit = 500
    for exch_name, exch in exchange_instances.items():
        for symbol in SYMBOLS:
            for tf in TIMEFRAMES:
                key = f"{exch_name}_{symbol.replace('/', '')}_{tf}"
                start_ts = last_ts.get(key, None)

                if start_ts is None:
                    start_ts = int((datetime.now(timezone.utc) - timedelta(days=30)).timestamp() * 1000)

                try:
                    ohlcv = exch.fetch_ohlcv(symbol, timeframe=tf, since=start_ts, limit=limit)
                    if not ohlcv:
                        print(f"No more historical data for {key}")
                        last_ts[key] = now_ms # Avoid re-fetching immediately
                        continue

                    for candle in ohlcv:
                        ts_ms, open_, high, low, close, volume = candle
                        if ts_ms >= now_ms:
                            break
                        msg = {
                            'exchange': exch_name,
                            'symbol': symbol.replace('/', ''),
                            'timeframe': tf,
                            'ts_candle': datetime.utcfromtimestamp(ts_ms / 1000).isoformat() + 'Z',
                            'ts_ingest': datetime.now(timezone.utc).isoformat(),
                            'open': open_,
                            'high': high,
                            'low': low,
                            'close': close,
                            'volume': volume
                        }
                        producer.send(TOPIC, msg)
                        print(f"Produced historical: {msg['symbol']} {msg['timeframe']} {msg['ts_candle']}")

                    last_ts[key] = ohlcv[-1][0] + 1

                except Exception as e:
                    print(f"Error fetching historical {symbol} {tf} from {exch_name}: {e}")

    producer.flush()

if __name__ == '__main__':
    print("🚀 Producer (Los Ojos) started...")
    while True:
        fetch_historical_and_publish()
        print(f"Sleeping for {FETCH_INTERVAL} seconds...")
        time.sleep(FETCH_INTERVAL)