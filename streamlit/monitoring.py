
# monitoring.py - lightweight utilities for alerts / logging
import smtplib, json, requests, time
from pathlib import Path

def send_telegram(bot_token, chat_id, text):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {'chat_id': chat_id, 'text': text}
    r = requests.post(url, data=payload, timeout=10)
    return r.ok

def append_log(path, record: dict):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'a') as f:
        f.write(json.dumps(record, default=str) + '\\n')
