import sqlite3
import json
import time
import hashlib
from typing import Optional


DB_PATH = "artifacts/cache.db"




def _get_conn():
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, response TEXT, ts INTEGER)")
return conn




def make_key(query: str) -> str:
return hashlib.sha256(query.encode('utf-8')).hexdigest()




def get_cached_response(query: str, ttl: int = 86400) -> Optional[dict]:
key = make_key(query)
conn = _get_conn()
cur = conn.cursor()
cur.execute("SELECT response, ts FROM cache WHERE key=?", (key,))
row = cur.fetchone()
if not row:
return None
response_json, ts = row
if time.time() - ts > ttl:
cur.execute("DELETE FROM cache WHERE key=?", (key,))
conn.commit()
return None
return json.loads(response_json)




def set_cached_response(query: str, response_obj: dict):
key = make_key(query)
conn = _get_conn()
cur = conn.cursor()
cur.execute("INSERT OR REPLACE INTO cache (key, response, ts) VALUES (?, ?, ?)",
(key, json.dumps(response_obj), int(time.time())))
conn.commit()