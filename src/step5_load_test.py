"""
Step 5 - Load test the live API.

Sends POST /predict requests with many requests in flight at once and measures
requests per second, latency (median / 95th / 99th percentile) and errors.
Also times /predict/batch, which scores 100 players per request.

Run:  python src/step5_load_test.py https://playerpulse-api.vercel.app
Output: outputs/load_test.csv
"""
import asyncio
import statistics
import sys
import time

import httpx
import pandas as pd
from pathlib import Path

URL = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
OUT = Path(__file__).resolve().parents[1] / "outputs" / "load_test.csv"
PLAYER = {"charclass": "Mage", "level_end": 70, "levels_gained": 0, "hours_played": 22.5, "active_days": 9,
          "n_zones": 14, "n_guilds": 1, "share_city": 0.3, "share_dungeon": 0.1, "share_battleground": 0.05,
          "share_arena": 0.0, "share_in_guild": 0.9, "days_since_last_seen": 6, "tenure_days": 180,
          "prev_hours": 30, "hours_change": -7.5}


async def run(path, body, concurrency, total):
    lat, errors = [], 0
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=30, limits=httpx.Limits(max_connections=concurrency)) as c:
        await c.post(URL + path, json=body)            # warm up
        async def one():
            nonlocal errors
            async with sem:
                t = time.perf_counter()
                try:
                    r = await c.post(URL + path, json=body)
                    if r.status_code != 200:
                        errors += 1
                except Exception:
                    errors += 1
                lat.append((time.perf_counter() - t) * 1000)
        t0 = time.perf_counter()
        await asyncio.gather(*[one() for _ in range(total)])
        wall = time.perf_counter() - t0
    lat.sort()
    q = lambda p: lat[min(len(lat) - 1, int(p * len(lat)))]
    return {"endpoint": path, "concurrency": concurrency, "requests": total,
            "requests_per_sec": round(total / wall, 1), "players_per_sec": None,
            "p50_ms": round(statistics.median(lat), 1), "p95_ms": round(q(0.95), 1),
            "p99_ms": round(q(0.99), 1), "error_rate": round(errors / total, 4)}


async def main():
    rows = []
    for conc in [10, 50, 100]:
        r = await run("/predict", PLAYER, conc, conc * 20)
        r["players_per_sec"] = r["requests_per_sec"]
        rows.append(r); print(r)
    r = await run("/predict/batch", [PLAYER] * 100, 20, 200)
    r["players_per_sec"] = round(r["requests_per_sec"] * 100, 1)
    rows.append(r); print(r)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nSaved {OUT}")
    print(pd.DataFrame(rows).to_string(index=False))

asyncio.run(main())
