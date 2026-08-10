#!/usr/bin/env python3
"""Test workers in PARALLEL. Pass if MAJORITY respond; report failures but don't block."""
import sys, time, concurrent.futures

keys_file = sys.argv[1]

routers = []
with open(keys_file) as f:
    for line in f:
        parts = line.strip().split("|")
        if len(parts) >= 3:
            routers.append({"id": int(parts[0]), "url": parts[1], "key": parts[2]})

MIN_OK = max(1, int(len(routers) * 0.7))  # require 70% to pass
print(f"🧪 Testing {len(routers)} workers in PARALLEL (min {MIN_OK} must respond)...")
print("=" * 60)

from openai import OpenAI

def ping(router):
    q = "Say hello in one word."
    url = router["url"].rstrip("/") + "/v1"
    try:
        client = OpenAI(base_url=url, api_key=router["key"])
        t0 = time.time()
        resp = client.chat.completions.create(
            model="oc/deepseek-v4-flash-free",
            messages=[{"role": "user", "content": q}],
            max_tokens=10,
            temperature=0,
            timeout=25
        )
        dt = time.time() - t0
        # Successful response (even empty content) = tunnel + 9Router alive
        return (router["id"], "OK", dt, "")
    except Exception as e:
        return (router["id"], "FAIL", 0, str(e)[:50])

t0 = time.time()
results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=len(routers)) as executor:
    futures = {executor.submit(ping, r): r for r in routers}
    for future in concurrent.futures.as_completed(futures):
        results.append(future.result())

results.sort(key=lambda x: x[0])
ok = 0
for rid, status, dt, err in results:
    if status == "OK":
        ok += 1
    icon = "✅" if status == "OK" else "❌"
    print(f"  {icon} Worker {rid} [{status}] {dt:.1f}s {err}")

elapsed = time.time() - t0
print("=" * 60)
print(f"📊 Result: {ok}/{len(routers)} responded ({elapsed:.1f}s). Need ≥{MIN_OK}.")

# HARD FAIL only if most workers are dead (pipeline would hang anyway)
if ok < MIN_OK:
    print(f"❌ Only {ok}/{len(routers)} — too few workers, aborting")
    sys.exit(1)

# Soft pass: continue with whatever workers are alive
print(f"✅ {ok}/{len(routers)} alive — proceeding with translation")
sys.exit(0)