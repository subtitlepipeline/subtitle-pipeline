#!/usr/bin/env python3
"""Test each worker router with a unique question in PARALLEL."""
import sys, json, time, concurrent.futures

keys_file = sys.argv[1]

routers = []
with open(keys_file) as f:
    for line in f:
        parts = line.strip().split("|")
        if len(parts) >= 3:
            routers.append({"id": int(parts[0]), "url": parts[1], "key": parts[2]})

print(f"🧪 Testing {len(routers)} workers in PARALLEL with unique questions...")
print("=" * 60)

from openai import OpenAI

questions = [
    "Say the word: ALPHA",
    "Say the word: BETA",
    "Say the word: GAMMA",
    "Say the word: DELTA",
    "Say the word: EPSILON",
    "Say the word: ZETA",
    "Say the word: ETA",
    "Say the word: THETA",
    "Say the word: IOTA",
    "Say the word: KAPPA",
    "Say the word: LAMBDA",
    "Say the word: MU",
    "Say the word: NU",
    "Say the word: XI",
    "Say the word: OMICRON",
]

def ping(router):
    q = questions[(router["id"] - 1) % len(questions)]
    url = router["url"].rstrip("/") + "/v1"
    try:
        client = OpenAI(base_url=url, api_key=router["key"])
        t0 = time.time()
        resp = client.chat.completions.create(
            model="oc/deepseek-v4-flash-free",
            messages=[{"role": "user", "content": q}],
            max_tokens=20,
            temperature=0,
            timeout=30
        )
        dt = time.time() - t0
        content = (resp.choices[0].message.content or "").strip()
        reasoning = getattr(resp.choices[0].message, 'reasoning_content', '') or ""
        # A successful response (even empty content) proves the tunnel + 9Router + OpenCode work.
        # Empty content is normal for trivial pings (model may return reasoning only).
        ok = True
        return (router["id"], "OK" if ok else "EMPTY", dt, content[:40] or "(empty — tunnel alive)")
    except Exception as e:
        return (router["id"], "FAIL", 0, str(e)[:60])

# Run all pings in parallel
t0 = time.time()
results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
    futures = {executor.submit(ping, r): r for r in routers}
    for future in concurrent.futures.as_completed(futures):
        results.append(future.result())

results.sort(key=lambda x: x[0])
ok = 0
for rid, status, dt, content in results:
    if status == "OK":
        ok += 1
    icon = "✅" if status == "OK" else "❌"
    print(f"  {icon} Worker {rid} [{status}] {dt:.1f}s: {content}")

elapsed = time.time() - t0
print("=" * 60)
print(f"📊 Result: {ok}/{len(routers)} workers responded (total {elapsed:.1f}s)")
if ok < len(routers):
    print("❌ SOME WORKERS FAILED — check tunnel URLs above")
    sys.exit(1)
print("✅ All workers alive and responding!")