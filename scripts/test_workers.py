#!/usr/bin/env python3
"""Test each worker router with a unique question and print responses."""
import sys, json, time

keys_file = sys.argv[1]

routers = []
with open(keys_file) as f:
    for line in f:
        parts = line.strip().split("|")
        if len(parts) >= 3:
            routers.append({"id": int(parts[0]), "url": parts[1], "key": parts[2]})

print(f"🧪 Testing {len(routers)} workers with unique questions...")
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
]

results = []
ok = 0
for i, router in enumerate(routers):
    q = questions[i % len(questions)]
    url = router["url"].rstrip("/") + "/v1"
    try:
        client = OpenAI(base_url=url, api_key=router["key"])
        t0 = time.time()
        resp = client.chat.completions.create(
            model="oc/deepseek-v4-flash-free",
            messages=[{"role": "user", "content": q}],
            max_tokens=60,
            temperature=0,
            timeout=60
        )
        dt = time.time() - t0
        content = (resp.choices[0].message.content or "").strip()
        if not content:
            reasoning = getattr(resp.choices[0].message, 'reasoning_content', '') or ""
            content = "(empty content, reasoning: " + reasoning[:40] + ")"
        status = "OK" if content else "EMPTY"
        if content:
            ok += 1
        results.append((router["id"], status, dt, content))
        print(f"  ✅ Worker {router['id']} [{status}] {dt:.1f}s: {content[:60]}")
    except Exception as e:
        results.append((router["id"], "FAIL", 0, str(e)[:80]))
        print(f"  ❌ Worker {router['id']} FAIL: {str(e)[:100]}")

print("=" * 60)
print(f"📊 Result: {ok}/{len(routers)} workers responded")
if ok < len(routers):
    print("❌ SOME WORKERS FAILED — check tunnel URLs above")
    sys.exit(1)
print("✅ All workers alive and responding!")