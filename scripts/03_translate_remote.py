# Script: 03_translate_remote.py
# This script is a wrapper around the parallel translation logic.
# It reads router endpoints from a file with format: idx|url|api_key
# (url = cloudflared tunnel URL of a worker's 9Router instance)

"""Step 3 PARALLEL remote v2: Two-pass translation using N remote 9Router instances.

Same logic as 03_translate_parallel.py but routers are reached via
public tunnel URLs (cloudflared) instead of local ports.

Router file format: idx|base_url|api_key
  base_url = https://xxx.trycloudflare.com  (the tunnel URL)
  The script appends /v1 when calling the OpenAI-compatible API.
"""
import sys, os, json, time, re, concurrent.futures, random

transcript_path = sys.argv[1]
output_path = sys.argv[2]
keys_file = sys.argv[3]
num_routers = int(sys.argv[4]) if len(sys.argv) > 4 else None

os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

# --- Load transcript ---
with open(transcript_path, encoding="utf-8") as f:
    data = json.load(f)
segments = data["segments"]
texts = [seg["text"] for seg in segments]
source_lang = data.get("language", "en")

lang_map = {"ar": "Arabic", "en": "English", "fa": "Persian", "tr": "Turkish",
            "ur": "Urdu", "hi": "Hindi", "es": "Spanish", "fr": "French",
            "de": "German", "ru": "Russian", "pt": "Portuguese", "id": "Indonesian"}
source_name = lang_map.get(source_lang, source_lang)

print(f"📝 {len(texts)} segments to translate (source: {source_name})")

# --- Load router endpoints ---
routers = []
with open(keys_file) as f:
    for line in f:
        parts = line.strip().split("|")
        if len(parts) >= 3:
            idx, url, key = parts[0], parts[1], parts[2]
            routers.append({"id": int(idx), "url": url, "key": key})

if num_routers:
    routers = routers[:num_routers]

print(f"🔑 Loaded {len(routers)} remote 9Router instances")
for r in routers:
    print(f"   [{r['id']}] {r['url']}")

from openai import OpenAI

# ============ PROMPTS ============

def build_translate_prompt(source_name, numbered):
    return f"""You are a master translator. Translate each English subtitle line to Persian (Farsi).

Rules:
- Translate MEANING, not word-for-word. Restructure to sound natural in Persian.
- Use polished, fluent Persian like professional dubbing and audiobooks.
- Correct verb-subject agreement (singular/plural).
- Keep technical terms and names in original form (AI, YouTube, API, etc).
- Keep each line concise — suitable for subtitles.
- Maintain context flow between consecutive lines.
- Use ZWNJ (نیم‌فاصله) in: می‌کند, کتاب‌ها, بی‌اختیار, نشسته‌اند, سریع‌تر.

Output ONLY the Persian translations, numbered. No thinking, no explanation.

{numbered}"""

def build_refine_prompt(numbered_pairs):
    """Build refinement prompt — Persian-only, detailed instructions for literary quality."""
    return f"""تو یک ویراستار حرفه‌ای ادبی هستی. هر خط فارسی زیر را به بهترین و روان‌ترین شکل ممکن بازنویسی کن.

قوانین:
- معنی هر خط باید دقیقاً حفظ شود — چیزی اضافه یا حذف نشود.
- لحن روان، طبیعی و ادبی — مثل ترجمه حرفه‌ای فیلم و سریال.
- گرامر و دستور زبان فارسی باید بی‌نقص باشد.
- فعل و فاعل باید با هم هماهنگ باشند (یک/چند).
- کلمات را به بهترین معادل فارسی برگردان.
- خط‌ها کوتاه و مناسب زیرنویس باشند.
- روانی و خوانایی متن برای مخاطب فارسی‌زبان اولویت اول است.
- نیم‌فاصله درست: می‌کند, کتاب‌ها, بی‌اختیار, نشسته‌اند, سریع‌تر.

فقط فارسی بازنویسی‌شده، شماره‌گذاری شده. بدون توضیح.

{numbered_pairs}"""

# ============ API CALL ============

def call_router(router, prompt, max_tokens=8000, temperature=0.2, retries=3, timeout=45):
    """Call a remote 9Router instance with retries (45s timeout — fast fail on dead workers)."""
    base_url = router["url"].rstrip("/") + "/v1"
    client = OpenAI(base_url=base_url, api_key=router["key"])
    last_error = None
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model="oc/deepseek-v4-flash-free",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout
            )
            content = response.choices[0].message.content or ""
            reasoning = getattr(response.choices[0].message, 'reasoning_content', '') or ""
            finish_reason = response.choices[0].finish_reason

            if not content.strip() and reasoning.strip():
                content = extract_persian_from_reasoning(reasoning)

            if content.strip():
                return content.strip()

            if finish_reason == "length":
                print(f"  ⚠️ Response truncated (finish_reason=length), trying larger max_tokens")
        except Exception as e:
            last_error = e
            print(f"  ⚠️ Attempt {attempt+1}/{retries+1} failed: {e}")
            if attempt < retries:
                time.sleep(3)
    raise last_error or Exception(f"Router {router['id']} failed after retries")

def extract_persian_from_reasoning(reasoning):
    """Extract Persian translations from reasoning_content."""
    lines = reasoning.split("\n")
    persian_lines = []
    found_numbered = False
    for line in lines:
        m = re.match(r'^\d+\.\s*(.+)', line.strip())
        if m:
            text = m.group(1).strip()
            if re.search(r'[\u0600-\u06FF]', text):
                persian_lines.append(text)
                found_numbered = True
                continue
        if found_numbered and line.strip() and not re.match(r'^\d+\.', line.strip()):
            break

    if persian_lines:
        return "\n".join(f"{i+1}. {t}" for i, t in enumerate(persian_lines))

    all_persian = [l.strip() for l in lines if re.search(r'[\u0600-\u06FF]', l.strip())]
    if all_persian:
        return "\n".join(f"{i+1}. {t}" for i, t in enumerate(all_persian))
    return ""

def call_router_with_fallback(prompt, routers, max_tokens=8000, temperature=0.2, retries=3, timeout=45, expected_count=None):
    """Try multiple routers until we get a complete response."""
    errors = []
    for router in routers:
        for attempt in range(retries):
            try:
                result = call_router(router, prompt, max_tokens=max_tokens,
                                     temperature=temperature, retries=0, timeout=timeout)
                if result.strip():
                    if expected_count:
                        parsed = parse_numbered_response(result, expected_count)
                        if len(parsed) == expected_count:
                            return result
                        else:
                            print(f"  ⚠️ Router {router['id']}: got {len(parsed)}/{expected_count} lines, retry {attempt+1}")
                            if attempt < retries - 1:
                                time.sleep(2)
                                continue
                    else:
                        return result
            except Exception as e:
                errors.append(f"router{router['id']}: {e}")
                if attempt < retries - 1:
                    time.sleep(2)
    raise Exception(f"All routers failed: {'; '.join(errors[:3])}")

def parse_numbered_response(content, expected_count):
    """Parse numbered response lines into a list."""
    lines = content.split("\n")
    result = []
    for line in lines:
        cleaned = re.sub(r'^\d+\.\s*', '', line.strip())
        if cleaned:
            result.append(cleaned)
    return result[:expected_count]

# ============ CHUNKING ============

BATCH_SIZE = 40
REFINE_BATCH_SIZE = 20
CONTEXT_LINES = 3

chunks = []
for i in range(0, len(texts), BATCH_SIZE):
    chunk_texts = texts[i:i + BATCH_SIZE]
    chunks.append({
        "start_idx": i,
        "texts": chunk_texts,
        "num": i // BATCH_SIZE + 1
    })

total_chunks = len(chunks)
print(f"📦 Split into {total_chunks} chunks of ~{BATCH_SIZE} segments each")

def get_context(start_idx):
    if start_idx == 0:
        return ""
    prev = texts[max(0, start_idx - CONTEXT_LINES):start_idx]
    ctx = "Context (previous lines for flow reference — do NOT translate these):\n"
    ctx += "\n".join(f"[CTX] {t}" for t in prev)
    ctx += "\n\nNow translate these lines:\n"
    return ctx

# ============ PASS 1: TRANSLATE ============

def translate_chunk(args):
    chunk, router = args
    chunk_num = chunk["num"]
    start_idx = chunk["start_idx"]
    chunk_texts = chunk["texts"]

    context = get_context(start_idx)
    numbered = "\n".join(f"{j+1}. {t}" for j, t in enumerate(chunk_texts))
    prompt = build_translate_prompt(source_name, context + numbered)

    router_order = routers[:]
    random.shuffle(router_order)

    try:
        raw = call_router_with_fallback(prompt, router_order, expected_count=len(chunk_texts))
        translations = parse_numbered_response(raw, len(chunk_texts))
        while len(translations) < len(chunk_texts):
            translations.append(chunk_texts[len(translations)])
        return (chunk_num, start_idx, translations, None)
    except Exception as e:
        print(f"  ❌ PASS 1 Chunk {chunk_num} FAILED after all retries: {e}")
        return (chunk_num, start_idx, chunk_texts, str(e))

def run_parallel_pass(pass_name, workers, task_fn, pairs):
    print(f"\n🚀 {pass_name}: {len(pairs)} chunks in parallel with {workers} routers...")
    start_time = time.time()

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(task_fn, pair): pair for pair in pairs}
        for future in concurrent.futures.as_completed(futures):
            chunk_num, start_idx, output, error = future.result()
            results[chunk_num] = (start_idx, output)
            if error:
                print(f"  ⚠️ {pass_name} Chunk {chunk_num} had errors (fallback)")
            else:
                print(f"  ✅ {pass_name} Chunk {chunk_num}/{len(pairs)} done")

    elapsed = time.time() - start_time
    print(f"⏱️ {pass_name} completed in {elapsed:.1f}s")
    return results

# --- PASS 1: Translate ---
translate_pairs = [(chunk, routers[i % len(routers)]) for i, chunk in enumerate(chunks)]
pass1_results = run_parallel_pass("PASS 1 (Translate)", len(routers), translate_chunk, translate_pairs)

raw_translations = [None] * len(texts)
for chunk_num in sorted(pass1_results.keys()):
    start_idx, translations = pass1_results[chunk_num]
    for j, t in enumerate(translations):
        idx = start_idx + j
        if idx < len(raw_translations):
            raw_translations[idx] = t

for i, t in enumerate(raw_translations):
    if t is None:
        raw_translations[i] = texts[i]

translated_count = sum(1 for i, t in enumerate(raw_translations) if t != texts[i])
print(f"📊 PASS 1: {translated_count}/{len(raw_translations)} segments translated")

# ============ PASS 2: REFINE ============

refine_chunks = []
for i in range(0, len(raw_translations), REFINE_BATCH_SIZE):
    refine_chunks.append({
        "start_idx": i,
        "translations": raw_translations[i:i + REFINE_BATCH_SIZE],
        "num": i // REFINE_BATCH_SIZE + 1
    })

def refine_chunk(args):
    chunk, router = args
    chunk_num = chunk["num"]
    start_idx = chunk["start_idx"]
    chunk_translations = chunk["translations"]

    pairs_text = []
    for j, fa in enumerate(chunk_translations):
        pairs_text.append(f"{j+1}. {fa}")

    numbered_pairs = "\n".join(pairs_text)
    prompt = build_refine_prompt(numbered_pairs)

    router_order = routers[:]
    random.shuffle(router_order)

    try:
        raw = call_router_with_fallback(prompt, router_order,
                                        expected_count=len(chunk_translations), temperature=0.15)
        refined = parse_numbered_response(raw, len(chunk_translations))
        while len(refined) < len(chunk_translations):
            refined.append(chunk_translations[len(refined)])
        return (chunk_num, start_idx, refined, None)
    except Exception as e:
        print(f"  ❌ PASS 2 Chunk {chunk_num} FAILED after all retries: {e}")
        return (chunk_num, start_idx, chunk_translations, str(e))

refine_pairs = [(chunk, routers[i % len(routers)]) for i, chunk in enumerate(refine_chunks)]
print(f"\n📦 PASS 2: {len(refine_chunks)} refine chunks of ~{REFINE_BATCH_SIZE} segments each")
pass2_results = run_parallel_pass("PASS 2 (Refine)", len(routers), refine_chunk, refine_pairs)

all_translations = [None] * len(texts)
for chunk_num in sorted(pass2_results.keys()):
    start_idx, translations = pass2_results[chunk_num]
    for j, t in enumerate(translations):
        idx = start_idx + j
        if idx < len(all_translations):
            all_translations[idx] = t

for i, t in enumerate(all_translations):
    if t is None:
        all_translations[i] = raw_translations[i]

# ============ PASS 3: UNIFY (whole-subtitle consistency) ============
# New phase: review the ENTIRE subtitle text for consistent tone, terminology,
# and natural flow. Uses overlapping windows so every line gets reviewed
# with both its neighbors AND context from the full document.

def build_unify_prompt(all_lines, start_idx, window_lines):
    """Prompt for PASS 3: unify tone/terminology across the whole subtitle."""
    full_preview = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(all_lines[:40]))
    window_text = "\n".join(f"[{start_idx + j + 1}] {t}" for j, t in enumerate(window_lines))
    return f"""تو ویراستار ارشد زیرنویس فارسی هستی. کل زیرنویس را یکپارچه کن.

نکات کلی برای حفظ یکنواختی در کل متن:
- یک اصطلاح انگلیسی در همه جای متن با یک معادل فارسی ثابت ترجمه شود.
- لحن در کل متن یکدست باشد (جدی/خودمانی/آموزشی متناسب با ویدیو).
- فعل‌ها و ضمایر در کل متن هماهنگ باشند.
- روانی و طبیعی بودن دیالوگ اولویت دارد؛ ترجمه تحت‌اللفظی ممنوع.
- نیم‌فاصله (ZWNJ) درست رعایت شود: می‌کند, کتاب‌ها, بی‌اختیار, نشسته‌اند, سریع‌تر.

نمونه‌ای از کل متن (برای درک لحن و اصطلاحات):
{full_preview}

حالا این خطوط را طوری بازنویسی کن که با کل متن هماهنگ باشند.
فقط فارسی بازنویسی‌شده، شماره‌گذاری شده (همان شماره‌ها). بدون توضیح.

{window_text}"""

def unify_chunk(args):
    """PASS 3: Unify a window of lines for global consistency."""
    chunk, router = args
    chunk_num = chunk["num"]
    start_idx = chunk["start_idx"]
    window_lines = chunk["lines"]

    prompt = build_unify_prompt(all_translations, start_idx, window_lines)

    router_order = routers[:]
    random.shuffle(router_order)

    try:
        raw = call_router_with_fallback(prompt, router_order,
                                        expected_count=len(window_lines),
                                        temperature=0.1, max_tokens=8000)
        unified = parse_numbered_response(raw, len(window_lines))
        # Pad if short (keep original window lines, offset by window position)
        while len(unified) < len(window_lines):
            unified.append(window_lines[len(unified)])
        return (chunk_num, start_idx, unified, None)
    except Exception as e:
        print(f"  ❌ PASS 3 Chunk {chunk_num} FAILED after all retries: {e}")
        return (chunk_num, start_idx, window_lines, str(e))

# Overlapping windows: 30 lines per window with 10-line overlap
WINDOW_SIZE = 30
OVERLAP = 10
unify_chunks = []
win_num = 1
start = 0
while start < len(all_translations):
    window = all_translations[start:start + WINDOW_SIZE]
    unify_chunks.append({"start_idx": start, "lines": window, "num": win_num})
    win_num += 1
    start += WINDOW_SIZE - OVERLAP

print(f"\n📦 PASS 3 (Unify): {len(unify_chunks)} windows of ~{WINDOW_SIZE} lines (overlap {OVERLAP})")

unify_pairs = [(chunk, routers[i % len(routers)]) for i, chunk in enumerate(unify_chunks)]
pass3_results = run_parallel_pass("PASS 3 (Unify)", len(routers), unify_chunk, unify_pairs)

# Merge PASS 3: for overlapping regions, last writer wins (later windows refine earlier)
# Count votes per index to pick most consistent version
from collections import defaultdict
vote_counts = defaultdict(int)
unified_votes = defaultdict(dict)

for chunk_num in sorted(pass3_results.keys()):
    start_idx, lines = pass3_results[chunk_num]
    for j, line in enumerate(lines):
        idx = start_idx + j
        if idx < len(all_translations):
            vote_counts[idx] += 1
            unified_votes[idx][chunk_num] = line

# For each line, prefer the version from the LAST window (most context), else majority
final_wa = list(all_translations)
for idx in range(len(all_translations)):
    if idx in unified_votes and unified_votes[idx]:
        # Take version from the highest-numbered chunk (later window)
        best_chunk = max(unified_votes[idx].keys())
        final_wa[idx] = unified_votes[idx][best_chunk]

all_translations = final_wa
unified_change = sum(1 for i, t in enumerate(all_translations)
                     if i < len(raw_translations) and t != raw_translations[i])
print(f"📊 PASS 3: {unified_change} lines refined for consistency")

# --- Save ---
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(all_translations, f, ensure_ascii=False, indent=2)

final_translated = sum(1 for i, t in enumerate(all_translations) if t != texts[i])
print(f"\n✅ Final: {final_translated}/{len(all_translations)} segments translated ({final_translated*100//len(all_translations)}% translated)")
print(f"📁 Saved: {output_path}")