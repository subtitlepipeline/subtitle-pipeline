#!/usr/bin/env python3
"""
Step 3 PARALLEL v2: Two-pass translation using 20x 9Router instances.

PASS 1: Translate all chunks in parallel (20 routers) → raw Persian
PASS 2: Refine each chunk in parallel (20 routers) → polished literary Persian

Each pass uses all 20 routers simultaneously.
"""
import sys, os, json, time, re, concurrent.futures

transcript_path = sys.argv[1]
output_path = sys.argv[2]
keys_file = sys.argv[3]
base_port = int(sys.argv[4])
num_routers = int(sys.argv[5])

os.makedirs(os.path.dirname(output_path), exist_ok=True)

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
print(f"⚡ Two-pass parallel translation using {num_routers} 9Router instances")

# --- Load router keys ---
routers = []
with open(keys_file) as f:
    for line in f:
        parts = line.strip().split("|")
        if len(parts) == 3:
            idx, port, key = parts
            routers.append({"id": int(idx), "port": int(port), "key": key})

print(f"🔑 Loaded {len(routers)} router instances (ports {routers[0]['port']}-{routers[-1]['port']})")

from openai import OpenAI

# ============ PROMPTS ============

def build_translate_prompt(source_name, numbered):
    return f"""You are a master translator specializing in {source_name}→Persian (Farsi) translation.

Translate each subtitle line below into accurate, natural Persian.

Rules:
1. Translate the MEANING faithfully — do not add, remove, or change meaning.
2. Use correct grammar: match subjects with verbs (singular/plural).
3. Keep ALL technical terms, names, and proper nouns in original form (AI, API, YouTube, Claude, Hermes, etc.).
4. For Arabic religious phrases, translate their full meaning naturally:
   - "صلی الله علیه وسلم" → "درود خدا بر او باد"
   - "رضی الله عنها" → "خدا از او راضی باشد"
   - "سبحانه و تعالی" → "منزه و برتر است"
5. Keep translations concise — match roughly the length of the source line.
6. Maintain context flow between consecutive lines.

Return ONLY the Persian translations, one per line, numbered exactly like the input.
Do NOT add any explanation, notes, or commentary.

{numbered}"""

def build_refine_prompt(numbered_pairs):
    """Build refinement prompt — receives English+Persian pairs, returns polished Persian."""
    return f"""You are an elite Persian literary editor and proofreader.

Below are subtitle lines with their original English source and an initial Persian translation.
Your task: Refine each Persian translation to the HIGHEST literary quality.

Refinement rules — follow STRICTLY:
1. REWRITE each line into the most elegant, fluent, polished Persian possible.
2. Use the kind of Persian found in quality audiobooks, professional dubbing, and literary translations.
3. Fix any grammar errors, awkward phrasing, or unnatural word choices.
4. Ensure perfect verb-subject agreement (یک/چند).
5. Use sophisticated but accessible vocabulary — not overly archaic, not colloquial.
6. Maintain EXACT meaning — do NOT add, remove, or alter the content.
7. The refined line must match the ORIGINAL English in meaning 100%.
8. Keep translations concise — suitable for subtitles (readable in the time shown on screen).
9. Keep ALL technical terms, names, and proper nouns in original form.
10. Ensure smooth flow between consecutive lines.

Return ONLY the refined Persian translations, one per line, numbered exactly like the input.
Do NOT include the English. Do NOT add any explanation or commentary.

{numbered_pairs}"""

# ============ API CALL ============

def call_router(port, key, prompt, max_tokens=4000, temperature=0.2, retries=3, timeout=180):
    """Call a 9Router instance with retries."""
    client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key=key)
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
            if not content and hasattr(response.choices[0].message, 'reasoning_content'):
                content = response.choices[0].message.reasoning_content or ""
            if content.strip():
                # Clean up reasoning artifacts if present
                content = re.sub(r'^\s*', '', content, flags=re.DOTALL)
                content = re.sub(r'^.*?\s*', '', content, flags=re.DOTALL)
                if content.strip():
                    return content.strip()
        except Exception as e:
            last_error = e
            if attempt < retries:
                time.sleep(5)
    raise last_error or Exception(f"Router on port {port} failed after retries")

def parse_numbered_response(content, expected_count):
    """Parse numbered response lines into a list."""
    lines = content.split("\n")
    result = []
    for line in lines:
        cleaned = re.sub(r'^\d+\.\s*', '', line.strip())
        if cleaned:
            result.append(cleaned)
    while len(result) < expected_count:
        result.append(result[-1] if result else "")
    return result[:expected_count]

# ============ CHUNKING ============

BATCH_SIZE = 40
CONTEXT_LINES = 3  # More context for better coherence

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
    """Get context lines from previous chunk."""
    if start_idx == 0:
        return ""
    prev = texts[max(0, start_idx - CONTEXT_LINES):start_idx]
    ctx = "Context (previous lines for flow reference — do NOT translate these):\n"
    ctx += "\n".join(f"[CTX] {t}" for t in prev)
    ctx += "\n\nNow translate these lines:\n"
    return ctx

# ============ PASS 1: TRANSLATE ============

def translate_chunk(args):
    """PASS 1: Translate one chunk of English → Persian."""
    chunk, router = args
    chunk_num = chunk["num"]
    start_idx = chunk["start_idx"]
    chunk_texts = chunk["texts"]
    port = router["port"]
    key = router["key"]

    context = get_context(start_idx)
    numbered = "\n".join(f"{j+1}. {t}" for j, t in enumerate(chunk_texts))
    prompt = build_translate_prompt(source_name, context + numbered)

    try:
        raw = call_router(port, key, prompt)
        translations = parse_numbered_response(raw, len(chunk_texts))
        return (chunk_num, start_idx, translations, None)
    except Exception as e:
        print(f"  ❌ PASS 1 Chunk {chunk_num} (port {port}) FAILED: {e}")
        return (chunk_num, start_idx, chunk_texts, str(e))

def run_parallel_pass(pass_name, workers, task_fn, pairs):
    """Run a translation/refinement pass in parallel."""
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
pass1_results = run_parallel_pass("PASS 1 (Translate)", num_routers, translate_chunk, translate_pairs)

# Merge pass 1 results
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

def refine_chunk(args):
    """PASS 2: Refine Persian translations to literary quality."""
    chunk, router = args
    chunk_num = chunk["num"]
    start_idx = chunk["start_idx"]
    chunk_texts = chunk["texts"]  # English originals
    chunk_translations = raw_translations[start_idx:start_idx + len(chunk_texts)]  # Persian
    port = router["port"]
    key = router["key"]

    # Build pairs: "1. EN: <english> | FA: <persian>"
    pairs_text = []
    for j, (en, fa) in enumerate(zip(chunk_texts, chunk_translations)):
        pairs_text.append(f"{j+1}. EN: {en} | FA: {fa}")

    numbered_pairs = "\n".join(pairs_text)
    prompt = build_refine_prompt(numbered_pairs)

    try:
        raw = call_router(port, key, prompt, temperature=0.15)
        refined = parse_numbered_response(raw, len(chunk_texts))
        return (chunk_num, start_idx, refined, None)
    except Exception as e:
        print(f"  ❌ PASS 2 Chunk {chunk_num} (port {port}) FAILED: {e}")
        # Fallback: keep pass 1 translation
        return (chunk_num, start_idx, chunk_translations, str(e))

# --- PASS 2: Refine ---
refine_pairs = [(chunk, routers[i % len(routers)]) for i, chunk in enumerate(chunks)]
pass2_results = run_parallel_pass("PASS 2 (Refine)", num_routers, refine_chunk, refine_pairs)

# Merge pass 2 results
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

# --- Save ---
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(all_translations, f, ensure_ascii=False, indent=2)

final_translated = sum(1 for i, t in enumerate(all_translations) if t != texts[i])
print(f"\n✅ Final: {final_translated}/{len(all_translations)} segments translated ({final_translated*100//len(all_translations)}% translated)")
print(f"📁 Saved: {output_path}")
