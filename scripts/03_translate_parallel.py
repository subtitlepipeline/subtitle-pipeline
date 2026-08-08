#!/usr/bin/env python3
"""
Step 3 PARALLEL: Translate transcripts to Persian using 20x 9Router instances.
Each instance gets a chunk of segments — all translated simultaneously.
Maintains segment order and timing integrity.
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
print(f"⚡ Parallel translation using {num_routers} 9Router instances")

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

# --- Translation prompt (same quality as sequential version) ---
def build_prompt(source_name, numbered):
    return f"""You are an expert literary translator specializing in {source_name}→Persian (Farsi) translation.

Your task: Translate each subtitle line below into NATURAL, FLUENT, ELEGANT Persian.

Translation guidelines:
1. Use polished, literate Persian — the kind used in quality audiobooks and dubbing.
2. Translate the MEANING, not word-for-word. Restructure sentences to sound natural in Persian.
3. Use correct verb forms: match subject (singular/plural) with verb (یک/چند).
4. For Arabic religious phrases (صلوات، دعا، سلام): translate their FULL meaning to Persian naturally.
   - "صلی الله علیه وسلم" → "درود خدا بر او باد"
   - "رضی الله عنها" → "خدا از او راضی باشد"
   - "سبحانه و تعالی" → "منزه و برتر است"
5. Keep ALL technical terms, names, and proper nouns in their original form (AI, API, YouTube, Claude, Hermes, etc.).
6. Use formal but accessible Persian — not overly literary, not colloquial.
7. Maintain the context flow between lines — read consecutive lines as a connected narrative.
8. Keep translations concise — match roughly the length of the source line.

Return ONLY the Persian translations, one per line, numbered exactly like the input.
Do NOT add any explanation, notes, or commentary.

{numbered}"""

def call_router(port, key, prompt, max_tokens=3000, temperature=0.3, retries=3, timeout=120):
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
                return content.strip()
        except Exception as e:
            last_error = e
            if attempt < retries:
                time.sleep(3)
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

# --- Split segments into chunks for parallel translation ---
# IMPORTANT: Each chunk gets CONTEXT from neighbors to maintain coherence
BATCH_SIZE = 40
CONTEXT_LINES = 2  # Include 2 lines from previous chunk as context (not translated, just for flow)

# Create chunks
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

# --- Assign chunks to routers (round-robin) and translate in parallel ---
def translate_chunk(args):
    """Translate one chunk using one router instance."""
    chunk, router = args
    chunk_num = chunk["num"]
    start_idx = chunk["start_idx"]
    chunk_texts = chunk["texts"]
    port = router["port"]
    key = router["key"]
    router_id = router["id"]

    # Build context: include 2 previous lines for coherence (but mark them as context)
    context_prefix = ""
    if start_idx > 0:
        prev_lines = texts[max(0, start_idx - CONTEXT_LINES):start_idx]
        context_prefix = "Context (previous lines, for flow reference only — do NOT translate these):\n"
        context_prefix += "\n".join(f"[CTX] {t}" for t in prev_lines)
        context_prefix += "\n\nNow translate these lines:\n"

    numbered = "\n".join(f"{j+1}. {t}" for j, t in enumerate(chunk_texts))
    prompt = build_prompt(source_name, context_prefix + numbered)

    try:
        raw = call_router(port, key, prompt)
        translations = parse_numbered_response(raw, len(chunk_texts))
        return (chunk_num, start_idx, translations, None)
    except Exception as e:
        print(f"  ❌ Chunk {chunk_num} (router {router_id}, port {port}) FAILED: {e}")
        # Fallback: keep original text
        return (chunk_num, start_idx, chunk_texts, str(e))

# Run all chunks in parallel
print(f"🚀 Translating {total_chunks} chunks in parallel with {num_routers} routers...")

# Assign routers round-robin
chunk_router_pairs = []
for i, chunk in enumerate(chunks):
    router = routers[i % len(routers)]
    chunk_router_pairs.append((chunk, router))

start_time = time.time()

with concurrent.futures.ThreadPoolExecutor(max_workers=num_routers) as executor:
    futures = {executor.submit(translate_chunk, pair): pair for pair in chunk_router_pairs}
    results = {}
    for future in concurrent.futures.as_completed(futures):
        chunk_num, start_idx, translations, error = future.result()
        results[chunk_num] = (start_idx, translations)
        if error:
            print(f"  ⚠️ Chunk {chunk_num} had errors (fallback to original)")
        else:
            print(f"  ✅ Chunk {chunk_num}/{total_chunks} done ({len(translations)} segments)")

elapsed = time.time() - start_time
print(f"⏱️ Parallel translation completed in {elapsed:.1f}s")

# --- Merge results in original order ---
all_translations = [None] * len(texts)
for chunk_num in sorted(results.keys()):
    start_idx, translations = results[chunk_num]
    for j, t in enumerate(translations):
        idx = start_idx + j
        if idx < len(all_translations):
            all_translations[idx] = t

# Fill any gaps with original text
for i, t in enumerate(all_translations):
    if t is None:
        all_translations[i] = texts[i]

# --- Save ---
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(all_translations, f, ensure_ascii=False, indent=2)

translated_count = sum(1 for i, t in enumerate(all_translations) if t != texts[i])
print(f"✅ Translated {translated_count}/{len(all_translations)} segments ({translated_count*100//len(all_translations)}% translated)")
print(f"📁 Saved: {output_path}")
