#!/usr/bin/env python3
"""
Step 3 (Final): Translate transcripts to Persian using the SAME model as Hermes chat.
Primary model: railway (oc/deepseek-v4-flash-free) — SSE-only, MUST use stream=True
Fallback model: nvidia (z-ai/glm-5.2) — fast, clean JSON

If primary model fails (empty content or error after retries), switches to fallback.
"""
import sys, os, json, time, re

transcript_path = sys.argv[1]
output_path = sys.argv[2]
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# --- Load transcript ---
with open(transcript_path, encoding="utf-8") as f:
    data = json.load(f)
segments = data["segments"]
texts = [seg["text"] for seg in segments]

# Detect source language from Whisper output
source_lang = data.get("language", "en")
lang_map = {"ar": "Arabic", "en": "English", "fa": "Persian", "tr": "Turkish",
            "ur": "Urdu", "hi": "Hindi", "es": "Spanish", "fr": "French",
            "de": "German", "ru": "Russian", "pt": "Portuguese", "id": "Indonesian"}
source_name = lang_map.get(source_lang, source_lang)

print(f"📝 {len(texts)} segments to translate (source: {source_name})")
print(f"⚡ Translation via external 9Router API (same model as Hermes chat)")

# --- API config ---
ROUTER_BASE = os.environ.get("ROUTER_BASE", "https://9router.codol.ir/v1")
ROUTER_KEY = os.environ.get("ROUTER_KEY", "")
PRIMARY_MODEL = os.environ.get("ROUTER_MODEL", "nvidia/deepseek-ai/deepseek-v4-flash-0731")
FALLBACK_MODEL = os.environ.get("FALLBACK_MODEL", "nvidia")

if not ROUTER_KEY:
    print("❌ No ROUTER_KEY found")
    sys.exit(1)

print(f"🔑 API: {ROUTER_BASE}")
print(f"📌 Primary model: {PRIMARY_MODEL} (streaming)")
print(f"📌 Fallback model: {FALLBACK_MODEL}")

from openai import OpenAI
client = OpenAI(base_url=ROUTER_BASE, api_key=ROUTER_KEY)


def call_model(model, prompt, max_tokens=8000, temperature=0.3, retries=5, timeout=240):
    """Call a model with streaming (SSE) + retries. Returns content string or raises."""
    last_error = None
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
                stream=True
            )
            content_parts = []
            reasoning_parts = []
            for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta:
                    if delta.content:
                        content_parts.append(delta.content)
                    r = getattr(delta, 'reasoning_content', None) or ""
                    if r:
                        reasoning_parts.append(r)
            content = "".join(content_parts).strip()
            reasoning = "".join(reasoning_parts)

            if content:
                return content
            elif reasoning:
                print(f"  ⚠️ [{model}] Attempt {attempt+1}/{retries+1}: content empty (reasoning {len(reasoning)} chars) — retrying")
                last_error = Exception("Empty content: reasoning only")
            else:
                print(f"  ⚠️ [{model}] Attempt {attempt+1}/{retries+1}: empty stream — retrying")
                last_error = Exception("Empty stream from model")

        except Exception as e:
            last_error = e
            err_str = str(e)
            if "429" in err_str or "FreeUsageLimitError" in err_str:
                wait = 30 * (attempt + 1)
                print(f"  ⚠️ [{model}] Attempt {attempt+1}/{retries+1}: Rate limited (429) — waiting {wait}s")
                time.sleep(wait)
                continue
            elif "500" in err_str or "502" in err_str or "503" in err_str:
                wait = 10 * (attempt + 1)
                print(f"  ⚠️ [{model}] Attempt {attempt+1}/{retries+1}: Server error — waiting {wait}s")
                time.sleep(wait)
                continue
            else:
                print(f"  ⚠️ [{model}] Attempt {attempt+1}/{retries+1}: {e}")

        if attempt < retries:
            time.sleep(5 * (attempt + 1))

    raise last_error or Exception(f"{model} failed after all retries")


def call_translate(prompt, max_tokens=8000, temperature=0.3):
    """Try primary model first; if it fails, switch to fallback model."""
    try:
        return call_model(PRIMARY_MODEL, prompt, max_tokens, temperature)
    except Exception as primary_err:
        print(f"  ❌ Primary model ({PRIMARY_MODEL}) failed: {primary_err}")
        print(f"  🔄 Switching to fallback ({FALLBACK_MODEL})...")
        return call_model(FALLBACK_MODEL, prompt, max_tokens, temperature)


def parse_numbered_response(content, expected_count):
    """Parse numbered response lines into a list."""
    lines = content.split("\n")
    result = []
    for line in lines:
        cleaned = re.sub(r'^\d+\.?\s*', '', line.strip())
        # Skip lines that look like reasoning/thinking (English meta-commentary)
        if cleaned and not cleaned.startswith("We need") and not cleaned.startswith("Let's") \
           and not cleaned.startswith("- ") and not cleaned.startswith("Better:") \
           and not cleaned.startswith("Line:"):
            result.append(cleaned)
    while len(result) < expected_count:
        result.append(result[-1] if result else "")
    return result[:expected_count]


# --- Enhanced translation prompt ---
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
9. Use ZWNJ (نیم‌فاصله) correctly in compound words: می‌رود, به‌طور, هم‌زمان, راه‌آهن, سریع‌السیر, آن‌ها, سرعت‌ها

Return ONLY the Persian translations, one per line, numbered exactly like the input.
Do NOT add any explanation, notes, or commentary.

{numbered}"""


# --- Translate in batches ---
BATCH_SIZE = 40
translations = []

for i in range(0, len(texts), BATCH_SIZE):
    batch = texts[i:i + BATCH_SIZE]
    batch_num = i // BATCH_SIZE + 1
    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"\n🔄 Batch {batch_num}/{total_batches}...")

    numbered = "\n".join(f"{j+1}. {t}" for j, t in enumerate(batch))
    prompt = build_prompt(source_name, numbered)

    try:
        raw = call_translate(prompt)
        batch_translations = parse_numbered_response(raw, len(batch))
        translations.extend(batch_translations)
        print(f"  ✅ Got {len(batch_translations)} translations")
    except Exception as e:
        print(f"  ❌ Translation failed (both primary + fallback): {e}")
        print(f"  ❌ Batch {batch_num} could not be translated — aborting")
        sys.exit(1)

# --- Save ---
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(translations, f, ensure_ascii=False, indent=2)

print(f"\n✅ Translated {len(translations)} segments")
print(f"📁 Saved: {output_path}")