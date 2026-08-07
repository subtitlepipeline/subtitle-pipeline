#!/usr/bin/env python3
"""
Step 3: Translate transcripts to Persian using DeepSeek via LiteLLM.
One-pass with an enhanced, high-quality translation prompt.
"""
import sys, os, json, subprocess, time, re

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
print(f"⚡ Enhanced one-pass translation")

# --- Start LiteLLM proxy ---
api_keys_str = os.environ.get("NVIDIA_API_KEYS", "")
api_keys = [k.strip() for k in api_keys_str.split(",") if k.strip()]

if not api_keys:
    print("❌ No NVIDIA_API_KEYS found")
    sys.exit(1)

print(f"🔑 Found {len(api_keys)} NVIDIA API keys")

# Build LiteLLM config
config = {"model_list": []}
for i, key in enumerate(api_keys):
    config["model_list"].append({
        "model_name": "deepseek",
        "litellm_params": {
            "model": "openai/openai/gpt-oss-120b",
            "api_base": "https://integrate.api.nvidia.com/v1",
            "api_key": key
        }
    })
config["router_settings"] = {"routing_strategy": "simple-shuffle"}
config["litellm_settings"] = {"drop_params": True, "allowed_fails": 100, "cooldown_time": 1}

config_path = "/tmp/litellm_config.yaml"
import yaml
with open(config_path, "w") as f:
    yaml.dump(config, f)

print("🚀 Starting LiteLLM proxy on port 4000...")
litellm_proc = subprocess.Popen(
    ["litellm", "--config", config_path, "--host", "0.0.0.0", "--port", "4000"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT
)

# Wait for proxy to be ready
print("⏳ Waiting for LiteLLM to start...")
for attempt in range(60):
    try:
        import requests
        r = requests.post("http://localhost:4000/v1/chat/completions",
            json={"model": "deepseek", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5},
            timeout=10)
        if r.status_code in (200, 400):
            print("✅ LiteLLM is ready!")
            break
    except:
        pass
    time.sleep(2)
else:
    print("❌ LiteLLM failed to start")
    # Print LiteLLM logs for debugging
    litellm_proc.terminate()
    try:
        out = litellm_proc.stdout.read().decode('utf-8', errors='replace')
        print(f"LiteLLM logs:\n{out[-2000:]}")
    except:
        pass
    sys.exit(1)

from openai import OpenAI
client = OpenAI(base_url="http://localhost:4000/v1", api_key="sk-anything")

def call_deepseek(prompt, max_tokens=3000, temperature=0.3, retries=3, timeout=120):
    """Call DeepSeek with retries and timeout."""
    last_error = None
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model="deepseek",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout
            )
            content = response.choices[0].message.content or ""
            if content.strip():
                return content.strip()
        except Exception as e:
            last_error = e
            print(f"  ⚠️ Attempt {attempt+1}/{retries+1} failed: {e}")
            if attempt < retries:
                time.sleep(3)
    raise last_error or Exception("DeepSeek failed after retries")

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

# --- Enhanced translation prompt ---
# This prompt is carefully designed to produce fluent, natural Persian
# translations in a single pass, covering all the quality requirements
# that were previously handled by the two-pass approach.

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

# --- Translate in batches ---
BATCH_SIZE = 40
translations = []

for i in range(0, len(texts), BATCH_SIZE):
    batch = texts[i:i + BATCH_SIZE]
    batch_num = i // BATCH_SIZE + 1
    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"🔄 Batch {batch_num}/{total_batches}...")

    numbered = "\n".join(f"{j+1}. {t}" for j, t in enumerate(batch))
    prompt = build_prompt(source_name, numbered)

    try:
        raw = call_deepseek(prompt)
    except Exception as e:
        print(f"  ❌ Translation failed: {e}")
        translations.extend(batch)
        continue

    batch_translations = parse_numbered_response(raw, len(batch))
    translations.extend(batch_translations)
    print(f"  ✅ Got {len(batch_translations)} translations")

# --- Cleanup LiteLLM ---
litellm_proc.kill()
try:
    litellm_proc.wait(timeout=5)
except:
    pass

# --- Save ---
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(translations, f, ensure_ascii=False, indent=2)

print(f"✅ Translated {len(translations)} segments")
print(f"📁 Saved: {output_path}")
