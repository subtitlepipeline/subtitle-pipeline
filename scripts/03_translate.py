#!/usr/bin/env python3
"""
Step 3: Translate transcripts to Persian using DeepSeek via LiteLLM.
- English→Persian: one-pass (fast)
- Non-English→Persian (Arabic, Turkish, etc.): two-pass + self-correction (quality)
"""
import sys, os, json, subprocess, time, re, http.client

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

# All languages use two-pass + self-correction for maximum quality
print(f"🔁 Two-pass + self-correction enabled")

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
            "model": "openai/deepseek-ai/deepseek-v4-flash",
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
    litellm_proc.terminate()
    out, _ = litellm_proc.communicate(timeout=5)
    print(out.decode()[-2000:])
    sys.exit(1)

from openai import OpenAI
client = OpenAI(base_url="http://localhost:4000/v1", api_key="sk-anything")

def call_deepseek(prompt, max_tokens=2000, temperature=0.3, retries=3, timeout=120):
    """Call DeepSeek with retries and timeout on failure."""
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
            if attempt < retries:
                time.sleep(2)
    raise last_error or Exception("DeepSeek failed after retries")

def parse_numbered_response(content, expected_count):
    """Parse numbered response lines into a list."""
    lines = content.split("\n")
    result = []
    for line in lines:
        cleaned = re.sub(r'^\d+\.\s*', '', line.strip())
        if cleaned:
            result.append(cleaned)
    # Fix count mismatch
    while len(result) < expected_count:
        result.append(result[-1] if result else "")
    return result[:expected_count]

# --- Translate in batches ---
BATCH_SIZE = 40
translations = []

for i in range(0, len(texts), BATCH_SIZE):
    batch = texts[i:i + BATCH_SIZE]
    batch_num = i // BATCH_SIZE + 1
    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"🔄 Batch {batch_num}/{total_batches}...")

    numbered = "\n".join(f"{j+1}. {t}" for j, t in enumerate(batch))

    # === PASS 1: Translate ===
    prompt1 = f"""You are a professional translator from {source_name} to Persian (Farsi).
Translate the following {source_name} subtitle lines to fluent, natural Persian.
For Arabic religious phrases (salawat, du'a, etc.): translate their meaning to Persian.
Return ONLY the Persian translations, one per line, numbered exactly as input.
Keep technical terms in English when they are common.

{numbered}"""

    try:
        raw = call_deepseek(prompt1)
    except Exception as e:
        print(f"  ❌ Translation failed: {e}")
        translations.extend(batch)
        continue

    pass1 = parse_numbered_response(raw, len(batch))
    print(f"  ✅ Pass 1: {len(pass1)} translations")

    # === PASS 2: Self-correction & polishing (all languages) ===
    pass1_numbered = "\n".join(f"{j+1}. {t}" for j, t in enumerate(pass1))
    prompt2 = f"""You are a Persian language editor. Review and improve the following Persian translations.

First, check for these issues:
1. Arabic phrases left untranslated → translate them to Persian
2. Grammatical errors (wrong verb conjugations, pronoun mismatches)
3. Unnatural or awkward phrasing → make it fluent Persian

Then rewrite each line to be NATURAL, FLUENT Persian.
Return ONLY the corrected Persian translations, one per line, numbered exactly as input.
Do NOT change the meaning.

{pass1_numbered}"""

    try:
        raw2 = call_deepseek(prompt2, max_tokens=2000, temperature=0.2)
    except Exception as e:
        print(f"  ⚠️ Self-correction failed, keeping pass 1: {e}")
        translations.extend(pass1)
        continue

    pass2 = parse_numbered_response(raw2, len(batch))
    print(f"  ✅ Pass 2: corrected {sum(1 for a,b in zip(pass1, pass2) if a != b)}/{len(pass2)} lines")
    translations.extend(pass2)

# --- Cleanup LiteLLM ---
litellm_proc.terminate()
litellm_proc.wait(timeout=10)

# --- Save ---
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(translations, f, ensure_ascii=False, indent=2)

print(f"✅ Translated {len(translations)} segments")
print(f"📁 Saved: {output_path}")
