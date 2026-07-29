#!/usr/bin/env python3
"""
Step 3: Translate transcripts to Persian using DeepSeek via LiteLLM.
Starts a local LiteLLM proxy with NVIDIA API keys, then translates in batches.
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
source_lang = data.get("language", "auto")
print(f"📝 {len(texts)} segments to translate (source: {source_lang})")

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
    config["model_list"].extend([
        {
            "model_name": "deepseek",
            "litellm_params": {
                "model": "openai/deepseek-ai/deepseek-v4-flash",
                "api_base": "https://integrate.api.nvidia.com/v1",
                "api_key": key
            }
        },
        {
            "model_name": "glm",
            "litellm_params": {
                "model": "openai/z-ai/glm-5.2",
                "api_base": "https://integrate.api.nvidia.com/v1",
                "api_key": key
            }
        }
    ])
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
    # Print logs
    litellm_proc.terminate()
    out, _ = litellm_proc.communicate(timeout=5)
    print(out.decode()[-2000:])
    sys.exit(1)

# --- Translate in batches ---
BATCH_SIZE = 20
translations = []

from openai import OpenAI
client = OpenAI(base_url="http://localhost:4000/v1", api_key="sk-anything")

for i in range(0, len(texts), BATCH_SIZE):
    batch = texts[i:i + BATCH_SIZE]
    batch_num = i // BATCH_SIZE + 1
    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"🔄 Translating batch {batch_num}/{total_batches}...")

    numbered = "\n".join(f"{j+1}. {t}" for j, t in enumerate(batch))
    lang_name = {"ar": "Arabic", "en": "English", "fa": "Persian", "tr": "Turkish", "ur": "Urdu", "hi": "Hindi"}
    source_name = lang_name.get(source_lang, source_lang)
    prompt = f"""Translate the following {source_name} subtitle lines to Persian (Farsi). 
Return ONLY the Persian translations, one per line, numbered exactly like the input.
Keep technical terms (AI, API, YouTube, etc.) in English. Do NOT add any explanation.
If there are Arabic religious phrases (like salawat, du'a), translate them to Persian too.
Use natural, fluent Persian.

{numbered}"""

    try:
        response = client.chat.completions.create(
            model="deepseek",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.3
        )
        content = response.choices[0].message.content.strip()
        
        # Parse numbered lines
        lines = content.split("\n")
        batch_translations = []
        for line in lines:
            # Remove leading number and dot
            cleaned = re.sub(r'^\d+\.\s*', '', line.strip())
            if cleaned:
                batch_translations.append(cleaned)
        
        # Ensure we have exactly BATCH_SIZE translations
        while len(batch_translations) < len(batch):
            batch_translations.append(batch_translations[-1] if batch_translations else "")
        
        translations.extend(batch_translations[:len(batch)])
        print(f"  ✅ Got {len(batch_translations[:len(batch)])} translations")
        
    except Exception as e:
        print(f"  ❌ Error: {e}, using fallback to GLM")
        try:
            response = client.chat.completions.create(
                model="glm",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.3
            )
            content = response.choices[0].message.content.strip()
            # Strip CJK that GLM injects
            content = re.sub(r'[\u4e00-\u9fff]+', '', content)
            lines = content.split("\n")
            batch_translations = []
            for line in lines:
                cleaned = re.sub(r'^\d+\.\s*', '', line.strip())
                if cleaned:
                    batch_translations.append(cleaned)
            while len(batch_translations) < len(batch):
                batch_translations.append(batch_translations[-1] if batch_translations else "")
            translations.extend(batch_translations[:len(batch)])
        except Exception as e2:
            print(f"  ❌ GLM also failed: {e2}")
            translations.extend(batch)  # fallback to English

# --- Cleanup LiteLLM ---
litellm_proc.terminate()
litellm_proc.wait(timeout=10)

# --- Save ---
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(translations, f, ensure_ascii=False, indent=2)

print(f"✅ Translated {len(translations)} segments")
print(f"📁 Saved: {output_path}")
