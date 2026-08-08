#!/usr/bin/env python3
"""Step 2: Whisper transcription with automatic model selection.
- English audio → base model (74MB, fast)
- Non-English audio → large-v3 (best quality)
"""
import sys, os, json
from faster_whisper import WhisperModel

video_path = sys.argv[1]
output_path = sys.argv[2]
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Quick detect language with tiny model first
print("🔍 Detecting language (quick scan)...")
detect_model = WhisperModel("tiny", device="cpu", compute_type="int8")
_, info = detect_model.transcribe(video_path, beam_size=1, vad_filter=True)
lang = info.language
print(f"🌐 Detected language: {lang}")

# Choose model based on language
if lang == "en":
    model_name = "base"
    print(f"🎤 English detected → using '{model_name}' model (74MB, fast)")
else:
    model_name = "large-v3"
    print(f"🎤 Non-English ({lang}) → using '{model_name}' model (best quality)")

model = WhisperModel(model_name, device="cpu", compute_type="int8")

print(f"🎤 Transcribing with {model_name}...")
segments, info = model.transcribe(video_path, beam_size=5, vad_filter=True)

seg_list = []
for seg in segments:
    seg_list.append({
        "start": round(seg.start, 2),
        "end": round(seg.end, 2),
        "text": seg.text.strip()
    })

with open(output_path, "w", encoding="utf-8") as f:
    json.dump({"segments": seg_list, "language": info.language}, f, ensure_ascii=False, indent=2)

print(f"✅ Whisper done: {len(seg_list)} segments, language: {info.language} (model: {model_name})")
print(f"📁 Saved: {output_path}")
