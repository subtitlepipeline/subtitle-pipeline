#!/usr/bin/env python3
"""Step 2: Whisper transcription with large-v3 (best quality)."""
import sys, os, json
from faster_whisper import WhisperModel

video_path = sys.argv[1]
output_path = sys.argv[2]
os.makedirs(os.path.dirname(output_path), exist_ok=True)

print("🎤 Loading Whisper large-v3 (this takes a few minutes)...")
# large-v3 = best quality, int8 for CPU efficiency
model = WhisperModel("large-v3", device="cpu", compute_type="int8")

print("🎤 Transcribing...")
segments, info = model.transcribe(video_path, beam_size=5, vad_filter=True)

seg_list = []
for seg in segments:
    seg_list.append({
        "start": round(seg.start, 2),
        "end": round(seg.end, 2),
        "text": seg.text.strip()
    })

with open(output_path, "w", encoding="utf-8") as f:
    json.dump({"segments": seg_list}, f, ensure_ascii=False, indent=2)

print(f"✅ Whisper done: {len(seg_list)} segments, language: {info.language}")
print(f"📁 Saved: {output_path}")
