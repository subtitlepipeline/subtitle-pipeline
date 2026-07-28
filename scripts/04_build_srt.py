#!/usr/bin/env python3
"""Step 4: Build SRT from transcript JSON + translations JSON."""
import sys, os, json

transcript_path = sys.argv[1]
translations_path = sys.argv[2]
output_path = sys.argv[3]
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(transcript_path, encoding="utf-8") as f:
    data = json.load(f)
with open(translations_path, encoding="utf-8") as f:
    translations = json.load(f)

segments = data["segments"]

def sec_to_srt(s):
    h=int(s//3600); m=int((s%3600)//60); sec=int(s%60); ms=int((s-int(s))*1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"

lines = []
for i, seg in enumerate(segments):
    text = translations[i] if i < len(translations) else seg["text"]
    lines.append(f"{i+1}")
    lines.append(f"{sec_to_srt(seg['start'])} --> {sec_to_srt(seg['end'])}")
    lines.append(text)
    lines.append("")

with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"✅ SRT saved: {output_path} ({len(segments)} entries)")
