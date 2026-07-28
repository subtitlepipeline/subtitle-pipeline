#!/usr/bin/env python3
"""Step 5: Burn-in ASS subtitles into video with ffmpeg."""
import sys, os, subprocess

video_path = sys.argv[1]
ass_path = sys.argv[2]
output_path = sys.argv[3]
os.makedirs(os.path.dirname(output_path), exist_ok=True)

cmd = [
    "ffmpeg", "-y",
    "-i", video_path,
    "-vf", f"ass={ass_path}",
    "-c:v", "libx264",
    "-crf", "23",
    "-preset", "fast",
    "-c:a", "aac",
    "-b:a", "128k",
    output_path
]

print(f"🔥 Burning subtitles into video...")
subprocess.run(cmd, check=True)
print(f"✅ Burn-in complete: {output_path}")

# Probe output size
size = os.path.getsize(output_path)
print(f"📊 Output size: {size / (1024*1024):.1f} MB")
