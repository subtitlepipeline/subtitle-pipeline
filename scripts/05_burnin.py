#!/usr/bin/env python3
"""Step 5: Burn-in ASS subtitles into video with ffmpeg.
Outputs 3 resolutions: 1080p, 720p, 480p in one pass.
"""
import sys, os, subprocess

video_path = sys.argv[1]
ass_path = sys.argv[2]
output_prefix = sys.argv[3]  # e.g. "output/My Video Title" → produces .1080p.mp4, .720p.mp4, .480p.mp4
os.makedirs(os.path.dirname(output_prefix) or ".", exist_ok=True)

# Build filter_complex: split video into 3, scale each, burn ASS into each
# ASS path needs escaping for ffmpeg filter (replace : with \: and ' with \'')
ass_escaped = ass_path.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")

filter_complex = (
    f"[0:v]split=3[v1][v2][v3];"
    f"[v1]scale=1920:1080:force_original_aspect_ratio=decrease,"
    f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2,ass='{ass_escaped}'[out1080];"
    f"[v2]scale=1280:720:force_original_aspect_ratio=decrease,"
    f"pad=1280:720:(ow-iw)/2:(oh-ih)/2,ass='{ass_escaped}'[out720];"
    f"[v3]scale=854:480:force_original_aspect_ratio=decrease,"
    f"pad=854:480:(ow-iw)/2:(oh-ih)/2,ass='{ass_escaped}'[out480]"
)

out_1080 = f"{output_prefix}.1080p.mp4"
out_720 = f"{output_prefix}.720p.mp4"
out_480 = f"{output_prefix}.480p.mp4"

cmd = [
    "ffmpeg", "-y",
    "-i", video_path,
    "-filter_complex", filter_complex,
    "-map", "[out1080]", "-map", "a:0",
    "-c:v:0", "libx264", "-crf", "23", "-preset", "fast",
    "-c:a:0", "aac", "-b:a:0", "128k",
    "-map", "[out720]", "-map", "a:0",
    "-c:v:1", "libx264", "-crf", "23", "-preset", "fast",
    "-c:a:1", "aac", "-b:a:1", "128k",
    "-map", "[out480]", "-map", "a:0",
    "-c:v:2", "libx264", "-crf", "23", "-preset", "fast",
    "-c:a:2", "aac", "-b:a:2", "128k",
    out_1080, out_720, out_480
]

print(f"🔥 Burning subtitles into 3 resolutions...")
print(f"   1080p → {out_1080}")
print(f"   720p  → {out_720}")
print(f"   480p  → {out_480}")
subprocess.run(cmd, check=True)

for f in [out_1080, out_720, out_480]:
    if os.path.exists(f):
        print(f"✅ {os.path.basename(f)}: {os.path.getsize(f) / (1024*1024):.1f} MB")
