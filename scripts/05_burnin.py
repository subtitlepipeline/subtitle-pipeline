#!/usr/bin/env python3
"""Step 5: Burn-in ASS subtitles into original resolution, then scale to 720p and 480p."""
import sys, os, subprocess

video_path = sys.argv[1]   # input video (original resolution)
ass_path = sys.argv[2]     # ASS subtitle file
output_prefix = sys.argv[3]  # e.g. "output/Title" → Title.1080p.mp4, Title.720p.mp4, Title.480p.mp4

os.makedirs(os.path.dirname(output_prefix) or ".", exist_ok=True)

# Get original resolution
probe = subprocess.run([
    "ffprobe", "-v", "error",
    "-select_streams", "v:0",
    "-show_entries", "stream=width,height",
    "-of", "csv=p=0", video_path
], capture_output=True, text=True)
orig_w, orig_h = map(int, probe.stdout.strip().split(","))
print(f"📐 Original: {orig_w}x{orig_h}")

out_1080 = f"{output_prefix}.1080p.mp4"
out_720 = f"{output_prefix}.720p.mp4"
out_480 = f"{output_prefix}.480p.mp4"

# Step 1: Burn-in at original resolution
# If original is already 1080p, just burn. If it's different, scale to 1080p first.
target_w, target_h = 1920, 1080
if orig_w <= 1920 and orig_h <= 1080:
    # Original is smaller or equal — use original size, no upscale
    target_w, target_h = orig_w, orig_h

# Escape colons for ffmpeg filter
ass_esc = ass_path.replace(":", "\\:")
filter_burn = f"ass='{ass_esc}'"

print(f"🔥 Step 1: Burning subtitles at {target_w}x{target_h}...")
cmd1 = [
    "ffmpeg", "-y",
    "-i", video_path,
    "-vf", filter_burn,
    "-c:v", "libx264", "-crf", "23", "-preset", "fast",
    "-c:a", "aac", "-b:a", "128k",
    out_1080
]
subprocess.run(cmd1, check=True)
print(f"✅ 1080p done: {os.path.getsize(out_1080) / (1024*1024):.1f} MB")

# Step 2: Scale to 720p
print(f"\n🔽 Step 2: Scaling to 720p...")
cmd2 = [
    "ffmpeg", "-y",
    "-i", out_1080,
    "-vf", "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2",
    "-c:v", "libx264", "-crf", "23", "-preset", "fast",
    "-c:a", "aac", "-b:a", "128k",
    out_720
]
subprocess.run(cmd2, check=True)
print(f"✅ 720p done: {os.path.getsize(out_720) / (1024*1024):.1f} MB")

# Step 3: Scale to 480p
print(f"\n🔽 Step 3: Scaling to 480p...")
cmd3 = [
    "ffmpeg", "-y",
    "-i", out_1080,
    "-vf", "scale=854:480:force_original_aspect_ratio=decrease,pad=854:480:(ow-iw)/2:(oh-ih)/2",
    "-c:v", "libx264", "-crf", "23", "-preset", "fast",
    "-c:a", "aac", "-b:a", "128k",
    out_480
]
subprocess.run(cmd3, check=True)
print(f"✅ 480p done: {os.path.getsize(out_480) / (1024*1024):.1f} MB")

for f in [out_1080, out_720, out_480]:
    if os.path.exists(f):
        print(f"✅ {os.path.basename(f)}")
