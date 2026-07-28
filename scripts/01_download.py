#!/usr/bin/env python3
"""Step 1: Download video with yt-dlp at 1080p."""
import sys, os, subprocess

url = sys.argv[1]
output_path = sys.argv[2]
os.makedirs(os.path.dirname(output_path), exist_ok=True)

cmd = [
    "yt-dlp",
    "--js-runtimes", "deno",
    "-f", "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
    "-o", output_path,
    "--merge-output-format", "mp4",
    "--no-playlist",
    "--extractor-args", "youtube:player_client=android,web",
    url
]
print(f"⬇️ Downloading: {url}")
subprocess.run(cmd, check=True)
print(f"✅ Saved: {output_path}")
