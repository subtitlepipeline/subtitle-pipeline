#!/usr/bin/env python3
"""Step 1: Download video with yt-dlp at 1080p.
Tries multiple player clients to bypass YouTube bot detection on cloud IPs.
"""
import sys, os, subprocess

url = sys.argv[1]
output_path = sys.argv[2]
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Try different player client combinations (in order of evasion effectiveness)
clients_to_try = [
    "youtube:player_client=ios,tv",
    "youtube:player_client=android_vr,ios",
    "youtube:player_client=mediaconnect,android",
    "youtube:player_client=web_safari,ios",
]

last_error = None
for i, client_args in enumerate(clients_to_try):
    cmd = [
        "yt-dlp",
        "--js-runtimes", "deno",
        "-f", "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
        "-o", output_path,
        "--merge-output-format", "mp4",
        "--no-playlist",
        "--no-warnings",
        "--extractor-args", client_args,
        url
    ]
    print(f"⬇️ Attempt {i+1}/{len(clients_to_try)} (client={client_args})")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)
        print(f"✅ Downloaded: {output_path}")
        print(result.stdout[-500:] if result.stdout else "")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        last_error = e
        err = e.stderr or ""
        print(f"❌ Failed: {err[-200:]}")
        if os.path.exists(output_path):
            os.remove(output_path)
        # If it's not a bot detection error, don't retry with other clients
        if "Sign in to confirm" not in err and "bot" not in err.lower():
            raise
    except subprocess.TimeoutExpired:
        print("⏰ Timed out")
        if os.path.exists(output_path):
            os.remove(output_path)

print(f"❌ All attempts failed. Last error: {last_error}")
sys.exit(1)
