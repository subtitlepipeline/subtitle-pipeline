#!/usr/bin/env python3
"""
Step 6: Upload final video + subtitles to GitHub Release.
Outputs download URL to result.json and stdout.
"""
import sys, os, json, subprocess, uuid, datetime

# Find output files
final_video = sys.argv[1] if len(sys.argv) > 1 else "output/final.mp4"
srt_file = sys.argv[2] if len(sys.argv) > 2 else "output/subtitle.srt"
ass_file = sys.argv[3] if len(sys.argv) > 3 else "output/subtitle.ass"

gh_token = os.environ["GH_TOKEN"]

# Generate unique tag
tag = f"v{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
print(f"🏷️ Creating release: {tag}")

# Create release
result = subprocess.run([
    "gh", "release", "create", tag,
    "--title", f"Subtitle {tag}",
    "--notes", "Auto-generated subtitle video",
    "--repo", "subtitlepipeline/subtitle-pipeline"
], capture_output=True, text=True, env={**os.environ, "GH_TOKEN": gh_token})

if result.returncode != 0:
    print(f"❌ Failed to create release: {result.stderr}")
    sys.exit(1)

print(f"✅ Release created")

# Upload assets
assets = [final_video, srt_file, ass_file]
download_urls = []

for asset in assets:
    if not os.path.exists(asset):
        print(f"⚠️ Skipping {asset} (not found)")
        continue
    print(f"📤 Uploading {asset}...")
    r = subprocess.run([
        "gh", "release", "upload", tag, asset,
        "--repo", "subtitlepipeline/subtitle-pipeline"
    ], capture_output=True, text=True, env={**os.environ, "GH_TOKEN": gh_token})
    if r.returncode != 0:
        print(f"❌ Upload failed: {r.stderr}")
    else:
        print(f"✅ Uploaded {asset}")

# Get download URLs
r = subprocess.run([
    "gh", "release", "view", tag, "--json", "assets",
    "--repo", "subtitlepipeline/subtitle-pipeline"
], capture_output=True, text=True, env={**os.environ, "GH_TOKEN": gh_token})

release_data = json.loads(r.stdout)
for asset in release_data.get("assets", []):
    download_urls.append({
        "name": asset["name"],
        "url": asset["url"]
    })

# Save result
result_data = {
    "tag": tag,
    "download_urls": download_urls,
    "video_url": download_urls[0]["url"] if download_urls else None
}

output_json = "output/result.json"
os.makedirs("output", exist_ok=True)
with open(output_json, "w") as f:
    json.dump(result_data, f, indent=2)

# Output for workflow
if download_urls:
    print(f"::set-output name=download_url::{download_urls[0]['url']}")

print(f"\n✅ Done! Download URLs:")
for d in download_urls:
    print(f"  {d['name']}: {d['url']}")
print(f"\n📁 Result saved: {output_json}")
