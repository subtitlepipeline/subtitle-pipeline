#!/usr/bin/env python3
"""
Step 6: Upload final videos + subtitles to GitHub Release.
Uses the original video title as filename.
Outputs result.json with download URLs.
"""
import sys, os, json, subprocess, datetime

# Args: <prefix> <srt> <ass> <title>
# prefix = output/video_title (without extension)
prefix = sys.argv[1] if len(sys.argv) > 1 else "output/final"
srt_file = sys.argv[2] if len(sys.argv) > 2 else "output/subtitle.srt"
ass_file = sys.argv[3] if len(sys.argv) > 3 else "output/subtitle.ass"
title = sys.argv[4] if len(sys.argv) > 4 else "subtitle"

# Fix encoding: if title looks like mojibake, try to fix it
try:
    # If the title has replacement chars or looks wrong, try reading from title.txt
    if '\\x' in repr(title) or '?' in title:
        with open("output/title.txt", encoding="utf-8") as f:
            title = f.read().strip()
except:
    pass

gh_token = os.environ["GH_TOKEN"]

# Sanitize title for filename (keep Unicode letters incl Arabic/Persian, remove special chars)
import re
# re.UNICODE flag makes \w match Arabic/Persian letters too
safe_title = re.sub(r'[^\w\s.-]', '', title, flags=re.UNICODE).strip()
# Replace multiple spaces/underscores with single underscore, remove leading/trailing dots
safe_title = re.sub(r'[\s]+', '_', safe_title)
safe_title = re.sub(r'[._]{2,}', '_', safe_title)
safe_title = safe_title.strip('._')
if not safe_title:
    safe_title = "subtitle"

# Generate unique tag
tag = f"v{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
print(f"🏷️ Creating release: {tag}")

# Create release
result = subprocess.run([
    "gh", "release", "create", tag,
    "--title", safe_title,
    "--notes", f"Auto-generated subtitle: {title}",
    "--repo", "subtitlepipeline/subtitle-pipeline"
], capture_output=True, text=True, env={**os.environ, "GH_TOKEN": gh_token})

if result.returncode != 0:
    print(f"❌ Failed to create release: {result.stderr}")
    sys.exit(1)

print(f"✅ Release created: {safe_title}")

# Collect assets: 3 video resolutions + srt + ass
assets = []
for res in ["1080p", "720p", "480p"]:
    f = f"{prefix}.{res}.mp4"
    if os.path.exists(f):
        # Rename to use the real title
        dest = f"{safe_title}.{res}.mp4"
        os.rename(f, dest)
        assets.append(dest)

if os.path.exists(srt_file):
    dest_srt = f"{safe_title}.srt"
    os.rename(srt_file, dest_srt)
    assets.append(dest_srt)

if os.path.exists(ass_file):
    dest_ass = f"{safe_title}.ass"
    os.rename(ass_file, dest_ass)
    assets.append(dest_ass)

# Upload assets
download_urls = []
for asset in assets:
    if not os.path.exists(asset):
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
    "title": title,
    "download_urls": download_urls,
}

output_json = "output/result.json"
os.makedirs("output", exist_ok=True)
with open(output_json, "w") as f:
    json.dump(result_data, f, indent=2)

print(f"\n✅ Done! Download URLs:")
for d in download_urls:
    print(f"  {d['name']}: {d['url']}")
print(f"\n📁 Result saved: {output_json}")
