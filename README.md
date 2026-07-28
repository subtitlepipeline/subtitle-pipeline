# Subtitle Pipeline

Automated subtitle pipeline running on GitHub Actions:

1. **yt-dlp** → Download video (1080p)
2. **Whisper large-v3** → Transcription (best quality)
3. **DeepSeek via LiteLLM** → Translate to Persian
4. **validate_subs.py** → Validate 3x (strip CJK/Cyrillic, keep tech terms)
5. **cinema ASS** → Cinema-style subtitles (Vazirmatn, RTL)
6. **ffmpeg burn-in** → Hardcode subtitles into video
7. **GitHub Release** → Upload + download link

## Usage

Trigger via GitHub Actions (workflow_dispatch) with a YouTube URL.

## Required Secrets

- `NVIDIA_API_KEYS`: Comma-separated NVIDIA API keys for LiteLLM

## Structure

```
scripts/
  01_download.py      # yt-dlp download
  02_whisper.py       # Whisper large-v3 transcription
  03_translate.py     # DeepSeek translation via LiteLLM proxy
  04_build_srt.py     # Build SRT from transcript + translations
  05_burnin.py        # ffmpeg burn-in
  06_upload.py        # Upload to GitHub Release
  validate_subs.py    # Validate subtitle text (3x)
  make_cinema_ass.py  # Cinema-style ASS builder
.github/workflows/
  subtitle.yml        # Main workflow
```
