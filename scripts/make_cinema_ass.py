#!/usr/bin/env python3
"""
Build cinema-style ASS subtitle with:
- 75% dark background (BF000000 = ~75% opacity, user-confirmed July 2026)
- BorderStyle=3 (opaque box)
- Outline=6 (padding inside box)
- MarginL/R=30 (padding from screen edges)
- RTL via \u202B prefix (redundant — fribidi auto-detects, but kept for clarity)
- Font size scales with resolution

Usage:
  python3 make_cinema_ass.py <input.srt> <output.ass> [fontsize]

Default fontsize=16 (for 480p). For 1080p use 24, for 720p use 18-20.
Rule of thumb: fontsize ≈ video_height * 0.022
"""
import pysubs2
import sys


def make_ass(srt_path, ass_path, fontsize=16):
    subs = pysubs2.load(srt_path, encoding="utf-8")

    style = pysubs2.SSAStyle()
    style.fontname = "Vazirmatn"
    style.fontsize = fontsize
    style.bold = True
    style.primarycolor = int("00FFFFFF", 16)   # white
    style.outlinecolor = int("00000000", 16)   # black
    style.backcolor = int("BF000000", 16)       # 75% dark (user-confirmed)
    style.borderstyle = 3                       # opaque box
    style.outline = 6                           # padding inside box
    style.shadow = 0
    style.alignment = 2                         # bottom-center
    style.marginl = 30
    style.marginr = 30
    style.marginv = 20
    style.encoding = -1  # auto-detect base direction — fixes Persian RTL punctuation
    subs.styles["Default"] = style

    for ev in subs.events:
        text = ev.text.strip()
        if text and not text.startswith("{"):
            ev.text = "\u202B" + text

    subs.save(ass_path, encoding="utf-8")
    print(f"✅ ASS saved: {ass_path} ({len(subs)} events, fontsize={fontsize})")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: make_cinema_ass.py <input.srt> <output.ass> [fontsize]")
        print("  fontsize: 16 for 480p (default), 24 for 1080p, 20 for 720p")
        sys.exit(1)
    fontsize = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    make_ass(sys.argv[1], sys.argv[2], fontsize)
