#!/usr/bin/env python3
"""
Validate subtitle text — remove non-target-language characters.
Usage: python3 validate_subs.py <srt_file> <language>
language: fa (Persian), en (English)

For Persian: detect and remove CJK, Cyrillic, Korean, Devanagari from subtitle text.
  NEVER remove English/Latin characters — tech terms (Claude, Hermes, AI, API,
  React, Telegram, etc.) are part of Persian subtitles and must be preserved.
  The user's rule is absolute: "فقط کلمات چینی و روسی حذف بشه اگر بود"
  and "من که بهت گفتم هیچ کلمه انگلیسی ای حذف نشه".
For English: detect and remove CJK, Cyrillic, Arabic/Persian, Korean from subtitle text.

Reports any suspicious characters found, auto-fixes them, and saves the cleaned SRT.
Run 2-3 times until zero issues are reported.
"""
import sys
import re


def validate_persian(text):
    ONLY removes: CJK (Chinese/Japanese), Cyrillic (Russian), Korean.
    NEVER removes English/Latin characters — tech terms like Claude, Hermes, AI,
    API, Telegram are part of Persian subtitles and must be preserved."""
    issues = []

    # CJK Chinese/Japanese
    cjk = re.findall(r'[\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF]', text)
    if cjk:
        issues.append(f"CJK chars: {''.join(cjk[:20])}")
        text = re.sub(r'[\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF]', '', text)

    # Cyrillic (Russian)
    cyr = re.findall(r'[\u0400-\u04FF]', text)
    if cyr:
        issues.append(f"Cyrillic: {''.join(cyr[:20])}")
        text = re.sub(r'[\u0400-\u04FF]', '', text)

    # Korean
    kor = re.findall(r'[\uAC00-\uD7AF]', text)
    if kor:
        issues.append(f"Korean: {''.join(kor[:20])}")
        text = re.sub(r'[\uAC00-\uD7AF]', '', text)

    # Devanagari (Hindi)
    dev = re.findall(r'[\u0900-\u097F]', text)
    if dev:
        issues.append(f"Devanagari: {''.join(dev[:20])}")
        text = re.sub(r'[\u0900-\u097F]', '', text)

    # Clean up double spaces
    text = re.sub(r'  +', ' ', text).strip()

    return text, issues


def validate_english(text):
    """Find and remove non-English characters from text."""
    issues = []

    cjk = re.findall(r'[\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF]', text)
    if cjk:
        issues.append(f"CJK chars: {''.join(cjk[:20])}")
        text = re.sub(r'[\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF]', '', text)

    cyr = re.findall(r'[\u0400-\u04FF]', text)
    if cyr:
        issues.append(f"Cyrillic: {''.join(cyr[:20])}")
        text = re.sub(r'[\u0400-\u04FF]', '', text)

    arabic = re.findall(r'[\u0600-\u06FF]', text)
    if arabic:
        issues.append(f"Arabic/Persian: {''.join(arabic[:20])}")
        text = re.sub(r'[\u0600-\u06FF]', '', text)

    kor = re.findall(r'[\uAC00-\uD7AF]', text)
    if kor:
        issues.append(f"Korean: {''.join(kor[:20])}")
        text = re.sub(r'[\uAC00-\uD7AF]', '', text)

    text = re.sub(r'  +', ' ', text).strip()

    return text, issues


def check_srt(filepath, language):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    blocks = content.strip().split('\n\n')
    total_issues = 0
    fixed_blocks = []

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            fixed_blocks.append(block)
            continue

        idx = lines[0]
        timing = lines[1]
        text = '\n'.join(lines[2:])

        if language == 'fa':
            cleaned, issues = validate_persian(text)
        else:
            cleaned, issues = validate_english(text)

        if issues:
            total_issues += len(issues)
            for issue in issues:
                print(f"[{idx}] {issue}")
                print(f"  Before: {text[:80]}")
                print(f"  After:  {cleaned[:80]}")

        fixed_blocks.append(f"{idx}\n{timing}\n{cleaned}")

    if total_issues == 0:
        print(f"OK: All {len(blocks)} entries clean for {language}")
    else:
        print(f"FIXED: {total_issues} issues in {len(blocks)} entries")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(fixed_blocks) + '\n')
        print(f"Saved to {filepath}")

    return total_issues


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python3 validate_subs.py <srt_file> <fa|en>")
        sys.exit(1)

    filepath = sys.argv[1]
    lang = sys.argv[2]
    check_srt(filepath, lang)
