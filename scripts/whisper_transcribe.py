#!/usr/bin/env python3
"""Whisper large-v3 transcription with fine-grained, subtitle-ready segments.

Segmentation does NOT rely on model punctuation (Whisper often emits none for
Persian). Instead a segment is closed when ANY of these is true:
  * the word ends with sentence/clause punctuation
  * the pause before the next word exceeds GAP_SEC
  * the segment reached MAX_WORDS words
  * the segment reached MAX_CHARS characters
  * the segment reached MAX_DUR seconds
"""

import json
import re
import sys

GAP_SEC = 0.35     # silence between words that forces a cut
MAX_WORDS = 7      # hard cap on words per segment
MAX_CHARS = 42     # hard cap on characters per segment (one subtitle line)
MAX_DUR = 3.5      # hard cap on segment duration in seconds
MIN_DUR = 0.20     # never emit a segment shorter than this (merged forward)

PUNCT_END = re.compile(r'[.!?؟،؛:]$')


def build_segments(words, gap_sec=GAP_SEC, max_words=MAX_WORDS,
                   max_chars=MAX_CHARS, max_dur=MAX_DUR, min_dur=MIN_DUR):
    """words: list of {'start': float, 'end': float, 'word': str} -> segments."""
    segments = []
    cur = []

    def flush():
        if not cur:
            return
        text = ' '.join(w['word'] for w in cur)
        text = re.sub(r'\s+([،.؟!؛:])', r'\1', text).strip()
        if not text:
            cur.clear()
            return
        segments.append({
            'start': round(cur[0]['start'], 3),
            'end': round(cur[-1]['end'], 3),
            'text': text,
        })
        cur.clear()

    for i, w in enumerate(words):
        if not w['word']:
            continue
        cur.append(w)

        cur_text = ' '.join(x['word'] for x in cur)
        dur = cur[-1]['end'] - cur[0]['start']
        nxt = words[i + 1] if i + 1 < len(words) else None
        gap = (nxt['start'] - w['end']) if nxt else float('inf')

        hit_punct = bool(PUNCT_END.search(w['word']))
        hit_gap = gap >= gap_sec
        hit_words = len(cur) >= max_words
        hit_chars = len(cur_text) >= max_chars
        hit_dur = dur >= max_dur

        if hit_punct or hit_gap or hit_words or hit_chars or hit_dur:
            # avoid emitting a sliver unless punctuation/gap justifies it
            if dur < min_dur and not (hit_punct or hit_gap) and nxt is not None:
                continue
            flush()

    flush()

    # merge any leftover ultra-short segment into its neighbour
    merged = []
    for seg in segments:
        if merged and (seg['end'] - seg['start']) < min_dur:
            prev = merged[-1]
            prev['text'] = f"{prev['text']} {seg['text']}".strip()
            prev['end'] = seg['end']
        else:
            merged.append(seg)
    return merged


def collect_words(whisper_segments):
    """Flatten faster-whisper segments into a flat word list."""
    words = []
    for seg in whisper_segments:
        if seg.words:
            for w in seg.words:
                token = w.word.strip()
                if token:
                    words.append({'start': w.start, 'end': w.end, 'word': token})
        else:  # fallback: even split across the segment
            toks = seg.text.strip().split()
            if not toks:
                continue
            per = (seg.end - seg.start) / len(toks)
            for i, t in enumerate(toks):
                words.append({
                    'start': seg.start + i * per,
                    'end': seg.start + (i + 1) * per,
                    'word': t,
                })
    return words


def main():
    from faster_whisper import WhisperModel

    audio = sys.argv[1] if len(sys.argv) > 1 else 'output/video.mp4'
    out_path = sys.argv[2] if len(sys.argv) > 2 else 'output/transcript.json'

    print('Loading Whisper large-v3...', flush=True)
    model = WhisperModel('large-v3', device='cpu', compute_type='int8',
                         cpu_threads=4, num_workers=1)

    print('Transcribing with word timestamps...', flush=True)
    seg_iter, info = model.transcribe(
        audio,
        beam_size=5,
        vad_filter=False,
        word_timestamps=True,
        condition_on_previous_text=False,
    )

    words = collect_words(seg_iter)
    print(f'Collected {len(words)} words', flush=True)

    segments = build_segments(words)

    payload = {
        'language': info.language,
        'language_probability': round(info.language_probability, 3),
        'duration': round(info.duration, 2),
        'segments': segments,
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f'Done: {len(segments)} segments, language={info.language}', flush=True)
    for i, s in enumerate(segments[:40], 1):
        m, sec = divmod(int(s['start']), 60)
        print(f'{i:3d}. [{m:02d}:{sec:02d}] {s["text"]}')


if __name__ == '__main__':
    main()
