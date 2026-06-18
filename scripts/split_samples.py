#!/usr/bin/env python3
"""Split a multi-note WAV file into individual per-note WAV files.

Usage:
    python split_samples.py <input.wav>

Output structure:
    output/set_1/C4.wav
    output/set_2/C4.wav   (if the same note appears more than once)
"""

import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf

# ── Tunable constants ──────────────────────────────────────────────────────────
SILENCE_THRESHOLD_DB   = -30   # dB below peak; used for edge-trimming each note    -50
MIN_NOTE_SPACING_S     = 0.20  # minimum time (s) between two onsets    0.5
MIN_SEGMENT_DURATION_S = 0.20  # discard segments shorter than this     0.3
PITCH_CONFIDENCE_MIN   = 0.35  # pyin voiced-probability cutoff (0–1)   0.75
FRAME_HOP_LENGTH       = 512   # hop size shared by onset detector and pyin 512
FRAME_LENGTH           = 2048  # frame size for RMS (used in trim)      2048
OUTPUT_DIR             = "output"
# 0.0 = only accept notes clearly closer to a natural (strict, half-semitone boundary)
# 1.0 = accept nearest note regardless of whether it is natural or a semitone
# intermediate values scale the acceptance radius linearly in MIDI units:
#   radius = 0.5 + SEMITONE_TOLERANCE * 0.5  (0.5 is the half-bandwidth of a semitone)
SEMITONE_TOLERANCE     = 0.9

# ── Note tables ────────────────────────────────────────────────────────────────
_NATURAL_PCS = {0, 2, 4, 5, 7, 9, 11}   # C D E F G A B (pitch classes mod 12)
_PC_TO_NAME  = {0: "C", 2: "D", 4: "E", 5: "F", 7: "G", 9: "A", 11: "B"}


def freq_to_note(freq_hz: float) -> Optional[str]:
    """Return e.g. 'C4' snapped to the nearest natural note, or None if rejected.

    Rejection is controlled by SEMITONE_TOLERANCE:
      0.0 → discard if the pitch is closer to a semitone than to a natural note
      1.0 → accept even exact semitone pitches (snapped to nearest natural)
    """
    if not (freq_hz > 0):
        return None

    midi_float = 69.0 + 12.0 * np.log2(freq_hz / 440.0)
    center = round(midi_float)

    min_nat_dist  = float("inf")
    best_nat_midi = None

    for m in range(center - 2, center + 3):
        dist = abs(midi_float - m)
        pc   = m % 12
        if pc in _NATURAL_PCS and dist < min_nat_dist:
            min_nat_dist  = dist
            best_nat_midi = m

    acceptance_radius = 0.5 + SEMITONE_TOLERANCE * 0.5
    if best_nat_midi is None or min_nat_dist >= acceptance_radius:
        return None

    pc     = best_nat_midi % 12
    octave = best_nat_midi // 12 - 1   # MIDI 60 → C4
    return f"{_PC_TO_NAME[pc]}{octave}"


def find_segments(y: np.ndarray, sr: int) -> list:
    """Return [(start_sample, end_sample), ...] for each note.

    Uses onset detection so that slowly-decaying notes whose amplitude never
    reaches true silence are still split correctly.
    """
    # Detect onset frames
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr,
        hop_length=FRAME_HOP_LENGTH,
        backtrack=True,   # snap each onset back to the nearest preceding dip
        units="frames",
    )

    if len(onset_frames) == 0:
        return []

    # Convert to samples and enforce minimum spacing between onsets
    min_gap_samples = int(MIN_NOTE_SPACING_S * sr)
    onset_samples: list[int] = []
    for f in onset_frames:
        s = librosa.frames_to_samples(int(f), hop_length=FRAME_HOP_LENGTH)
        if onset_samples and s - onset_samples[-1] < min_gap_samples:
            continue   # too close to the previous onset — skip
        onset_samples.append(s)

    # Build (start, end) pairs: each note runs until the next onset
    segments: list = []
    for i, start in enumerate(onset_samples):
        end = onset_samples[i + 1] if i + 1 < len(onset_samples) else len(y)
        if (end - start) / sr >= MIN_SEGMENT_DURATION_S:
            segments.append((start, end))

    return segments


def detect_pitch(y: np.ndarray, sr: int) -> Optional[float]:
    """Return median voiced frequency (Hz) for the segment, or None."""
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        hop_length=FRAME_HOP_LENGTH,
    )

    mask = voiced_flag & (voiced_probs >= PITCH_CONFIDENCE_MIN) & np.isfinite(f0)
    valid = f0[mask]
    return float(np.median(valid)) if len(valid) > 0 else None


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input.wav>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"Error: file not found: {input_path}")
        sys.exit(1)

    print(f"Loading {input_path} …")
    y, sr = librosa.load(str(input_path), sr=None, mono=True)
    print(f"  {sr} Hz, {len(y)/sr:.2f}s")

    print("Segmenting by silence …")
    segments = find_segments(y, sr)
    print(f"  {len(segments)} segment(s) found")

    note_chunks: dict = defaultdict(list)

    for idx, (start, end) in enumerate(segments, 1):
        chunk = y[start:end]
        # Trim any residual silence from the edges of the chunk
        chunk, _ = librosa.effects.trim(chunk, top_db=-SILENCE_THRESHOLD_DB)

        freq = detect_pitch(chunk, sr)
        if freq is None:
            print(f"  Segment {idx}: pitch detection failed — skipping")
            continue

        note = freq_to_note(freq)
        if note is None:
            print(f"  Segment {idx}: {freq:.1f} Hz too close to a semitone — skipping (try raising SEMITONE_TOLERANCE)")
            continue

        print(f"  Segment {idx}: {freq:.1f} Hz → {note}")
        note_chunks[note].append(chunk)

    if not note_chunks:
        print("No valid note segments found.")
        sys.exit(1)

    max_sets = max(len(v) for v in note_chunks.values())

    for set_idx in range(1, max_sets + 1):
        set_dir = Path(OUTPUT_DIR) / f"set_{set_idx}"
        set_dir.mkdir(parents=True, exist_ok=True)

        for note, chunks in sorted(note_chunks.items()):
            if set_idx <= len(chunks):
                out = set_dir / f"{note}.wav"
                sf.write(str(out), chunks[set_idx - 1], sr)
                print(f"  Wrote {out}")

    print("Done.")


if __name__ == "__main__":
    main()
