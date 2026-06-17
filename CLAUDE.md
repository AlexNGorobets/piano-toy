# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Arduino-based piano toy. The Python tooling in `scripts/` pre-processes raw multi-note WAV recordings into individual per-note sample files for use on the device.

Audio samples live at the repo root:
- `167164__ultradust__cedar-kalimba-in-c.wav` — cedar kalimba in C (source: freesound.org)
- `490321__wash__goldon-grand-piano-g-major-morphagene-reel-toy-piano.wav` — Goldon grand piano G major

## Environment setup

```bash
cd scripts
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the splitter

```bash
python scripts/split_samples.py <input.wav>
```

Output is written to `output/set_N/<note>.wav` (e.g. `output/set_1/C4.wav`).

## Architecture: scripts/split_samples.py

**Pipeline** (in order):

1. **Load** — `librosa.load` with original sample rate, forced mono
2. **Onset detection** — `librosa.onset.onset_detect(backtrack=True)` finds note attack transients; silence-gap detection is avoided because kalimba/piano decays don't reach true silence between notes
3. **Minimum spacing filter** — onsets closer than `MIN_NOTE_SPACING_S` are merged to suppress sub-note micro-onsets
4. **Segment extraction** — each note = [onset_i, onset_{i+1}]; last note runs to end of file
5. **Edge trim** — `librosa.effects.trim` removes residual silence at each segment's boundaries
6. **Pitch detection** — `librosa.pyin` (probabilistic YIN); median of confident voiced frames gives a stable Hz estimate
7. **Note classification** — Hz → fractional MIDI → compare distance to nearest natural note (C D E F G A B) vs nearest semitone; if closer to semitone the segment is discarded
8. **Set assignment** — segments grouped by note name; N occurrences of the same note → `set_1 … set_N`

**Key constants** (top of file, all tunable):

| Constant | Default | Purpose |
|---|---|---|
| `SILENCE_THRESHOLD_DB` | -40 | Edge-trim threshold |
| `MIN_NOTE_SPACING_S` | 0.80 | Minimum gap between onsets |
| `MIN_SEGMENT_DURATION_S` | 0.30 | Discard noise/bleed shorter than this |
| `PITCH_CONFIDENCE_MIN` | 0.75 | pyin voiced-probability cutoff |
