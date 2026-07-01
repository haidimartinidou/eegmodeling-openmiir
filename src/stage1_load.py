"""Stage 1 (inspect): load one OpenMIIR subject and report raw structure.

This does NO preprocessing. It loads a raw .fif file, then prints the channel
layout, sampling rate, recording duration, and the event/trigger structure so we
can see exactly what we are working with before any filtering or epoching.

Usage:
    uv run python src/stage1_load.py            # defaults to subject P01
    uv run python src/stage1_load.py P04        # any subject id

Expected data location (not version-controlled):
    data/eeg/<SUBJECT>-raw.fif      e.g. data/eeg/P01-raw.fif
"""

from __future__ import annotations

import sys
from pathlib import Path

import mne

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "eeg"


def find_raw(subject: str) -> Path:
    """Locate <SUBJECT>-raw.fif, searching data/eeg/ then data/ as a fallback."""
    candidates = [
        DATA_DIR / f"{subject}-raw.fif",
        DATA_DIR.parent / f"{subject}-raw.fif",
    ]
    for path in candidates:
        if path.exists():
            return path
    searched = "\n  ".join(str(c) for c in candidates)
    raise SystemExit(
        f"Could not find {subject}-raw.fif. Looked in:\n  {searched}\n"
        f"Place the OpenMIIR raw file there and re-run."
    )


def main() -> None:
    subject = sys.argv[1] if len(sys.argv) > 1 else "P01"
    raw_path = find_raw(subject)

    # preload=False: read header/metadata without pulling ~700 MB into RAM yet.
    raw = mne.io.read_raw_fif(raw_path, preload=False, verbose="ERROR")
    info = raw.info

    print("=" * 70)
    print(f"OpenMIIR subject {subject}  —  {raw_path}")
    print("=" * 70)

    # --- recording-level facts ---
    n_chan = info["nchan"]
    sfreq = info["sfreq"]
    n_times = raw.n_times
    duration_s = n_times / sfreq
    print(f"sampling rate     : {sfreq:.1f} Hz")
    print(f"duration          : {duration_s:.1f} s  ({duration_s / 60:.1f} min)")
    print(f"samples           : {n_times:,}")
    print(f"total channels    : {n_chan}")
    print(f"highpass / lowpass: {info['highpass']:.2f} / {info['lowpass']:.1f} Hz")
    meas_date = info.get("meas_date")
    print(f"measurement date  : {meas_date}")

    # --- channel breakdown by type ---
    type_counts: dict[str, int] = {}
    for ch_type in raw.get_channel_types():
        type_counts[ch_type] = type_counts.get(ch_type, 0) + 1
    print("\nchannel types     :")
    for ch_type, count in sorted(type_counts.items()):
        print(f"  {ch_type:8s} {count}")

    # exclude=[] so channels marked 'bad' are still counted/listed by type
    # (otherwise a bad EEG channel would silently fall out of the EEG pick).
    eeg_picks = mne.pick_types(info, eeg=True, exclude=[])
    eeg_names = [info["ch_names"][i] for i in eeg_picks]
    print(f"\nEEG channels ({len(eeg_names)}):")
    print("  " + ", ".join(eeg_names))

    non_eeg = [
        info["ch_names"][i] for i in range(n_chan) if i not in set(eeg_picks)
    ]
    if non_eeg:
        print(f"\nnon-EEG channels ({len(non_eeg)}):")
        print("  " + ", ".join(non_eeg))

    print(f"\nbad channels      : {info['bads'] or 'none marked'}")

    # --- events / triggers ---
    stim_picks = mne.pick_types(info, stim=True)
    if len(stim_picks) == 0:
        print(
            "\nNo stim channel found via pick_types(stim=True); "
            "trigger info may be stored as annotations instead."
        )
        if raw.annotations:
            print(f"annotations       : {len(raw.annotations)} found")
            descriptions = sorted(set(raw.annotations.description))
            print("  descriptions: " + ", ".join(descriptions[:20]))
        return

    stim_name = info["ch_names"][stim_picks[0]]
    print(f"\nstim channel      : {stim_name}")
    events = mne.find_events(raw, stim_channel=stim_name, verbose="ERROR")
    print(f"events found      : {len(events)}")

    if len(events):
        codes, counts = _unique_counts(events[:, 2])
        print("event codes (id -> count):")
        for code, count in zip(codes, counts):
            print(f"  {code:>5d} : {count}")
        first_t = events[0, 0] / sfreq
        last_t = events[-1, 0] / sfreq
        print(f"first event at    : {first_t:.2f} s")
        print(f"last event at     : {last_t:.2f} s")

    print("=" * 70)
    print("Loaded successfully. No preprocessing applied.")


def _unique_counts(values):
    """Return sorted unique values and their counts (numpy-free for clarity)."""
    counts: dict[int, int] = {}
    for v in values:
        counts[int(v)] = counts.get(int(v), 0) + 1
    keys = sorted(counts)
    return keys, [counts[k] for k in keys]


if __name__ == "__main__":
    main()
