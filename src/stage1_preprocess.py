"""Stage 1 (preprocess): clean and epoch one OpenMIIR subject for TRF analysis.

Pipeline (see README / project plan):
  1. load raw, set biosemi64 montage, confirm EOG channels
  2. extract events @ 512 Hz (before any decimation)
  3. band-pass 1-30 Hz, FIR, zero-phase  (broadband; the 1-8 Hz TRF filter is
     applied later, at the TRF step, so alpha is preserved here)
  4. average reference (excludes bads automatically)
  5. ICA x2 -- apply Stober's provided solution AND fit a fresh one, then print
     a comparison of the flagged ocular components BEFORE saving anything
  6. interpolate the pre-marked bad channels (P8, P10, T8)
  7. epoch all 240 stimulus trials, decimate 512 -> 64 Hz, attach metadata
  8. write a preprocessing report (PSD before/after, ICA topographies, interp
     list) and the cleaned epochs, then STOP for review

Reference is applied BEFORE ICA (not after) because the provided ICA carries an
average-reference projection and was fit on the 61 good channels; matching that
keeps both ICA solutions comparable.

Run:
    uv run python src/stage1_preprocess.py            # subject P01, provided ICA
    uv run python src/stage1_preprocess.py P01 --ica fresh
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write figures to disk, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mne
from mne.preprocessing import ICA
from mne.preprocessing.ica import ICA as _ICAClass

PROJECT = Path(__file__).resolve().parent.parent
EEG_DIR = PROJECT / "data" / "eeg"
ICA_DIR = PROJECT / "data" / "ica"
DERIV = PROJECT / "derivatives"
REPORT = DERIV / "report"

# OpenMIIR trial design: stimulus codes = stimulus_id * 10 + condition
STIMULUS_IDS = [1, 2, 3, 4, 11, 12, 13, 14, 21, 22, 23, 24]
CONDITIONS = {1: "perception", 2: "cued imag", 3: "imag fix cross", 4: "imagination"}
BAD_CHANNELS = ["P8", "P10", "T8"]  # pre-marked in the raw file

# two-tier filter: broadband stored here, 1-8 Hz TRF band applied downstream
L_FREQ, H_FREQ = 1.0, 30.0
TARGET_SFREQ = 64.0  # 512 / 8 -> exact integer decimation, no resample jitter


# --------------------------------------------------------------------------- #
# Step 1-2: load + events
# --------------------------------------------------------------------------- #
def load_raw(subject: str) -> mne.io.Raw:
    path = EEG_DIR / f"{subject}-raw.fif"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run stage1_load.py first / place the file.")
    raw = mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
    montage = mne.channels.make_standard_montage("biosemi64")
    raw.set_montage(montage, on_missing="ignore", verbose="ERROR")
    n_eog = len(mne.pick_types(raw.info, eog=True))
    print(f"[load] {subject}: {raw.info['nchan']} ch, {raw.info['sfreq']:.0f} Hz, "
          f"{raw.n_times / raw.info['sfreq'] / 60:.1f} min, {n_eog} EOG, "
          f"bads={raw.info['bads']}")
    return raw


def get_stimulus_events(raw: mne.io.Raw) -> tuple[np.ndarray, dict]:
    events = mne.find_events(raw, stim_channel="STI 014", verbose="ERROR")
    event_id = {
        f"stim{sid}/cond{c}": sid * 10 + c
        for sid in STIMULUS_IDS
        for c in CONDITIONS
    }
    valid = set(event_id.values())
    n_stim = int(np.isin(events[:, 2], list(valid)).sum())
    print(f"[events] {len(events)} total, {n_stim} stimulus trials "
          f"(expect 240 = 12x4x5)")
    return events, event_id


# --------------------------------------------------------------------------- #
# Step 3-4: filter + reference
# --------------------------------------------------------------------------- #
def bandpass(raw: mne.io.Raw) -> mne.io.Raw:
    raw.filter(L_FREQ, H_FREQ, method="fir", phase="zero",
               fir_design="firwin", verbose="ERROR")
    print(f"[filter] band-pass {L_FREQ}-{H_FREQ} Hz, FIR zero-phase")
    return raw


def average_reference(raw: mne.io.Raw) -> mne.io.Raw:
    # projection=False applies the reference immediately; bads are excluded from
    # the average by MNE, so P8/P10/T8 do not contaminate it.
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    print("[reference] average (bads excluded from the average)")
    return raw


# --------------------------------------------------------------------------- #
# Step 5: ICA -- provided (legacy) + fresh, then compare
# --------------------------------------------------------------------------- #
def load_provided_ica(subject: str) -> ICA:
    """Load Stober's 2015 ICA. Legacy files encode 'use 100% of components' as
    n_components=1.0, which current MNE rejects; map it to None (= all)."""
    path = ICA_DIR / f"{subject}-100p_64c-ica.fif"
    if not path.exists():
        raise SystemExit(f"Missing provided ICA {path}.")
    orig_init = _ICAClass.__init__

    def patched_init(self, *a, **k):
        if k.get("n_components") == 1.0:
            k["n_components"] = None
        return orig_init(self, *a, **k)

    _ICAClass.__init__ = patched_init
    try:
        ica = mne.preprocessing.read_ica(path, verbose="ERROR")
    finally:
        _ICAClass.__init__ = orig_init
    ica.exclude = list(ica.exclude)  # Stober's manual selection
    return ica


def fit_fresh_ica(raw: mne.io.Raw) -> ICA:
    # mirror the provided solution: fit on the good EEG channels only.
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    ica = ICA(n_components=0.99, method="fastica", max_iter="auto",
              random_state=97, verbose="ERROR")
    ica.fit(raw, picks=picks, verbose="ERROR")
    return ica


def _max_eog_score(scores) -> np.ndarray:
    """Collapse per-EOG-channel score arrays to one max|r| per component."""
    arr = np.atleast_2d(np.array(scores))
    return np.max(np.abs(arr), axis=0)


def compare_icas(raw: mne.io.Raw, provided: ICA, fresh: ICA) -> dict:
    """Print provided-vs-fresh comparison; return EOG indices for each."""
    prov_eog, prov_scores = provided.find_bads_eog(raw, verbose="ERROR")
    fresh_eog, fresh_scores = fresh.find_bads_eog(raw, verbose="ERROR")
    fresh.exclude = sorted(fresh_eog)

    prov_max = _max_eog_score(prov_scores)
    fresh_max = _max_eog_score(fresh_scores)

    print("\n" + "=" * 68)
    print("ICA COMPARISON  (provided vs fresh)  -- nothing saved yet")
    print("=" * 68)
    print(f"{'':22s}{'provided (Stober)':>22s}{'fresh':>20s}")
    print(f"{'n components':22s}{provided.n_components_:>22d}{fresh.n_components_:>20d}")
    print(f"{'method':22s}{'fastica':>22s}{'fastica':>20s}")
    print(f"{'manual exclude':22s}{str(provided.exclude):>22s}{'-':>20s}")
    print(f"{'auto EOG-detected':22s}{str(sorted(prov_eog)):>22s}{str(sorted(fresh_eog)):>20s}")

    overlap = sorted(set(provided.exclude) & set(prov_eog))
    print(f"\nprovided: Stober's manual picks {provided.exclude} vs auto-EOG "
          f"{sorted(prov_eog)} -> overlap {overlap}")
    print("\nmax |EOG correlation| of the components each method removes:")
    print("  provided (manual excludes):")
    for c in provided.exclude:
        print(f"    comp {c:2d}: |r|={prov_max[c]:.3f}")
    print("  fresh (auto EOG excludes):")
    for c in sorted(fresh_eog):
        print(f"    comp {c:2d}: |r|={fresh_max[c]:.3f}")
    print("\nNote: component indices are not comparable across decompositions; "
          "compare by EOG correlation and topography (see report figures).")
    print("=" * 68 + "\n")

    return {
        "provided_exclude": list(provided.exclude),
        "provided_auto_eog": sorted(int(i) for i in prov_eog),
        "fresh_exclude": sorted(int(i) for i in fresh_eog),
        "prov_scores": prov_scores,
        "fresh_scores": fresh_scores,
    }


# --------------------------------------------------------------------------- #
# Step 6: interpolate bads
# --------------------------------------------------------------------------- #
def interpolate_bads(raw: mne.io.Raw) -> mne.io.Raw:
    raw.info["bads"] = list(BAD_CHANNELS)
    raw.interpolate_bads(reset_bads=True, verbose="ERROR")
    print(f"[interpolate] spherical-spline: {BAD_CHANNELS} -> reset to good")
    return raw


# --------------------------------------------------------------------------- #
# Step 7: epoch
# --------------------------------------------------------------------------- #
def make_epochs(raw, events, event_id):
    decim = int(round(raw.info["sfreq"] / TARGET_SFREQ))
    assert raw.info["sfreq"] / decim == TARGET_SFREQ, "non-integer decimation"

    # data-driven tmax: largest window that cannot overlap the next trial
    stim_codes = set(event_id.values())
    stim_samps = np.sort([s for s, _, c in events if c in stim_codes])
    min_gap_s = np.min(np.diff(stim_samps)) / raw.info["sfreq"]
    tmin = -0.5
    tmax = float(min(np.floor((min_gap_s - 0.5) * 10) / 10, 18.0))
    print(f"[epoch] min inter-trial gap {min_gap_s:.2f}s -> window "
          f"[{tmin}, {tmax}]s, decim {decim} -> {TARGET_SFREQ:.0f} Hz")

    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                        baseline=None, decim=decim, preload=True,
                        on_missing="ignore", verbose="ERROR")

    # attach trial metadata; audio-onset (1000) time per trial deferred to Stage 2
    codes = epochs.events[:, 2]
    epochs.metadata = pd.DataFrame({
        "code": codes,
        "stimulus_id": codes // 10,
        "condition": codes % 10,
        "condition_name": [CONDITIONS[c % 10] for c in codes],
    })
    print(f"[epoch] {len(epochs)} epochs "
          f"({(epochs.metadata.condition == 1).sum()} perception)")
    return epochs


# --------------------------------------------------------------------------- #
# Step 8: report + save
# --------------------------------------------------------------------------- #
def write_report(subject, raw_before, raw_clean, provided, fresh, cmp, epochs):
    REPORT.mkdir(parents=True, exist_ok=True)
    paths = []

    # PSD before vs after
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    raw_before.compute_psd(fmax=60, verbose="ERROR").plot(
        axes=axes[0], show=False, average=True)
    axes[0].set_title(f"{subject} PSD before (raw)")
    raw_clean.compute_psd(fmax=60, verbose="ERROR").plot(
        axes=axes[1], show=False, average=True)
    axes[1].set_title(f"{subject} PSD after (filtered+ICA+ref+interp)")
    p = REPORT / f"{subject}_psd_before_after.png"
    fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig); paths.append(p)

    # excluded ICA components -- provided + fresh
    for tag, ica, excl in [("provided", provided, provided.exclude),
                           ("fresh", fresh, fresh.exclude)]:
        if excl:
            fig = ica.plot_components(picks=excl, show=False)
            fig = fig[0] if isinstance(fig, list) else fig
            fig.suptitle(f"{subject} ICA excluded ({tag}): {list(excl)}")
            p = REPORT / f"{subject}_ica_excluded_{tag}.png"
            fig.savefig(p, dpi=110); plt.close(fig); paths.append(p)

    # perception-subset sanity evoked (onset-locked butterfly)
    perc = epochs["cond1"]
    if len(perc):
        evk = perc.average()
        fig = evk.plot(spatial_colors=True, show=False)
        fig.suptitle(f"{subject} perception onset-locked evoked (n={len(perc)})")
        p = REPORT / f"{subject}_perception_evoked.png"
        fig.savefig(p, dpi=110); plt.close(fig); paths.append(p)

    # text summary
    summary = REPORT / f"{subject}_preprocess_summary.txt"
    with open(summary, "w") as f:
        f.write(f"OpenMIIR Stage-1 preprocessing summary -- {subject}\n")
        f.write("=" * 60 + "\n")
        f.write(f"filter            : {L_FREQ}-{H_FREQ} Hz FIR zero-phase\n")
        f.write(f"reference         : average (before ICA)\n")
        f.write(f"sampling rate     : {TARGET_SFREQ:.0f} Hz (decimated from 512)\n")
        f.write(f"interpolated bads : {BAD_CHANNELS}\n")
        f.write(f"epochs            : {len(epochs)} (all conditions)\n")
        for c, name in CONDITIONS.items():
            n = int((epochs.metadata.condition == c).sum())
            f.write(f"  cond {c} {name:16s}: {n}\n")
        f.write("\nICA comparison:\n")
        f.write(f"  provided manual exclude : {cmp['provided_exclude']}\n")
        f.write(f"  provided auto EOG       : {cmp['provided_auto_eog']}\n")
        f.write(f"  fresh    auto EOG       : {cmp['fresh_exclude']}\n")
    paths.append(summary)

    print("[report] wrote:")
    for p in paths:
        print(f"  {p.relative_to(PROJECT)}")


def save_epochs(subject, epochs, events):
    DERIV.mkdir(parents=True, exist_ok=True)
    epo_path = DERIV / f"{subject}-epo.fif"
    epochs.save(epo_path, overwrite=True, verbose="ERROR")
    np.save(DERIV / f"{subject}-events.npy", events)
    print(f"[save] {epo_path.relative_to(PROJECT)}  "
          f"({len(epochs)} epochs, {epochs.info['sfreq']:.0f} Hz)")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subject", nargs="?", default="P01")
    ap.add_argument("--ica", choices=["provided", "fresh"], default="provided",
                    help="which ICA to apply to the SAVED output")
    args = ap.parse_args()
    subject = args.subject

    raw = load_raw(subject)
    raw_before = raw.copy()  # for the before-PSD
    events, event_id = get_stimulus_events(raw)

    bandpass(raw)
    average_reference(raw)

    provided = load_provided_ica(subject)
    fresh = fit_fresh_ica(raw)
    cmp = compare_icas(raw, provided, fresh)

    canonical = provided if args.ica == "provided" else fresh
    print(f"[ica] applying '{args.ica}' ICA to saved output "
          f"(exclude={canonical.exclude})")
    canonical.apply(raw, verbose="ERROR")

    interpolate_bads(raw)
    raw_clean = raw

    epochs = make_epochs(raw_clean, events, event_id)
    write_report(subject, raw_before, raw_clean, provided, fresh, cmp, epochs)
    save_epochs(subject, epochs, events)

    print("\nStage 1 complete for", subject,
          "- review derivatives/report/ before we proceed to Stage 2.")


if __name__ == "__main__":
    main()
