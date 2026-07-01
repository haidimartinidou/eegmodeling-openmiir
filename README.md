# OpenMIIR EEG Encoding-Model Analysis

An EEG encoding-model analysis of the [OpenMIIR](https://github.com/sstober/openmiir)
dataset (Stober et al., 2015): 64-channel EEG recorded while participants
perceived and imagined short music stimuli. The goal is to relate a
stimulus-derived predictor stream (note onsets, acoustic features, and
information-theoretic surprise/entropy from a statistical music model) to the
recorded neural response using temporal response functions (TRFs).

## Status

Environment scaffolding only. **No analysis code has been written yet** — this
repository currently contains the dependency definition, lock file, and the
planned pipeline below.

## Environment

The project uses [uv](https://docs.astral.sh/uv/) for environment and dependency
management.

- **Python**: 3.13 (managed by uv; pinned in `.python-version`)
- **Dependencies**: declared in `pyproject.toml`, fully resolved in `uv.lock`
- **Virtual environment**: `.venv/` (created by `uv sync`)

```bash
# Install uv (once), if not already present:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Reproduce the environment from the lock file:
uv sync
```

`uv sync` installs the exact locked versions and creates `.venv/` automatically.
Run anything inside the environment with `uv run`, e.g. `uv run python ...` or
`uv run jupyter lab`. Add a dependency with `uv add <package>`.

### Verify the install

```bash
uv run python -c "import mne, numpy, scipy, pandas, mtrf, py2lispIDyOM; print('ok')"
```

### Key packages

Versions below are the resolved minimums in `uv.lock`; `mne` is pinned to the
1.12.x line in `pyproject.toml`, the rest float to the latest compatible release.

| Package          | Version | Role                                                   |
|------------------|---------|--------------------------------------------------------|
| `mne`            | 1.12.x  | EEG loading, filtering, epoching, artifact handling    |
| `numpy`          | 2.x     | Array math                                             |
| `scipy`          | 1.18+   | Signal processing, stats                               |
| `pandas`         | 3.x     | Event/metadata tables, results wrangling               |
| `mtrf` (mTRFpy)  | 2.1.2   | Forward (encoding) and backward TRF models             |
| `py2lispIDyOM`   | 1.0.2   | Python wrapper for the IDyOM music-modelling engine    |

> **External backend:** `py2lispIDyOM` only *wraps* IDyOM, which is written in
> Common Lisp. To actually run IDyOM you must install **SBCL** separately
> (`brew install sbcl`) plus Quicklisp/IDyOM. It is not currently installed on
> this machine — importing the Python package does not require it, but Stage 2
> model fitting does.

## Planned four-stage pipeline

```
  raw EEG + stimuli
        │
        ▼
 ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
 │  Stage 1     │──▶│  Stage 2     │──▶│  Stage 3     │──▶│  Stage 4     │
 │ Preprocess   │   │ Stimulus     │   │ Encoding     │   │ Evaluation   │
 │ EEG          │   │ modelling    │   │ model (TRF)  │   │ & stats      │
 └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
```

### Stage 1 — EEG preprocessing (`mne`)
Load OpenMIIR raw recordings, set the montage, band-pass filter, re-reference,
and remove ocular/muscle artifacts (ICA). Extract trial events from the stimulus
markers and epoch the continuous data per stimulus / condition (perception vs.
imagination). Downsample to a common rate matched to the predictor stream.
**Output:** cleaned, epoched EEG aligned to stimulus onsets.

### Stage 2 — Stimulus modelling (`py2lispIDyOM`, `numpy`/`scipy`)
Build the predictor (feature) matrices that the encoding model will map onto the
EEG. This includes low-level acoustic/structural features (e.g. note-onset
envelope, spectral features) and, via IDyOM, information-theoretic predictors —
note-by-note **information content (surprise)** and **entropy** of the melodies.
Resample every predictor onto the EEG time base from Stage 1.
**Output:** time-aligned design matrices, one per trial/stimulus.

### Stage 3 — Encoding model / TRF (`mtrf`, with `mne` cross-checks)
Fit forward TRF models that predict each EEG channel from the lagged stimulus
predictors, using ridge-regularized regression over a range of time lags.
Select the regularization parameter by cross-validation across trials.
`mne.decoding.ReceptiveField` is available as an independent cross-check of the
mTRFpy fits.
**Output:** fitted TRF weights per subject/condition and predicted EEG.

### Stage 4 — Evaluation & statistics (`scipy`, `pandas`, `mne`)
Quantify model performance (predicted-vs-actual correlation / R²) on held-out
data, compare nested models (e.g. acoustic-only vs. acoustic + IDyOM surprise)
to isolate the unique contribution of each predictor, and test effects across
subjects and conditions with cluster-based permutation statistics.
**Output:** performance tables, TRF topographies, and group-level statistics.

## Repository layout

```
openmiir-encoding/
├── .venv/                  # virtual environment, created by uv (not version-controlled)
├── data/                   # OpenMIIR recordings + stimuli (not version-controlled)
├── src/                    # analysis code (to be written, per stage above)
├── notebooks/              # exploratory notebooks
├── results/                # model outputs, figures, statistics
├── pyproject.toml          # project metadata + direct dependencies
├── uv.lock                 # fully resolved dependency lock
├── .python-version         # pins the interpreter to Python 3.13
└── README.md
```

## References

- Stober, S., Sternin, A., Owen, A. M., & Grahn, J. A. (2015). *Towards Music
  Imagery Information Retrieval: Introducing the OpenMIIR Dataset of EEG
  Recordings from Music Perception and Imagination.* ISMIR.
- Crosse, M. J. et al. (2016). *The Multivariate Temporal Response Function
  (mTRF) Toolbox.* Frontiers in Human Neuroscience.
- Pearce, M. T. (2005). *The Construction and Evaluation of Statistical Models
  of Melodic Structure (IDyOM).* PhD thesis, City University London.
