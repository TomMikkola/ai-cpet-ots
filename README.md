# AI-CPET-OTS

Research workflows for overtraining syndrome (OTS): **0 = control, 1 = OTS**. Includes six trained models; private data is required.

## Installation

Clone the repository and use **Python 3.11**. From the root:

```sh
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` (PowerShell) or `source .venv/bin/activate` (Linux/macOS), then:

```sh
python -m pip install -r requirements.txt
python -m pip install -e ".[dev]"
python -m ipykernel install --prefix .venv --name ai-cpet-ots --display-name "Python 3 (AI-CPET-OTS)"
python -m jupyter lab
```

Select this kernel for model/statistics notebooks. Raw-data preprocessing requires the separate [Windows/Python 3.7.4 environment](environments/README.md).

## Execution

For raw recordings, run [preprocess.ipynb](notebooks/preprocess.ipynb), review `run-report.json`, then set `AI_CPET_DATA_DIR` to its output directory. Skip this step for compatible trimmed data.

Run notebook cells top to bottom. Evaluate models 1-4 before the ensembles, then run [statistics.ipynb](notebooks/statistics/statistics.ipynb).

| Notebook | Inputs | Model | Default threshold |
| --- | --- | --- | --- |
| `model-1.ipynb` | Four clinical characteristics | MLP | 0.563 |
| `model-2.ipynb` | Eleven conventional CPET variables | MLP | 0.547 |
| `model-3.ipynb` | Fourteen breath-by-breath features | Attention CNN | 0.784 |
| `model-4.ipynb` | Trimmed beat-by-beat | CNN | 0.322 |
| `model-1-2.ipynb` | Clinical and conventional CPET features | Feature-fusion MLP | 0.722 |
| `model-1-3-4.ipynb` | Clinical, breath-by-breath, and beat-by-beat logits | Weighted probability fusion | 0.290 |

Training and saving default to disabled. Enable `training_model` or `save_model` deliberately; new checkpoints go to `outputs/`. To use them, set `AI_CPET_MODELS_DIR` to a directory containing all required checkpoints.

Apply sigmoid once to model logits. Model 1-3-4 already returns probabilities.

## Private data and paths

Set environment variables before imports; restart the kernel after changing them. Relative paths start at the checkout root.

| Variable | Default | Purpose |
| --- | --- | --- |
| `AI_CPET_RAW_DATA_DIR` | `data/raw/` | Preprocessing inputs |
| `AI_CPET_PREPROCESSED_DIR` | `data/preprocessed/` | New preprocessing destination |
| `AI_CPET_DATA_DIR` | `data/` | Model inputs |
| `AI_CPET_MODELS_DIR` | `models/` | Checkpoints |
| `AI_CPET_OUTPUT_DIR` | `outputs/` | Results and new checkpoints |

Model inputs, for each `<cohort>` (`control` or `ots`):

- `<cohort>_clinical.xlsx`: `testID`, age, height, `sex`, weight, in order. `male` maps to 1; other sex values map to 0.
- `<cohort>_cpet.xlsx`: `testID`; oxygen uptake at maximum exercise, aerobic threshold, anaerobic threshold; heart rate at those three points; maximum ventilation; power at those three points; `VE/VCO2`. Features are read by position.
- `<cohort>_bbb/<testID>.xlsx`: `testID`, `time` in seconds, then `VO2 ml/kg/min`, `VO2`, `VCO2`, `VE`, `VA`, `Fb`, `VT`, `VD`, `VD/VT`, `PETCO2`, `PETO2`, `RER`, `VE/VO2`, `VE/VCO2`.
- Beat-by-beat directories: use the names in [paths.py](src/ai_cpet_ots/paths.py), with one two-channel `.npy` array per subject at 1000 Hz and at least 30000 samples per channel. The loader uses the first channel and downsamples to 500 Hz.

Preserve feature order and study units. Match participant membership and order across modalities. Keep private inputs and results out of Git, including custom output locations.

The statistics notebook exports `preds-df.xlsx`, then reads it for paired comparisons of sensitivity, specificity, likelihood ratios, and AUROC. Results go to `stats.xlsx`; both files use `AI_CPET_OUTPUT_DIR`.

## Reproducibility limits

- Checkpoints contain weights, not fitted scalers or participant splits; reproducing transformations requires training data.
- Sequence loaders use filesystem order. Verify alignment before ensembles; original splits and cohort selection are not reconstructed.
- Optimization folds are unseeded; scaling precedes inner cross-validation. Padding, batch composition and signal-processing assumptions can affect results.
- Preprocessing uses a reconstructed legacy environment. These research workflows do not establish clinical performance or exact historical equivalence.

## Checks

```sh
python -m ruff check src/ai_cpet_ots notebooks scripts
python -m ruff format --check src/ai_cpet_ots notebooks scripts
python scripts/check_publication.py
```
