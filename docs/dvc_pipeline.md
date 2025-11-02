# DVC pipeline setup and usage

This repository includes a minimal `dvc.yaml` with two stages: `preprocess` and `train`.

Files added:
- `scripts/dvc_preprocess.py` — wrapper that runs the `Preprocessor` and writes `src/data/processed/german_credit_clean.csv`.
- `scripts/dvc_train.py` — wrapper that runs `scripts/main.py --train` and copies the latest artifacts to stable paths for DVC:
  - `src/models/artifacts/model.joblib`
  - `src/models/artifacts/metrics.json` (used as `metrics` in `dvc.yaml`)
- `dvc.yaml` — pipeline definition.

Quick start (from repo root):

1. Initialize DVC (only once):

```powershell
# install dvc if you don't have it
pip install dvc
# initialize dvc in repo
dvc init
```

2. (Optional) Configure a remote to push large data/artifacts (S3, GDrive, Azure, etc.):

```powershell
# example: local remote
dvc remote add -d local_remote dvc-storage
# for S3/GDrive see DVC docs
```

3. Run the pipeline (reproduces stages and creates the outputs):

```powershell
# run full pipeline
dvc repro
# or run a single stage
dvc repro preprocess
dvc repro train
```

4. Inspect the DAG:

```powershell
dvc dag
# for a visual graph (requires graphviz installed):
dvc dag --ascii
```

5. Track and push changes (typical flow):

```powershell
# add tracked outputs to git + dvc
git add dvc.yaml .dvcignore
# track generated outputs with dvc if you want to push large files to remote
# example: track processed data (this creates a .dvc file)
dvc add src/data/processed/german_credit_clean.csv
# commit and push
git add src/data/processed/german_credit_clean.csv.dvc
git commit -m "Add DVC pipeline and processed data"
# push data to remote
dvc push
```

Notes and recommendations:
- The training wrapper disables MLflow autolog by default (`--no-mlflow-autolog` in `dvc.yaml`) to avoid interactive UI artifacts during CI. You can enable MLflow and set `--mlflow-uri` to a remote tracking server if desired.
- The wrappers copy artifacts to stable filenames (`model.joblib`, `metrics.json`) so DVC can consistently track them.
- If your raw data is very large, add it to DVC and configure a remote storage (S3/GDrive) before running `dvc push`.
- To visualize the pipeline with a GUI, you can use `dvc dag --dot | dot -Tpng -o pipeline.png` (requires Graphviz `dot`).

If you want, I can also:
- create `.dvcignore` entries or a `.gitignore` update to avoid checking large files into Git,
- add a Makefile target to run `dvc repro` and common flows,
- create a GitHub Action to run `dvc repro` + run tests on PRs.
