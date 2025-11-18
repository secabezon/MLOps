"""Wrapper script to run a training experiment in a DVC stage and produce stable artifact paths.

This calls `scripts/main.py` programmatically (so we reuse the same CLI logic),
then locates the latest model/metrics files produced by `main` and writes them to
stable, DVC-tracked locations:
  - src/models/artifacts/model.joblib
  - src/models/artifacts/metrics.json

Usage:
  python scripts/dvc_train.py --raw src/data/processed/german_credit_clean.csv --model rf --experiment dvc_experiment
"""
from pathlib import Path
import sys
import shutil
import glob
import json
import time

# ensure project root importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

# import main function from scripts.main
from scripts.main import main as run_main


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--model', type=str, default='rf')
    p.add_argument('--experiment', type=str, default='dvc_experiment')
    p.add_argument('--no-mlflow-autolog', action='store_true')
    return p.parse_args(argv)


def latest_file(pattern):
    files = glob.glob(pattern)
    if not files:
        return None
    files = sorted(files, key=lambda p: Path(p).stat().st_mtime)
    return Path(files[-1])


def main(argv=None):
    args = parse_args(argv)

    # Run the existing CLI main with arguments to train the model.
    cli_args = [
        '--train',
        '--model', args.model,
        '--raw', str(args.raw),
        '--experiment', args.experiment,
    ]
    if args.no_mlflow_autolog:
        cli_args.append('--no-mlflow-autolog')

    # Call the existing main (it will save artifacts under src/models/artifacts/)
    print('Running training via scripts/main.py with args:', cli_args)
    run_main(cli_args)

    artifacts_dir = PROJECT_ROOT / 'src' / 'models' / 'artifacts'
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # find latest model_*.joblib and metrics_*.json created by main
    latest_model = latest_file(str(artifacts_dir / f"{args.model}_*.joblib"))
    latest_metrics = latest_file(str(artifacts_dir / f"metrics_{args.model}_*.json"))

    # copy to stable names for DVC
    stable_model = artifacts_dir / 'model.joblib'
    stable_metrics = artifacts_dir / 'metrics.json'

    if latest_model:
        shutil.copy2(latest_model, stable_model)
        print(f'Copied latest model {latest_model.name} -> {stable_model}')
    else:
        print('Warning: no model artifact found to copy')

    if latest_metrics:
        # also normalize metrics file (ensure json is valid and stable)
        with open(latest_metrics, 'r') as fh:
            metrics = json.load(fh)
        # write pretty-printed deterministic json
        with open(stable_metrics, 'w') as fh:
            json.dump(metrics, fh, sort_keys=True, indent=2)
        print(f'Copied latest metrics {latest_metrics.name} -> {stable_metrics}')
    else:
        print('Warning: no metrics artifact found to copy')

    print('DVC-friendly artifacts are ready.')


if __name__ == '__main__':
    main()
