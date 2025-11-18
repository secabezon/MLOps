"""Export all local MLflow experiments using mlflow-export-import.

Usage:
  .venv\Scripts\activate
  python scripts/export_mlflow_all.py --out tmp_mlflow_export

This script lists experiments from the local `mlruns` directory and calls
`mlflow-export-import experiments export` for each one, writing a subdir per experiment.
"""
import os
import subprocess
from pathlib import Path
from mlflow.tracking import MlflowClient
import argparse
import sys


def safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in '-_.' else '_' for c in name)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=str, default='tmp_mlflow_export')
    args = p.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # assume local mlruns in ./mlruns
    local_mlruns = Path('mlruns').resolve()
    if not local_mlruns.exists():
        print('No local mlruns directory found at', local_mlruns)
        return

    # point mlflow client to local file store (use a valid file URI on Windows)
    client = MlflowClient(tracking_uri=local_mlruns.as_uri())

    # MlflowClient may not expose list_experiments in all versions; enumerate
    # experiments by scanning the mlruns directory and calling get_experiment
    exps = []
    for sub in local_mlruns.iterdir():
        if not sub.is_dir():
            continue
        exp_id = sub.name
        try:
            exp = client.get_experiment(exp_id)
            if exp is not None:
                exps.append(exp)
        except Exception:
            # skip entries that are not valid experiment ids
            continue

    print('Found experiments:', [e.name for e in exps])

    for e in exps:
        name = e.name
        subdir = out_dir / safe_name(name)
        subdir.mkdir(parents=True, exist_ok=True)
        print('\nExporting experiment', name, '->', subdir)
        # Prefer calling the venv-installed CLI directly (exe in Scripts) if available,
        # otherwise fall back to python -m (older installs may expose different modules).
        scripts_dir = Path(sys.executable).parent
        exe = scripts_dir / 'mlflow-export-import.exe'
        if not exe.exists():
            exe = scripts_dir / 'mlflow-export-import'

        if exe.exists():
            cmd = [str(exe), 'experiments', 'export',
                   '--experiment-name', name,
                   '--output-dir', str(subdir),
                   '--export-run-metrics']
        else:
            # fallback: try python -m (best-effort)
            cmd = [sys.executable, '-m', 'mlflow_export_import.experiments', 'export',
                   '--experiment-name', name, '--output-dir', str(subdir), '--export-run-metrics']
        # call subprocess; rely on mlflow-export-import to export artifacts
        try:
            subprocess.run(cmd, check=True)
            print('Exported', name)
        except subprocess.CalledProcessError as exc:
            print('Failed to export', name, 'error:', exc)

if __name__ == '__main__':
    main()
