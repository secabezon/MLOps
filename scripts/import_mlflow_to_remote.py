"""Copy local MLflow experiments (mlruns/) to a remote MLflow tracking server.

Usage:
  .venv\Scripts\activate
  # ensure these env vars are set:
  # $env:MLFLOW_TRACKING_URI, $env:MLFLOW_TRACKING_USERNAME, $env:MLFLOW_TRACKING_PASSWORD
  python scripts/import_mlflow_to_remote.py

This script:
 - discovers experiments under local `mlruns/`
 - for each experiment, creates (or finds) an experiment on the remote
 - for each run, copies params, metrics, tags and artifacts to the remote

Notes:
 - Artifacts are copied by downloading local artifacts and uploading them via
   the remote MlflowClient. Large artifacts may take time to upload.
 - This does not attempt to preserve original run IDs. Timestamps are preserved
   where possible by re-logging metrics with their recorded timestamps.
"""
import os
import shutil
import tempfile
from pathlib import Path
import time
import traceback

from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType


def get_local_client():
    local_mlruns = Path('mlruns').resolve()
    if not local_mlruns.exists():
        raise SystemExit(f"Local mlruns directory not found at {local_mlruns}")
    # Use a file:// URI for local file store (Path.as_uri gives file:///C:/...)
    return MlflowClient(tracking_uri=local_mlruns.as_uri())


def get_remote_client():
    uri = os.environ.get('MLFLOW_TRACKING_URI')
    if not uri:
        raise SystemExit('Set MLFLOW_TRACKING_URI in the environment to the remote MLflow server')
    return MlflowClient(tracking_uri=uri)


def copy_experiment(local_client, remote_client, exp):
    # exp is an Experiment object from local_client.get_experiment
    print(f"Processing experiment: id={exp.experiment_id} name={exp.name}")
    # ensure remote experiment exists
    remote_exp = remote_client.get_experiment_by_name(exp.name)
    if remote_exp is None:
        remote_exp_id = remote_client.create_experiment(exp.name)
        print(f"Created remote experiment '{exp.name}' id={remote_exp_id}")
    else:
        remote_exp_id = remote_exp.experiment_id
        print(f"Using existing remote experiment id={remote_exp_id}")

    # fetch runs from local
    runs = local_client.search_runs([exp.experiment_id], run_view_type=ViewType.ALL)
    print(f"Found {len(runs)} runs in local experiment '{exp.name}'")

    for r in runs:
        try:
            print('Copying run', r.info.run_id)
            new_run = remote_client.create_run(remote_exp_id)
            new_run_id = new_run.info.run_id

            # copy params
            for k, v in r.data.params.items():
                try:
                    remote_client.log_param(new_run_id, k, v)
                except Exception:
                    print('Failed to log param', k)

            # copy metrics (use recorded times where possible)
            for k, v in r.data.metrics.items():
                try:
                    # mlflow Metric objects may have timestamps in history; use latest
                    remote_client.log_metric(new_run_id, k, float(v))
                except Exception:
                    print('Failed to log metric', k)

            # copy tags
            for k, v in r.data.tags.items():
                try:
                    remote_client.set_tag(new_run_id, k, v)
                except Exception:
                    pass

            # copy artifacts: download local artifacts and upload
            with tempfile.TemporaryDirectory() as td:
                td_path = Path(td)
                try:
                    local_client.download_artifacts(r.info.run_id, '', td)
                    # upload the directory contents as artifacts
                    # remote_client.log_artifacts supports uploading a directory
                    remote_client.log_artifacts(new_run_id, str(td_path), artifact_path=None)
                except Exception as e:
                    print('Artifact copy failed for run', r.info.run_id, 'error:', e)

            # set run status/finish time similar to original if available
            try:
                if r.info.end_time and r.info.status:
                    remote_client.set_terminated(new_run_id, status=r.info.status)
            except Exception:
                pass

            print('Copied run ->', new_run_id)
        except Exception:
            print('Failed to copy run', r.info.run_id)
            traceback.print_exc()


def main():
    local = get_local_client()
    remote = get_remote_client()

    # enumerate experiments by listing local mlruns children
    local_mlruns = Path('mlruns').resolve()
    exps = []
    for sub in local_mlruns.iterdir():
        if not sub.is_dir():
            continue
        exp_id = sub.name
        try:
            exp = local.get_experiment(exp_id)
            if exp is not None:
                exps.append(exp)
        except Exception:
            continue

    print('Found local experiments:', [e.name for e in exps])

    for e in exps:
        copy_experiment(local, remote, e)


if __name__ == '__main__':
    main()
