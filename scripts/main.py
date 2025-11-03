"""CLI para orquestar preprocesamiento, entrenamiento y logging de modelos.

Uso básico:
  python scripts/main.py --dry-run
  python scripts/main.py --train --model xgb --experiment "exp-name"
"""
from pathlib import Path
import argparse
import json
import sys

# Ensure project root is importable so `from src...` works when running
# the script from the repo root, VSCode, or CI systems.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.features.preprocessor import Preprocessor
from src.pipelines.sklearn_pipeline import build_pipeline
from src.models.trainer import ModelTrainer
from src.utils import mlflow_utils
from datetime import datetime
from pathlib import Path as _Path


OUT_PATH_CLEAN = PROJECT_ROOT / 'src' / 'data' / 'processed' / 'german_credit_clean.csv'
OUT_PATH_RAW = PROJECT_ROOT / 'src' / 'data' / 'raw' / 'german_credit_modified.csv'


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--raw', type=Path, default=OUT_PATH_RAW)
    p.add_argument('--target', type=str, default='kredit')
    p.add_argument('--model', type=str, default='logreg', help='logreg | rf | xgb')
    p.add_argument('--model-param', action='append', default=[],
                   help='Model parameter as key=value. Repeatable. Example: --model-param n_estimators=100')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--train', action='store_true')
    p.add_argument('--experiment', type=str, default='default')
    p.add_argument('--no-mlflow-autolog', action='store_true', default=False,
                   help='Disable mlflow.sklearn.autolog() when initializing experiment')
    p.add_argument('--mlflow-uri', type=str, default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # load raw data
    if not args.raw.exists():
        print(f"Raw file not found: {args.raw}")
        sys.exit(1)

    df = pd.read_csv(args.raw)

    # preprocess
    prep = Preprocessor()
    df_clean = prep.fit_transform(df)

    print(f"Data loaded: {df.shape} -> cleaned: {df_clean.shape}")

    if args.dry_run:
        print('Dry-run: showing first 5 rows of cleaned data:')
        print(df_clean.head().to_string())
        return

    if not args.train:
        print('Nothing to do: use --train to train a model or --dry-run to inspect data')
        return


    # split
    from sklearn.model_selection import train_test_split
    if args.target not in df_clean.columns:
        print(f"Target column '{args.target}' not found in data columns")
        sys.exit(1)

    X = df_clean.drop(columns=[args.target])
    y = df_clean[args.target]
    stratify_arg = y if y.nunique() <= 10 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_arg)

    

    # MLflow experiment init
    try:
        mlflow_uri = args.mlflow_uri
        # respect the CLI flag to disable autolog when requested
        enable_autolog = not getattr(args, 'no_mlflow_autolog', False)
        mlflow_utils.init_experiment(args.experiment, tracking_uri=mlflow_uri, autolog=enable_autolog)
        use_mlflow = True
    except Exception:
        print('MLflow not available or failed to init - continuing without MLflow')
        use_mlflow = False

    # parse model params from CLI (key=value pairs)
    def _coerce_value(v: str):
        if v.lower() in ('none', 'null'):
            return None
        if v.lower() in ('true', 'false'):
            return v.lower() == 'true'
        try:
            return int(v)
        except Exception:
            pass
        try:
            return float(v)
        except Exception:
            pass
        return v

    def parse_kv_list(kv_list):
        params = {}
        for item in kv_list:
            if not item or '=' not in item:
                continue
            k, v = item.split('=', 1)
            params[k] = _coerce_value(v)
        return params

    model_params = parse_kv_list(args.model_param)

    # build pipeline (pass model_params)
    pipe = build_pipeline(df_clean, model_name=args.model, target_col=args.target, model_params=model_params)

    trainer = ModelTrainer(pipe, model_name=args.model)

    # train and evaluate
    if use_mlflow:
        try:
            import mlflow
            with mlflow.start_run():
                trainer.fit(X_train, y_train)
                metrics = trainer.evaluate(X_test, y_test)
                print('metrics:', metrics)
                try:
                    mlflow.log_params({'model': args.model})
                    # also log explicit model_params passed via CLI
                    if model_params:
                        try:
                            mlflow.log_params(model_params)
                        except Exception:
                            # some params might be non-scalar; ignore logging failures
                            pass
                    for k, v in metrics.items():
                        mlflow.log_metric(k, float(v))
                except Exception:
                    pass

                # log model with signature
                try:
                    mlflow_utils.log_model_with_signature(trainer.estimator, X_train, args.model, registered_model_name=None)
                except Exception as e:
                    print('Failed to log model with signature:', e)

                # save local copy of the pipeline and metrics
                try:
                    ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
                    # use absolute project-root path so artifacts are saved in a stable location
                    artifact_dir = Path(__file__).resolve().parents[1] / 'src' / 'models' / 'artifacts'
                    artifact_dir.mkdir(parents=True, exist_ok=True)
                    model_file = artifact_dir / f"{args.model}_{ts}.joblib"
                    trainer.save(str(model_file), overwrite=True)
                    metrics_file = artifact_dir / f"metrics_{args.model}_{ts}.json"
                    with open(metrics_file, 'w') as fh:
                        json.dump(metrics, fh, indent=2)
                    try:
                        mlflow.log_artifact(str(model_file))
                        mlflow.log_artifact(str(metrics_file))
                    except Exception:
                        pass
                except Exception as e:
                    print('Warning: failed to save artifacts locally:', e)
        except Exception as e:
            print('MLflow run failed, falling back to local save. Error:', e)
            use_mlflow = False

    if not use_mlflow:
        trainer.fit(X_train, y_train)
        metrics = trainer.evaluate(X_test, y_test)
        print('metrics:', metrics)
        # save local copy of the pipeline and metrics
        try:
            ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
            # use absolute project-root path so artifacts are saved in a stable location
            artifact_dir = Path(__file__).resolve().parents[1] / 'src' / 'models' / 'artifacts'
            artifact_dir.mkdir(parents=True, exist_ok=True)
            model_file = artifact_dir / f"{args.model}_{ts}.joblib"
            trainer.save(str(model_file), overwrite=True)
            metrics_file = artifact_dir / f"metrics_{args.model}_{ts}.json"
            with open(metrics_file, 'w') as fh:
                json.dump(metrics, fh, indent=2)
            print(f"Saved model to {model_file} and metrics to {metrics_file}")
        except Exception as e:
            print('Warning: failed to save artifacts locally:', e)


if __name__ == '__main__':
    main()
