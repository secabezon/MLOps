"""Wrapper script to produce processed dataset for DVC pipelines.

Usage example (from repo root):
  python scripts/dvc_preprocess.py --raw src/data/raw/german_credit_modified.csv --out src/data/processed/german_credit_clean.csv

This imports the project's `Preprocessor` and writes a deterministic output file that DVC can track.
"""
from pathlib import Path
import sys

# ensure project root importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import pandas as pd
from src.features.preprocessor import Preprocessor


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if not args.raw.exists():
        raise SystemExit(f"Raw file not found: {args.raw}")

    df = pd.read_csv(args.raw)
    prep = Preprocessor()
    df_clean = prep.fit_transform(df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(args.out, index=False)
    print(f"Wrote processed data to: {args.out}")


if __name__ == '__main__':
    main()
