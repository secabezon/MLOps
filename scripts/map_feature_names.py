"""
Reconstruye nombres de features transformadas a partir del ColumnTransformer guardado en un pipeline.
Genera un top-10 interpretado para LogisticRegression y RandomForest si están disponibles.
"""
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

workspace = Path(__file__).resolve().parents[1]
clean_csv = workspace / 'src' / 'data' / 'processed' / 'german_credit_clean.csv'

logreg_path = next((p for p in [workspace/'src'/'models'/'pipeline_logreg.joblib', workspace/'models'/'pipeline_logreg.joblib'] if p.exists()), None)
rf_path = next((p for p in [workspace/'src'/'models'/'pipeline_model.joblib', workspace/'models'/'pipeline_model.joblib'] if p.exists()), None)

print('clean_csv', clean_csv.exists())
if clean_csv.exists():
    df = pd.read_csv(clean_csv)
else:
    df = None


def build_feature_names_from_column_transformer(ct: ColumnTransformer):
    feature_names = []
    for name, trans, cols in ct.transformers:
        if name == 'remainder' or trans == 'drop':
            continue
        # get column list
        try:
            col_list = list(cols)
        except Exception:
            col_list = cols
        # if pipeline, get last step
        if hasattr(trans, 'steps'):
            last = trans.steps[-1][1]
        else:
            last = trans
        # OneHotEncoder
        try:
            from sklearn.preprocessing import OneHotEncoder
            from sklearn.preprocessing import OrdinalEncoder
        except Exception:
            OneHotEncoder = None
            OrdinalEncoder = None
        if OneHotEncoder is not None and isinstance(last, OneHotEncoder):
            # build names from categories_ (some sklearn versions use .categories)
            cats = getattr(last, 'categories_', None) or getattr(last, 'categories', None)
            if cats is None:
                # fallback: one feature per input column
                for col in col_list:
                    feature_names.append(col)
            else:
                for col, cat in zip(col_list, cats):
                    for c in cat:
                        feature_names.append(f"{col}__{c}")
        elif OrdinalEncoder is not None and isinstance(last, OrdinalEncoder):
            # ordinal -> one feature per column
            for col in col_list:
                feature_names.append(col)
        else:
            # fallback: numeric/scaler/imputer -> 1 feature per input col
            for col in col_list:
                feature_names.append(col)
    return feature_names


def inspect_pipeline(path: Path):
    print('\nLoading', path)
    pipe = joblib.load(path)
    if not isinstance(pipe, Pipeline):
        print('Not a sklearn Pipeline')
        return None
    # find ColumnTransformer
    preproc = None
    for step_name, step in pipe.steps:
        if isinstance(step, ColumnTransformer):
            preproc = step
            break
        if isinstance(step, Pipeline):
            for sub_name, sub in step.steps:
                if isinstance(sub, ColumnTransformer):
                    preproc = sub
                    break
    if preproc is None:
        print('No ColumnTransformer found in pipeline')
        return None
    names = build_feature_names_from_column_transformer(preproc)
    print('Built', len(names), 'feature names (approx)')
    model = pipe.steps[-1][1]
    if hasattr(model, 'coef_'):
        coef = model.coef_
        if coef.ndim > 1:
            coef = coef[0]
        print('coef len', len(coef))
        # align
        if len(coef) == len(names):
            ser = pd.Series(coef, index=names).abs().sort_values(ascending=False)
            return ('coef', ser)
        else:
            print('Length mismatch coef vs names', len(coef), len(names))
            # create generic names
            ser = pd.Series(coef, index=[f'feat_{i}' for i in range(len(coef))]).abs().sort_values(ascending=False)
            return ('coef', ser)
    elif hasattr(model, 'feature_importances_'):
        imps = model.feature_importances_
        print('imp len', len(imps))
        if len(imps) == len(names):
            ser = pd.Series(imps, index=names).sort_values(ascending=False)
            return ('imp', ser)
        else:
            print('Length mismatch imp vs names', len(imps), len(names))
            ser = pd.Series(imps, index=[f'feat_{i}' for i in range(len(imps))]).sort_values(ascending=False)
            return ('imp', ser)
    else:
        print('Model has no coef_ or feature_importances_')
        return None

results = {}
for label, path in [('logreg', logreg_path), ('rf', rf_path)]:
    if path is None:
        print('No pipeline for', label)
        continue
    res = inspect_pipeline(path)
    if res is not None:
        results[label] = res
        typ, ser = res
        out = workspace / 'reports' / f'top_features_{label}_mapped.csv'
        ser.head(50).to_csv(out)
        print('Saved mapped top features to', out)

# print readable summary
for k, (typ, ser) in results.items():
    print(f"\nTop 10 for {k} (type={typ}):")
    print(ser.head(10).to_string())

if not results:
    print('No results')
else:
    print('\nDone, reports in', workspace / 'reports')
