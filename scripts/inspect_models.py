"""
Inspecciona pipelines guardados (LogisticRegression y RandomForest/XGBoost)
Extrae coeficientes y feature importances y guarda top features en CSV.
"""
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

workspace = Path(__file__).resolve().parents[1]
# candidate paths
candidates = {
    'logreg': [workspace / 'src' / 'models' / 'pipeline_logreg.joblib', workspace / 'models' / 'pipeline_logreg.joblib'],
    'rf': [workspace / 'src' / 'models' / 'pipeline_model.joblib', workspace / 'models' / 'pipeline_model.joblib']
}

found = {}
for k, paths in candidates.items():
    for p in paths:
        if p.exists():
            found[k] = p
            break

print('Found pipelines:', found)

reports_dir = workspace / 'reports'
reports_dir.mkdir(parents=True, exist_ok=True)


def get_feature_names_from_column_transformer(ct: ColumnTransformer):
    feature_names = []
    # ct.transformers may contain tuples (name, transformer, columns)
    for name, trans, cols in ct.transformers:
        if name == 'remainder' or trans == 'drop':
            continue
        # get column list
        try:
            col_list = list(cols)
        except Exception:
            col_list = cols
        # dive into pipeline
        if hasattr(trans, 'steps'):
            last = trans.steps[-1][1]
        else:
            last = trans
        # try the sklearn method
        try:
            if hasattr(last, 'get_feature_names_out'):
                # Some implementations accept input_features
                try:
                    names = list(last.get_feature_names_out(col_list))
                except Exception:
                    names = list(last.get_feature_names_out())
            else:
                names = [f"{name}__{c}" for c in col_list]
        except Exception:
            names = [f"{name}__{c}" for c in col_list]
        feature_names.extend(names)
    return feature_names


def inspect_pipeline(path: Path):
    print('\nLoading', path)
    pipe = joblib.load(path)
    if not isinstance(pipe, Pipeline):
        print('Loaded object is not a sklearn Pipeline')
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
    # attempt to get feature names
    try:
        feat_names = get_feature_names_from_column_transformer(preproc)
    except Exception as e:
        print('Error getting feature names from transformer:', e)
        feat_names = []
    print('Extracted approx feature names:', len(feat_names))
    # get model
    model = pipe.steps[-1][1]
    if hasattr(model, 'coef_'):
        coef = model.coef_
        if coef.ndim > 1:
            coef = coef[0]
        if len(feat_names) != len(coef):
            print('Length mismatch coef vs feat_names', len(coef), len(feat_names))
            feat_names = [f'feat_{i}' for i in range(len(coef))]
        s = pd.Series(coef, index=feat_names).abs().sort_values(ascending=False)
        return ('coef', s)
    elif hasattr(model, 'feature_importances_'):
        imps = model.feature_importances_
        if len(feat_names) != len(imps):
            print('Length mismatch imp vs feat_names', len(imps), len(feat_names))
            feat_names = [f'feat_{i}' for i in range(len(imps))]
        s = pd.Series(imps, index=feat_names).sort_values(ascending=False)
        return ('imp', s)
    else:
        print('Model has no coef_ or feature_importances_')
        return None

all_results = {}
for name, path in found.items():
    res = inspect_pipeline(path)
    if res is not None:
        all_results[name] = res
        typ, ser = res
        out_csv = reports_dir / f'top_features_{name}.csv'
        ser.head(50).to_csv(out_csv)
        print(f'Saved top features to {out_csv}')

# print summary
for k, (typ, ser) in all_results.items():
    print(f"\nTop 10 features for {k} (type={typ}):")
    print(ser.head(10).to_string())

if not all_results:
    print('\nNo results to report. Asegúrate de haber ejecutado los notebooks y creado los pipelines.')
else:
    print('\nReports escritos en', reports_dir)
