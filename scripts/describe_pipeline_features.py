from pathlib import Path
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest

workspace = Path(__file__).resolve().parents[1]
logreg_candidates = [workspace / 'src' / 'models' / 'pipeline_logreg.joblib', workspace / 'models' / 'pipeline_logreg.joblib']
rf_candidates = [workspace / 'src' / 'models' / 'pipeline_model.joblib', workspace / 'models' / 'pipeline_model.joblib']

logreg_path = next((p for p in logreg_candidates if p.exists()), None)
rf_path = next((p for p in rf_candidates if p.exists()), None)

print('logreg_path:', logreg_path)
print('rf_path:', rf_path)


def inspect(path):
    print('\nInspecting', path)
    pipe = joblib.load(path)
    print('Pipeline steps:', [s for s,_ in pipe.steps])
    preproc = None
    selector = None
    for step_name, step in pipe.steps:
        if isinstance(step, ColumnTransformer):
            preproc = step
        if isinstance(step, SelectKBest):
            selector = step
        if isinstance(step, Pipeline):
            for sub_name, sub in step.steps:
                if isinstance(sub, ColumnTransformer):
                    preproc = sub
                if isinstance(sub, SelectKBest):
                    selector = sub
    if preproc is None:
        print('  No ColumnTransformer found')
    else:
        print('  Found ColumnTransformer with transformers:')
        for name, transformer, cols in preproc.transformers:
            print('   -', name)
            try:
                col_list = list(cols)
                print('      cols sample (first 10):', col_list[:10])
                print('      cols count:', len(col_list))
            except Exception as e:
                print('      cols repr:', cols, 'error listing:', e)
            # try get_feature_names_out
            try:
                if hasattr(transformer, 'get_feature_names_out'):
                    try:
                        names = transformer.get_feature_names_out(col_list)
                    except Exception:
                        try:
                            names = transformer.get_feature_names_out()
                        except Exception as e:
                            names = None
                    print('      get_feature_names_out ->', None if names is None else f'len={len(names)} sample={list(names)[:5]}')
                else:
                    print('      transformer has no get_feature_names_out')
            except Exception as e:
                print('      error calling get_feature_names_out:', e)
    if selector is not None:
        try:
            mask = selector.get_support()
            print('  SelectKBest present. k=', mask.sum())
        except Exception as e:
            print('  SelectKBest present but error calling get_support():', e)
    model = pipe.steps[-1][1]
    if hasattr(model, 'coef_'):
        print('  Model coef shape:', model.coef_.shape)
    if hasattr(model, 'feature_importances_'):
        print('  Model feature_importances_ length:', len(model.feature_importances_))

if logreg_path:
    inspect(logreg_path)
if rf_path:
    inspect(rf_path)

print('\nDone')
