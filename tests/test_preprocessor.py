import pandas as pd
from src.features.preprocessor import Preprocessor


def test_preprocessor_impute_and_clean():
    df = pd.DataFrame({
        'Age ': ['20', 'na', '30', '40'],
        'Score': ['100', '200', 'na', '400'],
        'mixed_type_col': [1, 2, 3, 4]
    })

    prep = Preprocessor(missing_values=['na'], drop_cols=['mixed_type_col'])
    out = prep.fit_transform(df)

    # column names cleaned
    assert 'age' in out.columns
    # drop cols removed
    assert 'mixed_type_col' not in out.columns
    # numeric columns imputed -> no all-null in age/score
    assert out['age'].isna().sum() == 0
    assert out['score'].isna().sum() == 0
