import pandas as pd
from modeling_training import split,train
from pathlib import Path

OUT_PATH_CLEAN = Path('..') / 'src' / 'data'/ 'processed' /  'german_credit_clean.csv'


df = pd.read_csv(OUT_PATH_CLEAN)
print(df)

splitter = split(df.drop(columns=['kredit'], inplace=False), df['kredit'])

trainer = train(splitter)

print(trainer.xgboost())
print(trainer.X_train.shape)
print(trainer.X_test.shape)