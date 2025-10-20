import pandas as pd
from modeling_training import split, train
from preprocessing import load, preprocessing
from predict_eval import predict_eval
from pathlib import Path

pd.set_option('display.max_columns', None)

OUT_PATH_CLEAN = Path('..') / 'src' / 'data'/ 'processed' /  'german_credit_clean.csv'

OUT_PATH_RAW = Path('..') / 'src' / 'data'/ 'raw' /  'german_credit_modified.csv'
numeric_cols=[]
missingvals=["n/a","na","null","?","unknown","error","invalid","none",""," "]
df = load(OUT_PATH_RAW)
proc_df=preprocessing(df.df)
proc_df=proc_df.delete_cols(['mixed_type_col'])
proc_df=proc_df.clean_colsname(' ','_')
proc_df=proc_df.normalize_miss_val(missingvals)
proc_df=proc_df.convert_num(proc_df.df.columns)
proc_df=proc_df.imputer_val(proc_df.df.columns)
proc_df=proc_df.cap_outliers(proc_df.df.columns, 1.5)
proc_df=proc_df.dropduplicates()
splitter = split(proc_df,'kredit')

trainer = train(splitter)
xgb=trainer.xgboost()
predict_xgb=predict_eval(xgb)
y_pred_xgb=predict_xgb.predict(trainer.X_test)
y_pred_proba_xgb=predict_xgb.predict_proba(trainer.X_test)
eval_xgb=predict_xgb.eval(trainer.X_test, trainer.y_test)
print(eval_xgb)
rl=trainer.reglog()
predict_rl=predict_eval(rl)
y_pred_rl=predict_rl.predict(trainer.X_test)
y_pred_proba_rl=predict_rl.predict_proba(trainer.X_test)
eval_rl=predict_rl.eval(trainer.X_test, trainer.y_test)
print(eval_rl)
rf=trainer.rf()
predict_rf=predict_eval(rf)
y_pred_rf=predict_rf.predict(trainer.X_test)
y_pred_proba_rf=predict_rf.predict_proba(trainer.X_test)
eval_rf=predict_rf.eval(trainer.X_test, trainer.y_test)
print(eval_rf)