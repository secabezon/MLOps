import pandas as pd
from modeling_training import split, train
from preprocessing import load, preprocessing
from predict_eval import predict_eval
from pathlib import Path
import mlflow
from mlflow.models import infer_signature

pd.set_option('display.max_columns', None)

params={"n_estimators":300,"learning_rate":0.1,"max_depth":6}
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

mlflow.set_tracking_uri("http://127.0.0.1:5000")

###################### xgboost
mlflow.set_experiment("Test 3 modelos")

with mlflow.start_run():
    xgb=trainer.xgboost(**params)
    predict_xgb=predict_eval(xgb)
    y_pred_xgb=predict_xgb.predict(trainer.X_test)
    y_pred_proba_xgb=predict_xgb.predict_proba(trainer.X_test)
    eval_xgb=predict_xgb.eval(trainer.X_test, trainer.y_test)
    print(eval_xgb)

    mlflow.log_params(params)
    for key, value in eval_xgb.items():
        mlflow.log_metric(key, float(value))

    signature=infer_signature(trainer.X_train, predict_xgb.predict(trainer.X_train))

    model_info = mlflow.sklearn.log_model(
        sk_model=xgb,
        name='xgboost',
        signature=signature,
        input_example=trainer.X_train,
        registered_model_name='Modelo XgBoost'
    )

    mlflow.set_logged_model_tags(
        model_info.model_id,
        {"Training Info":"pred-xgboost"}
    )


######################### reg log
params={"max_iter":1000}

with mlflow.start_run():
    rl=trainer.reglog(**params)
    predict_rl=predict_eval(rl)
    y_pred_rl=predict_rl.predict(trainer.X_test)
    y_pred_proba_rl=predict_rl.predict_proba(trainer.X_test)
    eval_rl=predict_rl.eval(trainer.X_test, trainer.y_test)
    print(eval_rl)

    mlflow.log_params(params)
    for key, value in eval_rl.items():
        mlflow.log_metric(key, float(value))

    signature=infer_signature(trainer.X_train, predict_rl.predict(trainer.X_train))

    model_info = mlflow.sklearn.log_model(
        sk_model=rl,
        name='Reglog',
        signature=signature,
        input_example=trainer.X_train,
        registered_model_name='Modelo-Reglog'
    )

    mlflow.set_logged_model_tags(
        model_info.model_id,
        {"Training Info":"pred Regresion log"}
    )

######################### random forest
params={"n_estimators":100,"random_state":42,"n_jobs":-1}

with mlflow.start_run():
    rf=trainer.rf(**params)
    predict_rf=predict_eval(rf)
    y_pred_rf=predict_rf.predict(trainer.X_test)
    y_pred_proba_rf=predict_rf.predict_proba(trainer.X_test)
    eval_rf=predict_rf.eval(trainer.X_test, trainer.y_test)
    print(eval_rf)

    mlflow.log_params(params)
    for key, value in eval_rf.items():
        mlflow.log_metric(key, float(value))

    signature=infer_signature(trainer.X_train, predict_rf.predict(trainer.X_train))

    model_info = mlflow.sklearn.log_model(
        sk_model=rf,
        name='RandomForest',
        signature=signature,
        input_example=trainer.X_train,
        registered_model_name='Modelo-RandomForest'
    )

    mlflow.set_logged_model_tags(
        model_info.model_id,
        {"Training Info":"pred Random Forest"}
    )