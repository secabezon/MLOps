from typing import Optional


def init_experiment(name: str, tracking_uri: Optional[str] = None, autolog: bool = True):
    """Initialize or select an MLflow experiment.

    Parameters
    - name: experiment name to create/select
    - tracking_uri: optional MLflow tracking URI
    - autolog: whether to enable mlflow.sklearn.autolog()
    """
    try:
        import mlflow
    except Exception:
        raise ImportError('mlflow is not installed in this environment')

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    # Enable sklearn autologging if requested. This will capture params, metrics
    # and the model automatically for many sklearn estimators.
    if autolog:
        try:
            # Import the sklearn autolog function; in some mlflow versions it's
            # available under mlflow.sklearn.autolog
            mlflow.sklearn.autolog()
        except Exception:
            # If autolog isn't available or fails, continue without blocking.
            pass

    mlflow.set_experiment(name)


def log_model_with_signature(sklearn_model, X_sample, model_name: str, registered_model_name: Optional[str] = None):
    try:
        import mlflow
        from mlflow.models import infer_signature
    except Exception:
        raise ImportError('mlflow is not installed in this environment')

    signature = infer_signature(X_sample, sklearn_model.predict(X_sample))
    info = mlflow.sklearn.log_model(sk_model=sklearn_model, signature=signature, input_example=X_sample, registered_model_name=registered_model_name, artifact_path=model_name)
    return info
