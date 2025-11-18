"""Unit tests for model training and evaluation."""
import pytest
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


class TestModelTraining:
    """Test suite for model training and metrics."""

    def test_model_artifact_exists(self, model_path):
        """Test that model artifact exists."""
        # Check if any model exists in artifacts directory
        artifacts_dir = model_path.parent
        model_files = list(artifacts_dir.glob("*.joblib"))
        
        assert len(model_files) > 0, "No model artifacts found"

    def test_model_can_load(self, model_path):
        """Test that model can be loaded."""
        artifacts_dir = model_path.parent
        model_files = list(artifacts_dir.glob("*.joblib"))
        
        if model_files:
            model = joblib.load(model_files[0])
            assert model is not None
            assert hasattr(model, 'predict')

    def test_model_predictions_shape(self, model_path, sample_data):
        """Test that model produces correct prediction shape."""
        artifacts_dir = model_path.parent
        model_files = list(artifacts_dir.glob("*.joblib"))
        
        if not model_files:
            pytest.skip("No trained model found")
        
        model = joblib.load(model_files[0])
        X = sample_data.drop(columns=['kredit'])
        
        predictions = model.predict(X)
        
        assert len(predictions) == len(X)
        assert all(pred in [0, 1] for pred in predictions)

    def test_model_probability_predictions(self, model_path, sample_data):
        """Test that model produces valid probability predictions."""
        artifacts_dir = model_path.parent
        model_files = list(artifacts_dir.glob("*.joblib"))
        
        if not model_files:
            pytest.skip("No trained model found")
        
        model = joblib.load(model_files[0])
        X = sample_data.drop(columns=['kredit'])
        
        try:
            probas = model.predict_proba(X)
            
            assert probas.shape[0] == len(X)
            assert probas.shape[1] == 2  # Binary classification
            assert all((probas >= 0).all(axis=1))
            assert all((probas <= 1).all(axis=1))
            # Probabilities should sum to 1
            assert all(abs(probas.sum(axis=1) - 1.0) < 1e-5)
        except AttributeError:
            pytest.skip("Model does not support predict_proba")


class TestMetricsCalculation:
    """Test suite for metrics calculation."""

    def test_accuracy_calculation(self):
        """Test accuracy metric calculation."""
        y_true = [0, 1, 0, 1, 1]
        y_pred = [0, 1, 0, 0, 1]
        
        acc = accuracy_score(y_true, y_pred)
        
        assert 0 <= acc <= 1
        assert acc == 0.8

    def test_f1_score_calculation(self):
        """Test F1 score calculation."""
        y_true = [0, 1, 0, 1, 1, 0]
        y_pred = [0, 1, 0, 0, 1, 0]
        
        f1 = f1_score(y_true, y_pred)
        
        assert 0 <= f1 <= 1

    def test_roc_auc_calculation(self):
        """Test ROC AUC calculation."""
        y_true = [0, 1, 0, 1, 1, 0]
        y_proba = [0.1, 0.9, 0.2, 0.6, 0.85, 0.15]
        
        auc = roc_auc_score(y_true, y_proba)
        
        assert 0 <= auc <= 1

    def test_metrics_with_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = [0, 1, 0, 1, 1]
        y_pred = [0, 1, 0, 1, 1]
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        assert acc == 1.0
        assert f1 == 1.0
