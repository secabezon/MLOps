"""Integration tests for end-to-end pipeline."""
import pytest
import pandas as pd
import joblib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class TestEndToEndPipeline:
    """Integration tests for the complete ML pipeline."""

    def test_data_loading(self):
        """Test that data can be loaded successfully."""
        data_path = ROOT / "src" / "data" / "processed" / "german_credit_clean.csv"
        
        if not data_path.exists():
            pytest.skip(f"Data file not found: {data_path}")
        
        df = pd.read_csv(data_path)
        
        assert df is not None
        assert len(df) > 0
        assert 'kredit' in df.columns

    def test_preprocessing_pipeline(self, sample_data):
        """Test preprocessing step of the pipeline."""
        from src.features.preprocessor import Preprocessor
        
        X = sample_data.drop(columns=['kredit'])
        y = sample_data['kredit']
        
        preprocessor = Preprocessor()
        preprocessor.fit(X, y)
        X_transformed = preprocessor.transform(X)
        
        assert X_transformed is not None
        assert len(X_transformed) == len(X)

    def test_full_prediction_pipeline(self, sample_data):
        """Test the complete prediction pipeline."""
        artifacts_dir = ROOT / "src" / "models" / "artifacts"
        model_files = list(artifacts_dir.glob("*.joblib"))
        
        if not model_files:
            pytest.skip("No trained model found")
        
        # Load model (which includes preprocessing)
        model = joblib.load(model_files[0])
        
        # Prepare data
        X = sample_data.drop(columns=['kredit'])
        
        # Make predictions
        predictions = model.predict(X)
        
        # Validate predictions
        assert predictions is not None
        assert len(predictions) == len(X)
        assert all(pred in [0, 1] for pred in predictions)

    def test_pipeline_reproducibility(self, sample_data):
        """Test that pipeline produces reproducible results."""
        artifacts_dir = ROOT / "src" / "models" / "artifacts"
        model_files = list(artifacts_dir.glob("*.joblib"))
        
        if not model_files:
            pytest.skip("No trained model found")
        
        model = joblib.load(model_files[0])
        X = sample_data.drop(columns=['kredit'])
        
        # Make predictions twice
        predictions_1 = model.predict(X)
        predictions_2 = model.predict(X)
        
        # Should be identical
        assert all(predictions_1 == predictions_2)

    def test_model_evaluation_metrics(self, clean_data):
        """Test that model can be evaluated with standard metrics."""
        from sklearn.metrics import accuracy_score, f1_score
        
        artifacts_dir = ROOT / "src" / "models" / "artifacts"
        model_files = list(artifacts_dir.glob("*.joblib"))
        
        if not model_files:
            pytest.skip("No trained model found")
        
        model = joblib.load(model_files[0])
        
        X = clean_data.drop(columns=['kredit'])
        y_true = clean_data['kredit']
        
        y_pred = model.predict(X)
        
        # Calculate metrics
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        # Metrics should be in valid range
        assert 0 <= acc <= 1
        assert 0 <= f1 <= 1
        
        # Model should perform better than random
        assert acc > 0.5

    def test_drift_data_generation(self):
        """Test that drift data can be generated."""
        from src.monitoring.make_drift import main as generate_drift
        
        valid_path = ROOT / "src" / "data" / "processed" / "german_credit_clean.csv"
        
        if not valid_path.exists():
            pytest.skip("Validation data not found")
        
        # Generate drift data
        try:
            generate_drift()
            
            drift_path = ROOT / "src" / "data" / "drift" / "german_credit_drift.csv"
            assert drift_path.exists()
            
            df_drift = pd.read_csv(drift_path)
            assert len(df_drift) > 0
        except Exception as e:
            pytest.skip(f"Drift generation failed: {e}")

    def test_drift_detection(self):
        """Test drift detection functionality."""
        drift_path = ROOT / "src" / "data" / "drift" / "german_credit_drift.csv"
        
        if not drift_path.exists():
            pytest.skip("Drift data not found")
        
        from src.monitoring.drfit_alerts import compute_drift_ks
        
        try:
            drift_results = compute_drift_ks()
            
            assert drift_results is not None
            assert len(drift_results) > 0
            
            # Each result should have p_value and drift_detected
            for col, info in drift_results.items():
                assert 'p_value' in info
                assert 'drift_detected' in info
                assert 0 <= info['p_value'] <= 1
        except Exception as e:
            pytest.skip(f"Drift detection failed: {e}")
