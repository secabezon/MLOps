"""Unit tests for preprocessing module."""
import pytest
import pandas as pd
import numpy as np
from src.features.preprocessor import Preprocessor


class TestPreprocessor:
    """Test suite for Preprocessor class."""

    def test_preprocessor_init(self):
        """Test Preprocessor initialization."""
        preprocessor = Preprocessor()
        assert preprocessor is not None
        assert hasattr(preprocessor, 'fit')
        assert hasattr(preprocessor, 'transform')

    def test_preprocessor_fit_transform(self, sample_data):
        """Test basic fit and transform."""
        preprocessor = Preprocessor()
        
        X = sample_data.drop(columns=['kredit'])
        y = sample_data['kredit']
        
        # Fit
        preprocessor.fit(X, y)
        
        # Transform
        X_transformed = preprocessor.transform(X)
        
        assert X_transformed is not None
        assert len(X_transformed) == len(X)

    def test_preprocessor_handles_missing_values(self):
        """Test that preprocessor handles missing values."""
        df = pd.DataFrame({
            'laufkont': ['A11', 'A12', None, 'A14'],
            'laufzeit': [12, 24, 36, None],
            'hoehe': [1000, 2000, None, 4000],
            'kredit': [0, 1, 0, 1]
        })
        
        preprocessor = Preprocessor()
        X = df.drop(columns=['kredit'])
        
        preprocessor.fit(X)
        X_transformed = preprocessor.transform(X)
        
        # Should not raise error and should produce output
        assert X_transformed is not None

    def test_preprocessor_deterministic(self, sample_data):
        """Test that preprocessor produces deterministic results."""
        preprocessor = Preprocessor()
        
        X = sample_data.drop(columns=['kredit'])
        
        preprocessor.fit(X)
        X_transformed_1 = preprocessor.transform(X)
        X_transformed_2 = preprocessor.transform(X)
        
        # Results should be identical
        pd.testing.assert_frame_equal(
            pd.DataFrame(X_transformed_1),
            pd.DataFrame(X_transformed_2)
        )

    def test_preprocessor_column_names(self, clean_data):
        """Test that preprocessing maintains reasonable column structure."""
        preprocessor = Preprocessor()
        
        X = clean_data.drop(columns=['kredit'])
        preprocessor.fit(X)
        X_transformed = preprocessor.transform(X)
        
        # Should have some columns after transformation
        assert X_transformed.shape[1] > 0
