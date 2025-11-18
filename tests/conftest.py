"""Pytest configuration and fixtures."""
import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'laufkont': np.random.choice(['A11', 'A12', 'A13', 'A14'], 100),
        'laufzeit': np.random.randint(6, 72, 100),
        'moral': np.random.choice(['A30', 'A31', 'A32', 'A33', 'A34'], 100),
        'verw': np.random.choice(['A40', 'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A47', 'A48', 'A49', 'A410'], 100),
        'hoehe': np.random.randint(250, 20000, 100),
        'sparkont': np.random.choice(['A61', 'A62', 'A63', 'A64', 'A65'], 100),
        'beszeit': np.random.choice(['A71', 'A72', 'A73', 'A74', 'A75'], 100),
        'rate': np.random.randint(1, 5, 100),
        'famges': np.random.choice(['A91', 'A92', 'A93', 'A94'], 100),
        'buerge': np.random.choice(['A101', 'A102', 'A103'], 100),
        'wohnzeit': np.random.randint(1, 5, 100),
        'verm': np.random.choice(['A121', 'A122', 'A123', 'A124'], 100),
        'alter': np.random.randint(19, 75, 100),
        'weitkred': np.random.choice(['A141', 'A142', 'A143'], 100),
        'wohn': np.random.choice(['A151', 'A152', 'A153'], 100),
        'bishkred': np.random.choice(['1', '2', '3', '4'], 100),
        'beruf': np.random.choice(['A171', 'A172', 'A173', 'A174'], 100),
        'pers': np.random.choice(['1', '2'], 100),
        'telef': np.random.choice(['A191', 'A192'], 100),
        'gastarb': np.random.choice(['A201', 'A202'], 100),
        'kredit': np.random.choice([0, 1], 100)
    })


@pytest.fixture
def clean_data(sample_data):
    """Provide clean data (no missing values)."""
    return sample_data.dropna()


@pytest.fixture
def model_path():
    """Provide path to test model artifact."""
    return ROOT / "src" / "models" / "artifacts" / "model.joblib"
