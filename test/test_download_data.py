import sys
import os
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.COPY_download_data import main

@pytest.fixture # this makes pytest execute this code before running the test function
def mock_bank_data():
    """Create mock bank marketing data."""
    mock_data = MagicMock()
    
    # Create mock features DataFrame
    features = pd.DataFrame({
        'age': [30, 35, 40],
        'job': ['admin', 'technician', 'services']
    })
    
    # Create mock targets DataFrame
    targets = pd.DataFrame({
        'y': ['no', 'yes', 'no']
    })
    
    mock_data.data.features = features
    mock_data.data.targets = targets
    
    return mock_data

def test_csv_has_correct_shape(tmp_path, mock_bank_data):
    """Test that saved CSVs have the correct number of rows."""
    output_dir = str(tmp_path / "test_output")

    with patch('src.COPY_download_data.fetch_ucirepo', return_value=mock_bank_data):
        main(dataset_id=222, output_dir=output_dir)
        
        features = pd.read_csv(os.path.join(output_dir, "bank_marketing_features.csv"))
        targets = pd.read_csv(os.path.join(output_dir, "bank_marketing_targets.csv"))
        
        assert features.shape[0] == 3
        assert targets.shape[0] == 3
