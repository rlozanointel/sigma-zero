import pandas as pd
from pydantic import ValidationError
import pytest
from src.dmaic_agent import calculate_cpk, DefectRecord

def test_calculate_cpk_normal():
    # Arrange
    data = pd.Series([5.0, 5.05, 4.95, 5.0, 5.0])
    lsl = 4.5
    usl = 5.5

    # Act
    result = calculate_cpk(data, lsl, usl)

    # Assert
    assert "cpk" in result
    assert "sigma_level" in result
    assert result["status"] == "OK"
    assert result["cpk"] >= 1.33

def test_calculate_cpk_critical():
    # Arrange
    data = pd.Series([5.8, 5.9, 5.7, 5.8, 5.9]) # Values outside USL
    lsl = 4.5
    usl = 5.5

    # Act
    result = calculate_cpk(data, lsl, usl)

    # Assert
    assert result["status"] == "CRITICAL"
    assert result["cpk"] < 1.0

def test_defect_record_validation():
    # Act & Assert
    with pytest.raises(ValidationError):
        # Missing required field 'defect_type'
        DefectRecord(
            timestamp="2026-05-06T08:00:00Z",
            product_id="PROD-1001",
            line_id="L1",
            shift="Morning"
        )
