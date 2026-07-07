import pandas as pd
import pytest
from include.chem.quality import validate_input, DataQualityError

def test_valid():
    df = pd.DataFrame({"smiles": ["c1ccccc1*", "CC*"]})
    assert validate_input(df, "x")["valid"] == 2

def test_missing_column():
    with pytest.raises(DataQualityError):
        validate_input(pd.DataFrame({"foo": [1]}), "x")

def test_empty():
    with pytest.raises(DataQualityError):
        validate_input(pd.DataFrame({"smiles": []}), "x")

def test_too_many_invalid():
    with pytest.raises(DataQualityError):
        validate_input(pd.DataFrame({"smiles": ["not_a_smiles", "bad one", "???"]}), "x")