import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.cluster import KMeans


def fingerprint_matrix(smiles_list: list[str], n_bits: int = 2048) -> np.ndarray:
    """Morgan fingerprints (radius 2) as a dense (n_molecules, n_bits) array."""
    arrs = []
    for smi in smiles_list:
        m = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        arrs.append(arr)
    return np.array(arrs)


def cluster(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    df = df.copy()
    if len(df) < 2:
        df["cluster"] = 0
        return df
    X = fingerprint_matrix(df["smiles"].tolist())
    k = min(k, len(df))
    df["cluster"] = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
    return df