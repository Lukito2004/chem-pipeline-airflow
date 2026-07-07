import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski

def compute(smiles_list: list[str]) -> pd.DataFrame:
    rows = []
    for smi in smiles_list:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        rows.append({
            "smiles": smi,
            "mw": Descriptors.MolWt(m),
            "logp": Crippen.MolLogP(m),
            "hba": Lipinski.NumHAcceptors(m),
            "hbd": Lipinski.NumHDonors(m),
            "tpsa": Descriptors.TPSA(m),
            "rot_bonds": Descriptors.NumRotatableBonds(m),
        })
    return pd.DataFrame(rows)