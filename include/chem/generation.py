from rdkit import Chem

DUMMY = Chem.MolFromSmiles("*")

def combine(scaffold_smi: str, rgroup_smi: str) -> str | None:
    """Attach an R-group to a scaffold at the '*' dummy atom."""
    scaffold = Chem.MolFromSmiles(scaffold_smi)
    rgroup = Chem.MolFromSmiles(rgroup_smi)
    if scaffold is None or rgroup is None:
        return None
    products = Chem.ReplaceSubstructs(
        scaffold, DUMMY, rgroup, replacementConnectionPoint=0
    )
    if not products:
        return None
    product = products[0]
    try:
        Chem.SanitizeMol(product)
    except Exception:
        return None
    return Chem.MolToSmiles(product)

def generate(scaffolds: list[str], r_groups: list[str]) -> list[str]:
    """Full combinatorial enumeration scaffold x r_group."""
    out = []
    for s in scaffolds:
        for r in r_groups:
            smi = combine(s, r)
            if smi:
                out.append(smi)
    return sorted(set(out))