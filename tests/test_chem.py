from include.chem import generation, properties, clustering

def test_generate():
    mols = generation.generate(["c1ccccc1*"], ["C*", "CC*"])
    assert len(mols) == 2

def test_properties():
    df = properties.compute(["c1ccccc1C"])
    assert {"mw", "logp", "hba", "hbd"} <= set(df.columns)

def test_cluster():
    df = properties.compute(["c1ccccc1C", "CCCCCCCC", "c1ccncc1"])
    out = clustering.cluster(df, k=2)
    assert "cluster" in out.columns