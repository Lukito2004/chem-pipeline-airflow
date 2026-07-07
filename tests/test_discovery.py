from include.chem.discovery import dataset_ids_from_keys

def test_extracts_ids():
    keys = [
        "input/demo_scaffolds.csv",
        "input/demo_r_groups.csv",
        "input/batch_2024_scaffolds.csv",
        "input/notes.txt",
    ]
    assert dataset_ids_from_keys(keys) == ["batch_2024", "demo"]

def test_empty():
    assert dataset_ids_from_keys([]) == []