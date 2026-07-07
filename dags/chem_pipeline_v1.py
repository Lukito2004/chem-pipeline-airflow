from __future__ import annotations
import glob
import os
import tempfile

import pendulum
import pandas as pd
from airflow.decorators import dag, task
from airflow.exceptions import AirflowSkipException
from airflow.models.param import Param

from include.chem import (s3_io, generation, properties, clustering,
                          prediction, faerun_build)


@dag(
    dag_id="chem_pipeline_v1",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    tags=["cheminformatics", "iter1"],
    params={
        "dataset_id": Param("demo", type="string"),
        "n_clusters": Param(5, type="integer"),
        "run_chemprop": Param(True, type="boolean"),
        "run_faerun": Param(True, type="boolean"),
        "chemprop_epochs": Param(15, type="integer"),
    },
)
def chem_pipeline_v1():

    @task
    def generate_molecules(**ctx) -> str:
        ds_id = ctx["params"]["dataset_id"]
        scaffolds = s3_io.read_csv(f"input/{ds_id}_scaffolds.csv")["smiles"].tolist()
        rgroups = s3_io.read_csv(f"input/{ds_id}_r_groups.csv")["smiles"].tolist()
        mols = generation.generate(scaffolds, rgroups)
        s3_io.write_csv(pd.DataFrame({"smiles": mols}), f"output/{ds_id}/molecules.csv")
        return ds_id

    @task
    def compute_properties(ds_id: str) -> str:
        mols = s3_io.read_csv(f"output/{ds_id}/molecules.csv")["smiles"].tolist()
        s3_io.write_csv(properties.compute(mols), f"output/{ds_id}/properties.csv")
        return ds_id

    @task
    def cluster_molecules(ds_id: str, **ctx) -> str:
        df = s3_io.read_csv(f"output/{ds_id}/properties.csv")
        df = clustering.cluster(df, k=ctx["params"]["n_clusters"])
        s3_io.write_csv(df, f"output/{ds_id}/clusters.csv")
        return ds_id

    @task
    def predict_properties(ds_id: str, **ctx) -> str:
        if not ctx["params"]["run_chemprop"]:
            raise AirflowSkipException("run_chemprop=False")
        df = s3_io.read_csv(f"output/{ds_id}/properties.csv")
        out = prediction.predict_logp(df, max_epochs=ctx["params"]["chemprop_epochs"])
        s3_io.write_csv(out, f"output/{ds_id}/predictions.csv")
        return ds_id

    @task
    def build_faerun(ds_id: str, **ctx) -> str:
        if not ctx["params"]["run_faerun"]:
            raise AirflowSkipException("run_faerun=False")
        df = s3_io.read_csv(f"output/{ds_id}/clusters.csv")
        with tempfile.TemporaryDirectory() as tmp:
            html = faerun_build.build(df, os.path.join(tmp, "molecules"))
            if html is None:
                raise AirflowSkipException("faerun unavailable")
            for path in glob.glob(os.path.join(tmp, "molecules.*")):
                s3_io._hook().load_file(
                    path, key=f"output/{ds_id}/faerun/{os.path.basename(path)}",
                    bucket_name=s3_io.BUCKET, replace=True)
        return ds_id

    ds = generate_molecules()
    ds = compute_properties(ds)
    ds = cluster_molecules(ds)
    predict_properties(ds)
    build_faerun(ds)

chem_pipeline_v1()