from __future__ import annotations
import glob
import os
import tempfile

import pendulum
import pandas as pd
from airflow.decorators import dag, task
from airflow.models.param import Param
from airflow.operators.python import get_current_context

from include.chem import (s3_io, generation, properties, clustering,
                          prediction, faerun_build, quality, notifications)
from include.chem.discovery import dataset_ids_from_keys


@dag(
    dag_id="chem_pipeline_v3",
    schedule="@weekly",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_tasks=4,
    tags=["cheminformatics", "iter3"],
    default_args={"on_failure_callback": notifications.notify_failure},
    params={
        "overwrite": Param(False, type="boolean"),
        "n_clusters": Param(5, type="integer"),
        "run_chemprop": Param(True, type="boolean"),
        "run_faerun": Param(True, type="boolean"),
        "chemprop_epochs": Param(15, type="integer"),
    },
)
def chem_pipeline_v3():

    @task
    def discover_datasets(**ctx) -> list[str]:
        overwrite = ctx["params"]["overwrite"]
        hook = s3_io._hook()
        ids = dataset_ids_from_keys(s3_io.list_keys(prefix="input/"))
        selected = []
        for ds_id in ids:
            if not s3_io.exists(f"input/{ds_id}_r_groups.csv"):
                continue
            out_key = f"output/{ds_id}/clusters.csv"
            if overwrite or not s3_io.exists(out_key):
                selected.append(ds_id)
                continue
            out_mod = hook.get_key(out_key, bucket_name=s3_io.BUCKET).last_modified
            scaf_mod = hook.get_key(f"input/{ds_id}_scaffolds.csv", bucket_name=s3_io.BUCKET).last_modified
            rg_mod = hook.get_key(f"input/{ds_id}_r_groups.csv",  bucket_name=s3_io.BUCKET).last_modified
            if max(scaf_mod, rg_mod) > out_mod:
                selected.append(ds_id)
        print(f"Selected datasets: {selected}")
        return selected

    @task
    def process_dataset(ds_id: str) -> str:
        params = get_current_context()["params"]

        # The data quality gate
        scaf_df = s3_io.read_csv(f"input/{ds_id}_scaffolds.csv")
        rg_df = s3_io.read_csv(f"input/{ds_id}_r_groups.csv")
        quality.validate_input(scaf_df, f"{ds_id}_scaffolds")
        quality.validate_input(rg_df, f"{ds_id}_r_groups")

        # The pipeline
        mols = generation.generate(scaf_df["smiles"].tolist(), rg_df["smiles"].tolist())
        if not mols:
            raise quality.DataQualityError(f"{ds_id}: generated 0 valid molecules")
        s3_io.write_csv(pd.DataFrame({"smiles": mols}), f"output/{ds_id}/molecules.csv")

        props = properties.compute(mols)
        s3_io.write_csv(props, f"output/{ds_id}/properties.csv")

        clusters = clustering.cluster(props, k=params["n_clusters"])
        s3_io.write_csv(clusters, f"output/{ds_id}/clusters.csv")

        if params["run_chemprop"]:
            preds = prediction.predict_logp(props, max_epochs=params["chemprop_epochs"])
            s3_io.write_csv(preds, f"output/{ds_id}/predictions.csv")

        if params["run_faerun"]:
            with tempfile.TemporaryDirectory() as tmp:
                if faerun_build.build(clusters, os.path.join(tmp, "molecules")):
                    for path in glob.glob(os.path.join(tmp, "molecules.*")):
                        s3_io._hook().load_file(
                            path, key=f"output/{ds_id}/faerun/{os.path.basename(path)}",
                            bucket_name=s3_io.BUCKET, replace=True)
        return ds_id

    @task
    def report(selected: list[str]) -> None:
        notifications.notify_success(get_current_context()["dag"].dag_id, selected)

    datasets = discover_datasets()
    processed = process_dataset.expand(ds_id=datasets)
    summary = report(datasets)
    processed >> summary  # run the success summary only after processing

chem_pipeline_v3()