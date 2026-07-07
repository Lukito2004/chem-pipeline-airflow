"""Optional step 4: ChemProp molecular property prediction.

Trains a small message-passing neural network (MPNN) to predict logP from
SMILES, using the RDKit-computed logP as the training label, then predicts
logP for every generated molecule.

In production we'd train on experimental assay data. Here we train on a
computed property to demonstrate the ChemProp pipeline end-to-end. The whole
body is guarded so a library/API issue degrades to empty predictions instead
of failing the DAG.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def predict_logp(df: pd.DataFrame, max_epochs: int = 15) -> pd.DataFrame:
    df = df.copy()
    try:
        from chemprop import data, featurizers, models, nn
        from lightning import pytorch as pl

        smis = df["smiles"].tolist()
        targets = df["logp"].to_numpy().reshape(-1, 1)

        datapoints = [data.MoleculeDatapoint.from_smi(s, y)
                      for s, y in zip(smis, targets)]
        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        dset = data.MoleculeDataset(datapoints, featurizer)

        # normalize target for stable training; keep scaler to invert predictions
        scaler = dset.normalize_targets()
        train_loader = data.build_dataloader(dset, shuffle=True)
        predict_loader = data.build_dataloader(dset, shuffle=False)

        mp = nn.BondMessagePassing()
        agg = nn.MeanAggregation()
        output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
        ffn = nn.RegressionFFN(output_transform=output_transform)
        model = models.MPNN(mp, agg, ffn, batch_norm=True)

        trainer = pl.Trainer(
            max_epochs=max_epochs, accelerator="cpu", devices=1,
            enable_checkpointing=False, enable_progress_bar=False, logger=False,
        )
        trainer.fit(model, train_loader)

        batches = trainer.predict(model, predict_loader)
        preds = np.concatenate([b.numpy() for b in batches]).reshape(-1)
        df["chemprop_logp"] = preds
        return df
    except Exception as e:
        log.warning("ChemProp step failed (%s); writing empty predictions", e)
        df["chemprop_logp"] = np.nan
        return df