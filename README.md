# chem-pipeline-airflow

An Apache Airflow pipeline for a cheminformatics drug-discovery workflow. Scientists
drop paired CSV files into S3 (`<id>_scaffolds.csv` and `<id>_r_groups.csv`); the
pipeline generates candidate molecules and characterises them.

## Pipeline steps

For each dataset `id`:

1. **Molecule generation**: combine every scaffold x R-group at the `*` attachment point (RDKit)
2. **Property calculation**: MW, logP, HBA, HBD, TPSA, rotatable bonds (RDKit)
3. **Clustering**: K-Means on Morgan fingerprints
4. **ChemProp prediction** *(optional)*: trains a small MPNN
5. **Faerun graph** *(optional)*: interactive 2D map (PCA/t-SNE on fingerprints), coloured by cluster

Inputs are CSVs with a single `smiles` column, e.g.:

```csv
smiles
c1ccccc1*
CC*
```

## The three DAGs (iterations)

| DAG                | Trigger                    | Description                                                                                  |
| ------------------ | -------------------------- | -------------------------------------------------------------------------------------------- |
| `chem_pipeline_v1` | Manual, `dataset_id` param | Processes one dataset end-to-end                                                             |
| `chem_pipeline_v2` | `@weekly`                  | Auto-discovers new/changet with **dynamic task mapping**; `overwrite` param to reprocess all |
| `chem_pipeline_v3` | `@weekly`                  | Adds **data-quality checks** + **MS Teams notifications** on failure/success                 |

**Idempotency:** v2/v3 only process a dataset if it has no output yet or its input is newer than its output, so a re-run does no redundant work. `overwrite=True` forces reprocessing.

## Project structure

```
chem-pipeline-airflow/
├── dags/
│   ├── chem_pipeline_v1.py       # iteration 1
│   ├── chem_pipeline_v2.py       # iteration 2
│   └── chem_pipeline_v3.py       # iteration 3
├── include/chem/                 # business logic (unit-testable, no Airflow)
│   ├── s3_io.py                  # S3 read/write helpers
│   ├── generation.py             # scaffold x R-group enumeration
│   ├── properties.py             # RDKit descriptors
│   ├── clustering.py             # Morgan fingerprints + K-Means
│   ├── prediction.py             # optional ChemProp
│   ├── faerun_build.py           # optional Faerun graph
│   ├── discovery.py              # dataset-id discovery from S3 keys
│   ├── quality.py                # input data-quality checks
│   └── notifications.py          # MS Teams webhook
├── tests/                        # pytest unit tests
├── samples/                      # example input CSVs
├── Dockerfile                    # Airflow image + deps (CPU-only PyTorch)
├── docker-compose.yaml           # Airflow (Celery) + Postgres + Redis + MinIO
├── requirements.txt
└── conftest.py                   # puts repo root on sys.path for pytest
```

Business logic lives in `include/chem/` (light, testable), keeping the DAG files thin.

## Tech stack

- **Airflow 2.9.3** (CeleryExecutor, Postgres, Redis) via Docker Compose
- **RDKit** for cheminformatics, **scikit-learn** for clustering/projection
- **ChemProp** (CPU PyTorch) for property prediction, **Faerun** for visualization
- **MinIO** as S3-compatible object storage (drop-in for AWS S3: same `boto3`/`S3Hook`
  code, only the `endpoint_url` differs)

## Local setup

Prerequisites: Docker + Docker Compose, the AWS CLI (for uploading test data), >=4 GB
RAM for Docker.

```bash
# 1. One-time: set your UID so mounted files aren't owned by root (Linux)
echo "AIRFLOW_UID=$(id -u)" > .env

# 2. Build the image (installs RDKit/ChemProp/Faerun; first build is slow)
docker compose build

# 3. Start everything (Airflow + Postgres + Redis + MinIO)
docker compose up -d
```

Airflow UI: http://localhost:8080 (`airflow` / `airflow`)
MinIO console: http://localhost:9001 (`minioadmin` / `minioadmin`)

### Configure the S3 connection

MinIO auto-creates the `chem-pipeline` bucket on startup. Register it as Airflow's
`aws_default` connection:

```bash
docker compose run --rm airflow-cli airflow connections add aws_default \
  --conn-type aws \
  --conn-login minioadmin \
  --conn-password minioadmin \
  --conn-extra '{"endpoint_url":"http://minio:9000","region_name":"us-east-1"}'
```

The bucket name is read from the `CHEM_S3_BUCKET` env var (set in `docker-compose.yaml`,
default `chem-pipeline`).

### Upload sample data

```bash
aws s3 cp samples/demo_scaffolds.csv s3://chem-pipeline/input/demo_scaffolds.csv --endpoint-url http://localhost:9000
aws s3 cp samples/demo_r_groups.csv s3://chem-pipeline/input/demo_r_groups.csv  --endpoint-url http://localhost:9000
```

## Running

**UI:** DAGs -> unpause the DAG -> Trigger. For v1, pass config `{"dataset_id": "demo"}`.

**CLI:**

```bash
docker compose run --rm airflow-cli airflow dags trigger chem_pipeline_v1 --conf '{"dataset_id":"demo"}'
docker compose run --rm airflow-cli airflow dags trigger chem_pipeline_v2
docker compose run --rm airflow-cli airflow dags trigger chem_pipeline_v2 --conf '{"overwrite": true}'
```

Outputs land in `s3://chem-pipeline/output/<id>/` (`molecules.csv`, `properties.csv`,
`clusters.csv`, `predictions.csv`, `faerun/molecules.html`). Browse them in the MinIO
console. To view a Faerun graph, download the `faerun/` folder and serve it over HTTP
(`python3 -m http.server`); `file://` will not render it.

## Data quality & notifications (iteration 3)

Inputs are validated before any compute: `smiles` column present, file non-empty, and >= 80% of SMILES parseable by RDKit; generation must yield ≥1 molecule. Failures raise a `DataQualityError`, which fails the task and triggers an MS Teams alert via
`on_failure_callback`; successful runs post a summary.

The webhook URL is read from the Airflow Variable `msteams_webhook_url`:

```bash
docker compose run --rm airflow-cli airflow variables set msteams_webhook_url "<url>"
```

Demoed against webhook.site; for production set it to a Teams Incoming Webhook / Power
Automate Workflow URL, no code change required.

## Testing

```bash
docker compose run --rm --entrypoint bash \
  -v "$PWD/tests:/opt/airflow/tests" \
  -v "$PWD/conftest.py:/opt/airflow/conftest.py" \
  airflow-cli -c "pip install -q pytest && cd /opt/airflow && python -m pytest tests -q"
```

CI (GitHub Actions) runs the same tests on every push/PR.

## Branching policy

`feature/* -> dev -> prod`. Work happens on feature branches; PRs merge into `dev` for
integration; `dev` is released to `prod`. `dev` and `prod` are protected (PR-only).