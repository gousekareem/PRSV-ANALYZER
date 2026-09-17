# Data Version Control (DVC) setup

This repo has DVC initialized for real (not just documented): `git log` shows
an actual commit tracking `prsv_project/data/training_features.csv` and
`prsv_project/models/` via DVC's content-addressed `.dvc` pointer files
(`training_features.csv.dvc`, `models.dvc`). Every reported accuracy number
in `models/training_summary.json` is now reproducibly tied to the exact
dataset/model snapshot committed alongside it.

## What's already done
- `dvc init` has been run; `.dvc/config` and `.dvc/.gitignore` exist.
- `prsv_project/data/training_features.csv` is tracked (`dvc add`).
- `prsv_project/models/` (the trained SVM + scaler + label encoder + SHAP
  background + metadata) is tracked as one directory output.
- Both are committed to git as `.dvc` pointer files - the actual large
  binary content lives in DVC's local cache (`.dvc/cache/`, git-ignored),
  the same separation DVC always uses.

## What you need to add for a real remote
DVC's local cache alone doesn't share data with collaborators or CI. Point
it at a remote storage location:

```bash
# Any of: local path, S3, GCS, Azure Blob, SSH, Google Drive, etc.
dvc remote add -d storage s3://your-bucket/prsv-analyzer-dvc
# or, for a zero-cost option to start with:
dvc remote add -d storage /path/to/a/shared/network/drive

dvc push        # uploads the tracked training_features.csv + models/ content
git add .dvc/config
git commit -m "Configure DVC remote"
```

From then on, the normal DVC workflow applies:

```bash
# After pulling new code that references updated data/models:
dvc pull

# After retraining (ml/train_svm.py or ml/train_ensemble.py) and being happy
# with the new models/ contents:
dvc add prsv_project/models
git add prsv_project/models.dvc
git commit -m "Retrain: <describe what changed>"
dvc push
```

## Why this matters here specifically
This project's evaluation numbers (`models/evaluation/evaluation_metrics.json`,
`models/evaluation/ensemble_comparison.json`) are only meaningful if tied to
an exact, reproducible dataset+model pairing. Before this, "the model got
94% accuracy" had no way to specify which exact `training_features.csv` or
which exact `svm_model.joblib` that referred to if either changed later.
Now `git log -- prsv_project/models.dvc` gives that history directly.
