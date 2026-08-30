# Governance

## Why this document exists

`docs/LOCAL_COMPLETION_GUIDE.md`'s Governance and Operations phase asks
for a document explaining data, model, artifact, approval, rollback, and
retention practices. This is that document, describing what actually
happens in this repository today, not an aspirational policy for a team
that doesn't exist. Where a real practice would need more than what's
here, that's said explicitly, with a pointer to what would close the gap.

## Data governance

The raw dataset (`data/raw/fraud_raw.csv`) is the single source of truth
for training data. It's gitignored deliberately (see
`ENGINEERING_LOG.md`, the commit that untracked it) — a 750KB data file
doesn't belong in source control, and the pipeline is designed to
regenerate everything downstream of it on demand.

Every column's missingness is analyzed and documented before any
imputation policy is applied — see `docs/MISSINGNESS_ANALYSIS.md` and
`src/validation/imputation.py`'s `IMPUTATION_SPECS`. This is the data
governance mechanism that matters most here: no column gets silently
filled with a default. Each policy carries a `max_missing_rate` ceiling
that halts the pipeline if a new batch drifts past what the analysis was
validated against (`check_missing_rate_threshold`), rather than
continuing to impute on a stale assumption.

`data/validated/` and `data/processed/` are regenerated, not retained —
running `make validate` or `make features` overwrites them. Nothing
about their content is versioned beyond what the code that produced them
does; the validation stage's JSON report
(`artifacts/validation/validation_report_<timestamp>.json`) is the
closest thing to a data audit trail, and even that isn't currently
pruned or retained on any schedule (see Retention, below).

## Model governance

Training is deterministic (`random_state=42` throughout — see
`src/config/feature_config.py` and `src/config/training_config.py`), so
re-running `make train` against the same processed data reproduces the
same model. That's the primary governance property here: given the same
inputs, the output is auditable and repeatable, not a black box that
happened to come out a certain way once.

There is no model registry. `artifacts/training/model.joblib` is a single
path that gets overwritten on every training run. What actually
retains history is MLflow: `run_training()` logs a copy of the model,
the evaluation report, and the manifest into `.mlflow/<experiment_id>/<run_id>/artifacts/`
for every run (confirmed in this repo's own `.mlflow/` directory — two
real runs, each with its own preserved `model.joblib`). A reviewer can
run `mlflow ui --backend-store-uri file://<repo>/.mlflow` and pull an
older run's artifacts directly. This wasn't built as a formal model
registry, but it's the real mechanism doing that job today, and it's
worth naming honestly rather than pretending a heavier system exists.
The cloud-portable equivalent (`docs/ARCHITECTURE.md`'s mapping table)
is SageMaker Experiments, which would formalize this into an actual
registry with promotion stages.

## Artifact governance: training-serving skew

The one hard guarantee in this pipeline: `src/serving/main.py`'s startup
sequence (`load_and_verify_artifacts`) hashes the loaded model and
preprocessor and compares both against `manifest.json`, written at
training time by `train_model.py`'s `save_manifest()`. If either artifact
doesn't match what the manifest recorded, the service refuses to start
rather than silently serving a mismatched pair — see
`docs/ARCHITECTURE.md`'s "Artifact Manifest and Training-Serving Skew"
section, and the test that proves it
(`tests/test_serving.py::test_startup_fails_loudly_on_a_tampered_model`,
which tampers with a model file post-manifest and confirms startup
raises).

## Approval

This project doesn't have a multi-person approval board — pretending
otherwise would misrepresent its scale. What it does have is a real
division of labor, documented as it happened: implementation decisions
get made directly, but findings that cross into architecture decisions
(which dataset to use, whether to regenerate training data, whether to
pivot to a different problem shape) get surfaced explicitly before
acting on them, not decided unilaterally. `ENGINEERING_LOG.md`'s Entry 7
and Entry 9 are the concrete record of this: a finding (no predictive
signal in Dataset v1, then a PaySim alternative) was investigated,
written up, and routed to the role responsible for that class of
decision before any implementation followed.

Every non-trivial decision in this repository has a matching
`ENGINEERING_LOG.md` entry: Observation, Analysis, Decision, Tradeoffs,
Lessons Learned. That log is the approval and audit trail for a
single-maintainer project — the equivalent of a PR description and
review thread, just written as a permanent record instead of a closed
GitHub conversation. In an organization, this same information (why a
decision was made, what was rejected, what evidence supported it) is
what a model governance board or PR reviewer would ask for; here it's
written down as the decision happens instead of reconstructed after the
fact.

## Rollback

**Code**: standard git — `git revert` or checking out a prior commit.
Nothing project-specific here.

**Model**: no automated rollback exists, but the raw material for one
does. Since MLflow retains every run's artifacts under `.mlflow/`, rolling
back means copying an older run's `model.joblib` (and its matching
`preprocessor.joblib` and `manifest.json`, since the manifest hash check
means these three files always have to move together) back into
`artifacts/training/` and `artifacts/features/`. This is a manual
process today. `docs/ARCHITECTURE.md`'s AWS mapping table's answer for
what would replace this is versioned S3 objects plus ECR image tags —
each deployable image would be immutably tagged, and "rollback" would
mean pointing the ECS service at a prior image tag rather than
reconstructing files by hand.

**Serving**: there is no live, multi-version deployment to roll back
between. `docker compose` runs one container at a time; stopping the
current one and starting an image built from an older commit is the
entire rollback mechanism available locally. Blue-green or canary
rollback (per `docs/DEPLOYMENT.md`'s strategy discussion, if adopted)
would require the ECS/ALB setup `ARCHITECTURE.md` already maps this
project's `docker-compose` stack to.

## Retention

**MLflow runs** (`.mlflow/`) accumulate with no pruning — every training
run's artifacts stay indefinitely. At this project's scale (a handful of
runs total) that's a non-issue; it would need an actual policy before
being run continuously in any real setting. The cloud equivalent
(S3 lifecycle policies, per `ARCHITECTURE.md`'s mapping table) is exactly
where that policy would live.

**Validation reports** (`artifacts/validation/validation_report_*.json`)
are timestamped and also never pruned — each `make validate` run adds a
new file rather than overwriting the last one. Same gap, same eventual
answer: a retention policy belongs at the storage layer (S3 lifecycle
rules), not hand-rolled into this pipeline's code.

**Raw and processed data** (`data/`) is entirely local and gitignored;
nothing about it is retained beyond what's currently on disk. There is
no backup, versioning, or recovery mechanism for the raw dataset outside
of the fact that it was, at one point, committed to git history before
being deliberately untracked (recoverable via `git show`, as it was
earlier in this project — see the relevant `ENGINEERING_LOG.md` entry
for that recovery).
