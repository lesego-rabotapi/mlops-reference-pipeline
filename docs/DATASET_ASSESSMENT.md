# Dataset Assessment: predictive signal, separate from structural validity

## Why this document exists

`src/validation/` answers one question: is this data structurally sound?
Right schema, right dtypes, no duplicates, in-range values, missingness
handled with evidence-backed policies. `fraud_raw.csv` passes every one
of those checks cleanly.

That's a different question from: does this data contain a relationship
a supervised model could actually learn? A dataset can be structurally
perfect and still be unlearnable, if the label doesn't depend on the
features. This document is that second, separate assessment, and it's
deliberately kept separate. Great Expectations and the core validation
rules aren't the right place to test for predictive signal; see
[Why this isn't a validation rule](#why-this-isnt-a-validation-rule)
below.

**Verdict: Dataset v1 (`fraud_raw.csv`) is rejected for supervised
learning.**

## Methodology

`scripts/assess_dataset_signal.py` runs four independent checks, each
catching a kind of signal the others would miss:

1. **Linear correlation** (Pearson): catches monotonic linear
   relationships, blind to anything non-monotonic.
2. **Mutual information**: catches nonlinear or non-monotonic
   relationships correlation can't see (a U-shaped risk curve, for
   example), blind to relationships that only show up in combination
   with another feature.
3. **Decile / categorical-group fraud-rate spread**: catches a feature
   with a real "high-risk zone" even if its overall correlation washes
   out to near zero. Also checked pairwise (`is_first_transaction` x
   `device_type`) to catch interaction effects a univariate check would
   miss entirely.
4. **Row-order gap analysis** on the label itself: catches generation
   artifacts (clustering, periodicity, a drifting rate across the file)
   that would otherwise look identical to "no feature has signal" when
   the real story is closer to "the file was assembled with i.i.d.
   random labels."

Rerun with `python scripts/assess_dataset_signal.py`, it writes to
`artifacts/data_quality/signal_analysis.json`.

## Results (run against `fraud_raw.csv`, n=7,000)

| Check | Result |
|---|---|
| Max absolute correlation (any numeric feature) | 0.047 (`distance_from_home`) |
| Max mutual information (any feature) | 0.0053 (`velocity_score`), most features exactly 0.0000 |
| Max decile fraud-rate spread (any numeric feature) | 4.9 points (`velocity_score`: 8.6%-13.5%) |
| Max categorical-group fraud-rate spread | 2.7 points (`device_type`) |
| Pairwise interaction spread (`is_first_transaction` x `device_type`) | 8.3%-13.4%, within the noise band of the smallest cells (n≈109-321) |
| Row-order gap mean vs. expected under i.i.d. Bernoulli(0.103) | 9.71 observed vs. **9.71 expected** |

Every threshold check (correlation at least 0.10, mutual information at
least 0.01, group spread of at least 8 points) came back unflagged, full
detail in `artifacts/data_quality/signal_analysis.json`. The row-order
check is the most conclusive: the mean gap between consecutive fraud
rows matches the theoretical expectation for independent Bernoulli
sampling almost exactly, with no clustering, no periodicity, and no
drift between the first and second half of the file.

**Working hypothesis:** `is_fraud` in `fraud_raw.csv` was generated as
`Bernoulli(p≈0.103)`, independent of the other 12 columns. No model,
regardless of algorithm, hyperparameters, or feature engineering, can
learn a relationship that was never encoded in the generation process.

## How this was found

Not from this script directly, it was written after the finding, to
make the investigation repeatable. The finding itself came from
hardening the Training stage's evaluation metrics (see
[ENGINEERING_LOG.md, Entry 6](ENGINEERING_LOG.md)): switching from
accuracy and weighted-F1 to precision/recall/F1 on the fraud class
specifically, plus PR-AUC, exposed that the trained model predicted zero
fraud cases at any threshold, with ROC-AUC at 0.508 (a coin flip). That
result prompted this assessment, not the other way around. The metrics
were doing their job, they were designed to surface exactly this kind of
problem instead of hiding it behind a reassuring accuracy number, and
they did.

## Why this isn't a validation rule

It would be tempting to add a `predictive_signal_check` to
`VALIDATION_RULES` and have `validate_data.py` reject any dataset that
fails it. That's the wrong layer, for two reasons.

Validation runs on every batch, cheaply, before anything else. Signal
assessment needs a target column, a meaningful sample size, and, as this
document shows, real interpretive judgment about what "no signal" even
means (a strict binary flag would need to be paired with domain judgment
about acceptable strength, which this document's thresholds only
approximate). It's a different cadence and a different kind of check.

Conflating them also hides the distinction that matters. "This data is
well-formed" and "this data is useful for the modeling task" are
different claims, and keeping them as separate pipeline stages is what
lets you say, as is true here, "Validation: PASS, Signal Assessment:
FAIL" instead of one ambiguous PASS or FAIL that erases which question
actually failed.

## What comes next

This is an architecture decision, not an implementation one, flagged for
discussion rather than acted on unilaterally. The options on the table:

1. **Regenerate the dataset (Dataset v2)** with a documented,
   probabilistic data-generating process where `is_fraud` genuinely
   depends on the feature set (a risk score built from weighted feature
   contributions, passed through a sigmoid, then sampled via
   `Bernoulli(p)` rather than assigned deterministically, so a high-risk
   transaction can still turn out legitimate and vice versa, which is a
   more realistic and more interesting problem for the pipeline to
   handle than a deterministic rule would be).
2. **Keep Dataset v1** and reframe the deliverable around what the
   pipeline correctly did: detect and report a non-functional model
   rather than silently ship one.

Either way, this investigation and its artifacts are being kept, not
deleted. A rejected dataset with a documented reason is itself evidence
of a working data-quality gate, which is worth being able to show.

## Decision (resolved): stay on Dataset v1

A third option surfaced after this document was written: ChatGPT
proposed **PaySim** (`PS_20174392719_1491204439457_log.csv`, a
well-known public synthetic fraud dataset, 6.36M rows, 0.13% fraud rate)
as a candidate Dataset v2. Inspecting it directly confirmed it has real,
learnable signal, fraud occurs only in `TRANSFER`/`CASH_OUT` transaction
types (0% in `CASH_IN`/`DEBIT`/`PAYMENT`), plus a genuine `amount`
decile spread, unlike anything found in `fraud_raw.csv`'s signal
assessment above.

It was rejected as a same-project pivot, not on data-quality grounds:
its schema shares no columns with `fraud_raw.csv`, has zero missingness
(so none of the twelve MCAR-backed imputation policies in
`src/validation/imputation.py` apply), and carries far more extreme
class imbalance (0.13% vs. 10.3%). Adopting it would mean redoing the
same scope of work Entry 1 in `ENGINEERING_LOG.md` did for churn to
fraud, `SchemaRules`, `FeatureRules`, `feature_config.py`, and the
imbalance handling in training, not a dataset swap. That's a legitimate
future project, but an unnecessary pivot away from finishing this one.

**Final decision: keep Dataset v1**, with the deliverable framed
accordingly: a pipeline whose validation and evaluation stages correctly
identified that their input data couldn't support a supervised model,
end to end, rather than a trained fraud classifier. See
`ENGINEERING_LOG.md`, Entry 9.
