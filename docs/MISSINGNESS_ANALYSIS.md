# Missingness Analysis: fraud.csv

`data/raw/fraud_raw.csv` has real missingness in 12 of its 13 columns (only
the label, `is_fraud`, is complete). The Validation stage does not apply one
blanket imputation strategy to all of them -- each column's missingness is
tested for whether it looks random (MCAR: missing completely at random) or
structural (MNAR: missing not at random, i.e. the fact that it's missing
tells you something) before a policy is chosen. Getting this wrong matters:
naively filling an MNAR column can erase a real fraud signal instead of
being a neutral substitution.

This document is the record of that analysis: methodology, results, and the
policy each column ended up with. **Re-run the methodology, don't
pattern-match a verdict from this table onto a new column or a new batch of
data** -- see [Repeating this analysis](#repeating-this-analysis) below.

---

## Methodology

For each column with missingness, four checks are run against
`data/raw/fraud_raw.csv` (n=7,000):

1. **Fraud-rate split.** Compare `is_fraud` rate between rows where the
   column is missing vs. present, with a two-proportion z-test. A
   significant gap (p < 0.05) means missingness correlates with the label
   itself -- a strong MNAR signal, and a reason to *not* impute naively
   (you'd be smoothing away the thing you're trying to detect).
2. **Categorical driver crosstabs.** Missing rate of the column, grouped by
   every other low-cardinality column (`is_first_transaction`,
   `device_type`, `store_type`, `is_weekend`, `hour_of_day`). A wide spread
   across groups suggests the value is structurally uncomputable for some
   subgroup (e.g. near-100% missing for one group would mean "this field
   doesn't apply to that group," not "randomly lost").
3. **Continuous driver quartiles.** Same idea, but binning continuous
   columns (`transaction_amount`, `network_quality`, `prev_transactions`,
   `customer_age`, `distance_from_home`, `velocity_score`) into quartiles
   first.
4. **Cross-column missingness correlation.** Correlation between the
   column's missing-indicator and every other column's missing-indicator.
   A high correlation suggests a shared mechanism (e.g. a batch of rows
   from one faulty collector missing several fields together) worth
   investigating on its own, rather than treating each column as
   independently random.

None of these individually *prove* MCAR -- they're evidence-gathering, not
a formal statistical test (a true Little's MCAR test needs the joint
distribution across all variables at once). "No signal found" is read as
"no evidence against MCAR," combined with a domain judgment call: would
this field plausibly be uncomputable for a specific, identifiable subgroup?
If yes, that's a reason to look harder even without a strong statistical
signal.

## Policy once a column reads as MCAR

- **Continuous columns**: median imputation (outlier-robust; for every
  column analyzed so far, mean and median were close enough that this is a
  robustness default, not a skew correction).
- **Categorical/binary columns**: mode imputation -- the same MCAR-driven
  default `SimpleImputer(strategy="most_frequent")` already applies
  downstream in `build_features.py`'s categorical pipeline; making it
  explicit here means it's audited and rate-guarded before that ever runs.
- **Every imputed column** gets a `<column>_was_missing` indicator column
  (0/1) preserved in the validated dataset, so the imputation is auditable
  and reversible even though current evidence says none of these
  indicators are fraud-informative. They're excluded from model features
  by default (`COLUMNS_TO_DROP` in `src/config/feature_config.py`) but the
  data isn't destroyed -- move one into `NUMERIC_FEATURES` if future
  evidence changes that.
- **Every policy carries a missing-rate ceiling** (`max_missing_rate`): the
  smallest multiple of 5 that is `>= 2x` the validated missing rate. This
  is a drift guard, not a tuned hyperparameter -- if a new batch's
  missingness blows past it, validation halts instead of silently
  imputing an ever-larger share of the column on an assumption that was
  only ever checked at the original rate.

If a column reads as MNAR instead, it does **not** get this treatment --
naive imputation would need to be replaced with something that preserves
the signal (e.g. a dedicated "missing" category, or deferring to a
model-side handling strategy). No column analyzed so far has required this;
see [Results](#results).

## Repeating this analysis

```bash
python scripts/analyze_missingness.py <column> [<column> ...]
python scripts/analyze_missingness.py --all   # every column currently listed
```

The script implements the four checks above and flags (`**...**`) anything
that looks MNAR-leaning: a significant fraud-rate gap, a wide crosstab/
quartile spread, or a notable missingness correlation. Run it:

- Before writing an `ImputationSpec` for any **new column** that starts
  showing missingness.
- Whenever the **missing rate for an already-imputed column shifts
  meaningfully** (even below its ceiling) -- the ceiling stops silent
  drift past the validated rate, but it doesn't re-validate the MCAR
  assumption itself.
- Periodically against **new data batches**, not just this static CSV --
  the conclusions here are dataset-specific.

---

## Results

All 12 columns with missingness were tested and read as MCAR -- no
statistically significant fraud-rate gap, no crosstab/quartile spread over
~3 points, and no meaningful cross-column missingness correlation (max
observed: 0.037). Full per-column detail is in the docstring-adjacent
comments in `src/validation/imputation.py`; summary:

| Column | Missing | Strategy | Fraud rate (missing vs. present) | p-value | Max spread | Max miss-corr | Ceiling |
|---|---|---|---|---|---|---|---|
| `velocity_score` | 15.0% | median | 8.8% vs 10.6% | -- | -- | -- | 30% |
| `customer_age` | 12.0% | median | 9.4% vs 10.4% | -- | ~2 pts | 0.01 | 25% |
| `distance_from_home` | 10.0% | median | 8.1% vs 10.5% | -- | ~2 pts | 0.02 | 20% |
| `network_quality` | 9.0% | median | 9.7% vs 10.4% | 0.593 | 2.1 pts | 0.022 | 20% |
| `transaction_amount` | 8.0% | median | 10.4% vs 10.3% | 0.963 | 2.8 pts | 0.037 | 20% |
| `prev_transactions` | 7.0% | median | 11.0% vs 10.2% | 0.586 | 1.2 pts | 0.024 | 15% |
| `hour_of_day` | 5.0% | mode | 10.3% vs 10.3% | 0.993 | 1.8 pts | 0.013 | 10% |
| `device_type` | 4.0% | mode | 7.9% vs 10.4% | 0.170 | 1.4 pts | 0.027 | 10% |
| `num_items` | 3.0% | median | 6.7% vs 10.4% | **0.079** | 1.3 pts | 0.020 | 10% |
| `is_first_transaction` | 3.0% | mode | 11.0% vs 10.3% | 0.752 | 1.4 pts | 0.029 | 10% |
| `is_weekend` | 2.0% | mode | 7.1% vs 10.4% | 0.214 | 1.4 pts | 0.037 | 5% |
| `store_type` | 2.0% | mode | 6.4% vs 10.4% | 0.128 | 1.0 pt | 0.013 | 5% |

*`velocity_score`, `customer_age`, `distance_from_home` were analyzed by
direct crosstab before `scripts/analyze_missingness.py` existed, so they
don't have a recorded p-value/spread/correlation in this table -- their
narrative writeups (below and in `imputation.py`) cover the same ground.*

**Note on `num_items`**: p=0.079 is the closest to significant (0.05) of
any column tested, on the smallest missing-group sample (n=210). It's
still read as MCAR on current evidence, but it's the weakest case of the
twelve -- re-run its analysis before raising its ceiling or leaning on it
elsewhere.

**Note on `is_weekend` / `store_type`**: both have only a 2% baseline
missing rate, so their ceilings (5%) leave less absolute room than the
other columns before the guard fires. This is intentional -- small
baseline rates mean small absolute swings are large relative swings, so
they're not given the same wide berth.

### Narrative summaries

**`velocity_score`** (15% missing) -- cross-tabulated against
`is_first_transaction` (14.7% vs 18.3% missing) and `is_fraud` (8.8% vs
10.6% fraud rate): no sharp split. If velocity were structurally
uncomputable for first-time transactions, missingness there would approach
100%, not 18.3%.

**`customer_age`** (12%) and **`distance_from_home`** (10%) -- both flat
across every categorical driver and continuous quartile tested, fraud rate
flat-to-lower in the missing group, near-zero missingness correlation with
every other column.

**The remaining 9 columns** (`network_quality` through `store_type` in the
table above) -- same pattern, formalized with the two-proportion z-test
once `scripts/analyze_missingness.py` existed: no significant fraud-rate
gap, no wide driver/quartile spread, no meaningful missingness correlation
with any other column.

---

## Non-goals

- **`is_fraud`** (the target) is never imputed -- it has 0% missingness and
  is hard-required non-null by the `target_not_null` validation rule.
- This document does not cover feature *engineering* decisions (e.g.
  whether `hour_of_day`'s missing-indicator should ever become a feature)
  beyond the imputation policy itself -- see
  `src/config/feature_config.py` for how the validated columns map to
  model inputs.
