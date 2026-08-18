# Engineering Log

## Purpose

This is not a changelog — `git log` already records *what* changed. This
records **why**: the reasoning trail behind each decision, the alternatives
that were considered and rejected, and what went wrong along the way. It
exists so the thinking behind this pipeline can be studied and replicated
independently, not just the code.

Each entry follows the reasoning pattern from
[MENTORSHIP_OPERATING_PLAN.md](MENTORSHIP_OPERATING_PLAN.md):
**Observation -> Analysis -> Decision -> Tradeoffs**, plus an explicit
**Lessons Learned** section — the part most worth re-reading later, because
it's the part that generalizes beyond this specific repo.

Entries are appended as work happens, not rewritten after the fact. If a
later entry contradicts an earlier one, both stay — that's part of the
record.

---

## Entry 1: Replacing the churn scaffold with the fraud dataset

**Observation.** The repo's actual design brief (`CLAUDE.md`) described a
fraud-detection pipeline already validated against `data/fraud.csv`. The
repo itself was still wired entirely to a customer-churn dataset — schema,
config, rules, and tests all assumed columns like `CreditScore`,
`Geography`, `Exited`. `data/fraud.csv` didn't exist in the repo at all.

**Analysis.** Two explanations were possible: either this was the wrong
repo/branch, or the churn pipeline was leftover scaffolding from an earlier
phase that the design brief had already moved past. The mismatch was too
large to guess past — schema, business rules, and every fixture would need
rewriting either way, and guessing wrong would mean building the wrong
thing twice.

**Decision.** Asked rather than assumed. The user confirmed: replace churn
entirely, fraud is the one real dataset. Once that was confirmed, the raw
CSV was inspected directly (`pandas.read_csv(...).describe()`,
`.isnull().mean()`) rather than trusting the design brief's numbers
blindly — they matched (7,000 rows, 10.3% fraud rate, `velocity_score` at
15% missing), which was itself useful confirmation the brief was current.

**Tradeoffs.** Rewiring `SchemaRules`, `FeatureRules`, `feature_config.py`,
and every test fixture in one pass was a bigger change than a single
column fix, but a half-migrated state (some files fraud-shaped, some still
churn-shaped) would have been worse — it would silently pass some checks
against the wrong schema and fail others for confusing reasons.

**Lessons learned.**
- When a task's premise doesn't match the observable state of the repo,
  stop and surface the mismatch before writing code against either
  version. The cost of asking is one message; the cost of building against
  a guessed-wrong premise is a full rewrite.
- Verify a design document's factual claims against the actual data before
  trusting them for downstream decisions — a brief can describe a
  should-be-true state that isn't the current state.

---

## Entry 2: `velocity_score` missingness — MCAR vs. MNAR

**Observation.** `velocity_score` (15% missing, the highest of any column)
needed an imputation policy. The easy path is `fillna(median)` and move on.

**Analysis.** Missingness is not automatically safe to fill. If a column is
missing *because of* something related to the outcome being predicted
(MNAR — missing not at random), naive imputation doesn't just lose
information, it actively overwrites a real signal with a plausible-looking
lie. The only way to know which case this is: test it. Cross-tabulated
missingness against `is_first_transaction` (would velocity be
structurally uncomputable for first-time customers with no transaction
history?) and against `is_fraud` (is missingness itself concentrated in
fraud cases?).

**Decision.** Missing rate was 14.7% vs. 18.3% across `is_first_transaction`
— a small gap, not the near-100%-for-one-group split that structural
uncomputability would produce. Fraud rate was 8.8% (missing) vs. 10.6%
(present) — missingness isn't concentrated in fraud. Read as MCAR; median
imputation applied, with a `velocity_score_was_missing` indicator column
preserved and a missing-rate ceiling (30%, ~2x the validated 15%) guarding
the assumption against future drift.

**Tradeoffs.** This took real analysis time versus just filling and moving
on. The alternative — assuming MCAR without checking — would have been
indistinguishable in the short term and only would have surfaced as a
problem if the assumption were wrong and nobody caught it.

**Lessons learned.**
- "Missing" is not one category. Test *why* before choosing *how* to
  handle it — the two questions have different, sometimes contradictory,
  answers.
- Preserve the evidence trail of what was imputed (the indicator column)
  even when current evidence says it's not informative. That conclusion is
  dataset-specific and time-bound, not a permanent fact.
- A validated threshold is only valid at the rate it was validated against.
  Encode that as an explicit ceiling, not an implicit assumption someone
  has to remember.

---

## Entry 3: Extending the same analysis to `customer_age` and `distance_from_home`

**Observation.** These two columns (12% and 10% missing) were explicitly
flagged by the design brief as *not yet analyzed* — a caution against
pattern-matching "also has moderate missingness" onto the same MCAR
treatment used for `velocity_score`.

**Analysis.** Ran the identical methodology: fraud-rate split, categorical
driver crosstabs, continuous-driver quartile splits, cross-column
missingness correlation. Both came back flat across every driver tested,
fraud rate flat-to-lower in the missing group for both, near-zero
missingness correlation with any other column.

**Decision.** Same MCAR conclusion, same treatment (median + indicator +
ceiling) — but arrived at through the same evidence-gathering process, not
by assuming the earlier verdict transferred.

**Tradeoffs.** None beyond the time cost of running the analysis twice more
instead of once.

**Lessons learned.**
- A caution like "don't assume, verify" only has value if it's actually
  followed the second and third time, not just the first. The temptation
  to skip the re-check grows precisely when the pattern looks the most
  familiar.

---

## Entry 4: The remaining 9 columns, and formalizing the ceiling rule

**Observation.** 9 more columns had real missingness (2%-9%) and no policy
yet. Two of them (`hour_of_day`, `device_type`, plus `is_first_transaction`,
`is_weekend`, `store_type`) are categorical/binary, not continuous — median
doesn't apply to them.

**Analysis.** Same four checks, but formalized into a runnable script
(`scripts/analyze_missingness.py`) instead of one-off interactive queries,
plus a proper two-proportion z-test for the fraud-rate split instead of
eyeballing a percentage gap. All 9 came back MCAR (max p-value 0.079 on
`num_items`, still above the 0.05 threshold; max cross-column missingness
correlation 0.037 across all 12 analyzed columns).

**Decision.** Continuous columns got median imputation, matching the first
three. Categorical/binary columns got mode imputation instead — the same
default `SimpleImputer(strategy="most_frequent")` already used downstream
in `build_features.py`'s categorical pipeline, made explicit and
rate-guarded at the validation stage instead of happening silently later.
Also formalized the missing-rate ceiling as an explicit rule instead of an
ad hoc judgment call each time: *the smallest multiple of 5 that is >= 2x
the validated missing rate.*

**Tradeoffs.** `ImputationGuardRules`' per-column static methods (three
hand-written, identical except for which spec they called) were replaced
with rules generated from the `IMPUTATION_SPECS` registry — less code, but
one more level of indirection (`_make_missing_rate_guard` builds a closure)
that a reader has to understand before the individual rule names make
sense. Chosen because the alternative — nine more hand-written,
copy-pasted static methods — was a bigger cost: `rules.py` could silently
drift out of sync with `imputation.py`'s registry as it grew.

**Lessons learned.**
- Once a pattern repeats a third time with zero variation in the logic
  (only in the data it closes over), that's the signal to generalize it —
  not before, when you don't yet know if the repetition is real or
  coincidental. Generalizing after two real cases (`velocity_score`,
  `customer_age`+`distance_from_home` together) rather than trying to
  predict the abstraction upfront to `avoid duplication before it existed`
  kept the earlier entries honest about what evidence actually justified.
- A convention followed inconsistently (ceilings that were "roughly double,
  eyeballed" for the first three columns) becomes technical debt the
  moment a fourth column needs one. Write the rule down as a formula once
  it needs to apply more than twice.
- "Documented for repeatability" means a runnable script, not just prose.
  Prose describing a methodology rots — a script either still runs or
  visibly breaks.

---

## Entry 5: The Great Expectations bug — a test suite that proved nothing about the path that mattered

**Observation.** While confirming the Validation stage was genuinely done,
`python -m src.validation.validate_data` — the actual command the Makefile
and any future CI would run — was run by hand. It failed. Every test in
the suite was green.

**Analysis.** `run_great_expectations()` called `gx.from_pandas(df)`, a
pre-1.0 Great Expectations API. The installed version was 1.17.1, which
doesn't expose it. This had presumably been broken since the GX
integration was first written against a different installed version. It
was invisible because *every single test* that called `run_validation()`
passed `include_great_expectations=False` — the one code path that
mattered most for "does this actually work in production" had zero test
coverage, while everything else was thoroughly tested.

**Decision.** Migrated to the GX 1.x Data Context / ExpectationSuite /
Batch API. Then — more importantly — added a test that calls
`run_validation()` on its true default (`include_great_expectations=True`,
no override), plus direct pass/fail tests for `run_great_expectations()`
itself.

**Tradeoffs.** None really — this was a pure bug fix. The only cost was the
time between when it broke and when it was noticed, which was entirely a
function of the coverage gap.

**Lessons learned.**
- **A green test suite only proves what it exercises.** If every test
  disables the one thing you're worried about, you have zero signal on it,
  regardless of how many other tests pass. This is the single most
  important lesson in this log so far.
- The way to find this class of bug is to run the real command, not just
  the tests — CI would have caught this too (once it exists; see Entry 6's
  open item), but manual verification against the actual entrypoint is
  what caught it here, and should be habitual before calling any stage
  "done."
- When a test suite passes a flag to disable a subsystem in every single
  call site, ask why that subsystem doesn't get exercised elsewhere. If
  the answer is "convenience" rather than "this really can't be tested,"
  that's a coverage gap wearing a disguise.

---

## Entry 6: Hardening Training — imbalance-aware metrics, and what they revealed

**Observation.** The project's stated purpose (`CLAUDE.md`) is to
demonstrate handling of rare-event classification, which forces real
choices around evaluation metrics that a balanced-class toy problem never
would. The training stage as it stood computed `accuracy` first,
`f1_score(..., average="weighted")`, and no PR-AUC — none of which are
appropriate defaults for an imbalanced problem, and the model had no
`class_weight` set.

**Analysis.**
- **`class_weight` unset** means `RandomForestClassifier` optimizes as if
  false negatives and false positives on the tiny fraud class cost the
  same as on the large non-fraud class. This doesn't error — it just
  quietly trains a worse fraud detector while looking completely normal.
- **`average="weighted"` on F1** blends both classes' F1 scores
  proportional to their support. On a binary problem where one class is
  ~90% of the data, that means the number reported is dominated by how
  well the model does on the class nobody cares about, diluting exactly
  the signal the metric exists to surface.
- **ROC-AUC alone** is known to look optimistic under class imbalance — a
  huge population of easy true negatives makes the curve look better than
  the model's real usefulness. PR-AUC (average precision) is the standard
  companion metric that doesn't have this blind spot, and CLAUDE.md names
  it explicitly.

**Decision.** Set `class_weight="balanced"` (reweighted once on the full
training set — the standard first choice; `"balanced_subsample"` is the
per-bootstrap-tree variant, unnecessary complexity at this scale of
imbalance and this deliberate to state explicitly rather than leave as a
silent default). Switched precision/recall/F1 to `pos_label=1,
average="binary"` — the fraud class specifically. Added `pr_auc`
(`average_precision_score`) to the metric set. Added regression tests for
both: one confirming `class_weight` is actually configured, one confirming
`f1` differs from what a `"weighted"` average would have produced (proving
the fix actually changes behavior, not just that it doesn't crash).

**What this revealed — the actually important part.** Running the hardened
metrics against the real trained model on real data:

```
precision=0.0000 recall=0.0000 f1=0.0000 pr_auc=0.1114 roc_auc=0.5083 accuracy=0.8971
```

The model predicts zero fraud cases, ever, at the default threshold. Every
predicted probability across the entire 1,400-row test set topped out at
0.39 — never crossing 0.5. ROC-AUC of 0.508 is indistinguishable from a
coin flip. Feature importances are nearly uniform across every column
(0.07-0.15 for numerics, ~0.01 each for one-hot categoricals) — the model
found no feature more useful than any other, which is what "no real
signal" looks like in a trained model. Checked independently against the
raw data: every numeric feature's correlation with `is_fraud` is below
0.05 in magnitude. **This is a property of the dataset, not a bug in
today's change** — `class_weight="balanced"` is confirmed active on the
saved model; the classifier genuinely has nothing learnable to weight.

The old metric set (`accuracy=0.897`, which reads as "89.7% correct,"
alongside a `weighted` F1 that would have scored suspiciously close to
that same number) would have **hidden this completely**. A model that
predicts "not fraud" for every single row scores 89.7% accuracy on a
dataset that's 10.3% fraud, and a weighted F1 mostly reflects performance
on the 89.7% majority class it's also getting right by doing nothing. This
is close to the textbook example of why accuracy is the wrong headline
metric for this class of problem — and it took hardening the metrics to
actually see it, rather than just say it.

**Tradeoffs.** None on the implementation side. But this finding changes
what "harden Training" can mean: fixing the evaluation lens doesn't fix
what it's now correctly showing. `class_weight="balanced"` alone wasn't
enough to produce a usable model on this feature set — that's a modeling /
feature-signal problem, not a metrics problem, and it's outside today's
scope.

**Lessons learned.**
- The whole point of choosing the right metric is that it might tell you
  something you don't want to hear. If hardening a metric never changes
  the story, it probably wasn't doing anything.
- Accuracy on an imbalanced problem isn't just "less informative" than
  precision/recall — it can be actively misleading in the specific
  direction of hiding a non-functional model, because "predict the
  majority class always" is a free, high-accuracy strategy that requires
  learning nothing.
- A model that trains without error and produces a plausible-shaped output
  (probabilities between 0 and 1, predictions in {0,1}) has not been
  shown to work. "Ran successfully" and "learned something real" are
  different claims, and only evaluation — the right evaluation — can tell
  them apart.
- This is not yet a blocker resolved — it's a blocker *found*. The next
  honest step is investigating why (weak features as engineered, a
  fundamentally hard synthetic label, a model class mismatch, or something
  else) before treating any "next stage" as safe to build on top of this
  model's output.

---

## Entry 7: Investigating the signal finding — and treating it as an architecture decision, not an implementation one

**Observation.** Entry 6 found a model with no predictive power, and
flagged "why" as an open item. Four more targeted checks were run against
the raw data to distinguish between the possible causes: a preprocessing
bug, a leakage issue, a model/hyperparameter problem, or a property of the
dataset itself.

**Analysis.**
- **Mutual information** (catches nonlinear/non-monotonic relationships
  Pearson correlation is blind to): essentially zero for every feature
  (max 0.0053, most exactly 0.0000).
- **Decile fraud-rate spread per numeric feature**: flat 8-13% across every
  bin of every column — no feature has a high-risk zone, linear or not.
- **Pairwise interaction check** (`is_first_transaction` x `device_type`):
  still flat within the noise band of the smaller cells — rules out "no
  single feature matters, but a combination does," which a tree ensemble
  should otherwise be able to exploit if it existed.
- **Row-order gap analysis** on the label: mean gap between consecutive
  fraud rows was 9.71, against a theoretical expectation of 9.71 under
  independent Bernoulli(0.103) sampling — a near-exact match, with no
  clustering, periodicity, or drift across the file.

Together these rule out a pipeline bug (the raw file was checked directly,
upstream of any code written for this project) and point to a specific,
falsifiable hypothesis: `is_fraud` was very likely generated independently
of the other 12 columns.

**Decision.** This finding was surfaced to the user, who relayed it to
ChatGPT — the project's designated owner of architecture decisions (see
`CLAUDE.md`'s multi-agent division of labor). ChatGPT's call: reject
Dataset v1 for supervised learning rather than either (a) silently training
against it and reporting misleading numbers, or (b) reverse-engineering the
labels to make a model "work." The investigation and its evidence are
preserved, not discarded — a rejected dataset with a documented reason is
itself evidence of a working data-quality gate.

Two mechanical, non-design pieces followed as implementation:
- `scripts/assess_dataset_signal.py` — formalized the four ad hoc checks
  into a runnable, repeatable script (matching the same "documented for
  repeatability" bar as `scripts/analyze_missingness.py`), writing
  structured evidence to `artifacts/data_quality/signal_analysis.json`.
- `docs/DATASET_ASSESSMENT.md` — the narrative verdict and reasoning,
  including an explicit case for *why this isn't a Great Expectations or
  core validation rule*: "structurally sound" and "useful for the modeling
  task" are different claims, and collapsing them into one PASS/FAIL would
  erase which question actually failed.

What was **not** done: writing a Dataset v2 generator. ChatGPT explicitly
flagged that as its own architecture decision — which features should
influence fraud, how strongly, what interactions, what target prevalence,
how much irreducible noise — and cautioned against jumping straight to
implementation before that's decided. Per the same division of labor that
governs this whole project (architecture decisions aren't mine to make
unilaterally), that boundary was respected rather than worked around.

**Tradeoffs.** Building the generator now would have kept momentum, but at
the cost of embedding an unreviewed set of modeling assumptions (which
features matter, how much) directly into what becomes the project's
canonical dataset — exactly the kind of design decision that's supposed to
get scrutiny before implementation, not after.

**Lessons learned.**
- Investigating *why* is itself a skill distinct from noticing *that*
  something is wrong: each of the four checks here was chosen specifically
  because it could rule out a different class of explanation (bug vs.
  leakage vs. hyperparameters vs. genuine absence of signal), not just
  because more checks are better. A vague "let me look into it" produces
  a worse answer than a checklist designed around "what would each
  possible cause look like, and how would I tell them apart?"
- Not every problem uncovered while implementing is an implementation
  problem to solve. Recognizing that this specific finding crossed from
  "bug to fix" into "design decision to make" — and routing it to the
  role responsible for that decision instead of picking a direction
  myself — is itself the correct engineering behavior the project's
  stated division of labor exists to produce.
- Evidence of a *rejected* option is not waste. The instinct to delete a
  dead end and move on would have thrown away exactly the artifact that
  proves the pipeline's data-quality gate works.

---

## Open items (tracked here, not yet actioned)

- **No CI.** The GX bug (Entry 5) is exactly the kind of regression a
  `pytest`-on-push GitHub Actions workflow would catch automatically.
  Planned per `CLAUDE.md`'s roadmap; not yet built.
- **Feature engineering's redundant imputer.** `build_features.py`'s
  `SimpleImputer` inside the sklearn pipeline is now provably redundant —
  the Validation stage guarantees zero nulls reach it. Not wrong, just
  worth a conscious keep-as-defense-in-depth-or-remove decision rather
  than leaving it unexamined.
- **Dataset v2 generation.** Awaiting the architecture decision on the
  data-generating model (which features influence fraud, interaction
  structure, target prevalence, noise level) per Entry 7 — implementation
  starts once that's defined, not before.
