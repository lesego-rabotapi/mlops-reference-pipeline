# Engineering Log

## Purpose

`git log` already records what changed. This records why: the reasoning
behind each decision, what alternatives got considered and rejected, and
what went wrong along the way. The goal is that the thinking behind this
pipeline can be studied and replicated on its own, not just inferred from
the code.

Each entry follows the pattern from
[MENTORSHIP_OPERATING_PLAN.md](MENTORSHIP_OPERATING_PLAN.md): Observation,
Analysis, Decision, Tradeoffs, plus a Lessons Learned section. That last
part is usually the most worth rereading later, since it's the part that
generalizes past this specific repo.

Entries get appended as work happens, not rewritten afterward. If a later
entry contradicts an earlier one, both stay. That's part of the record
too.

---

## Entry 1: Replacing the churn scaffold with the fraud dataset

**Observation.** The repo's actual design brief (`CLAUDE.md`) described a
fraud-detection pipeline already validated against `data/fraud.csv`. The
repo itself was still wired entirely to a customer-churn dataset, schema,
config, rules, and tests all assumed columns like `CreditScore`,
`Geography`, `Exited`. `data/fraud.csv` didn't exist in the repo at all.

**Analysis.** Two explanations were possible: either this was the wrong
repo or branch, or the churn pipeline was leftover scaffolding from an
earlier phase the design brief had already moved past. The mismatch was
too large to guess past. Schema, business rules, and every fixture would
need rewriting either way, and guessing wrong would mean building the
wrong thing twice.

**Decision.** Asked rather than assumed. The user confirmed: replace
churn entirely, fraud is the one real dataset. Once that was confirmed,
the raw CSV was inspected directly (`pandas.read_csv(...).describe()`,
`.isnull().mean()`) rather than trusting the design brief's numbers on
faith. They matched (7,000 rows, 10.3% fraud rate, `velocity_score` at
15% missing), which was itself a useful sign the brief was current.

**Tradeoffs.** Rewiring `SchemaRules`, `FeatureRules`, `feature_config.py`,
and every test fixture in one pass was a bigger change than a single
column fix, but a half-migrated state (some files fraud-shaped, some
still churn-shaped) would have been worse. It would silently pass some
checks against the wrong schema and fail others for confusing reasons.

**Lessons learned.**
- When a task's premise doesn't match what the repo actually looks like,
  stop and say so before writing code against either version. Asking
  costs one message; building against a guessed-wrong premise costs a
  full rewrite.
- Check a design document's factual claims against the real data before
  trusting them for downstream decisions. A brief can describe a
  should-be-true state that isn't the current one.

---

## Entry 2: `velocity_score` missingness — MCAR vs. MNAR

**Observation.** `velocity_score` (15% missing, the highest of any
column) needed an imputation policy. The easy path is `fillna(median)`
and move on.

**Analysis.** Missingness isn't automatically safe to fill. If a column
is missing because of something related to the outcome being predicted
(MNAR, missing not at random), naive imputation doesn't just lose
information, it actively overwrites a real signal with a
plausible-looking lie. The only way to know which case this is: test it.
Cross-tabulated missingness against `is_first_transaction` (would
velocity be structurally uncomputable for first-time customers with no
transaction history?) and against `is_fraud` (is missingness itself
concentrated in fraud cases?).

**Decision.** Missing rate was 14.7% vs. 18.3% across
`is_first_transaction`, a small gap, not the near-100%-for-one-group
split structural uncomputability would produce. Fraud rate was 8.8%
(missing) vs. 10.6% (present), so missingness isn't concentrated in
fraud. Read as MCAR; median imputation applied, with a
`velocity_score_was_missing` indicator column preserved and a
missing-rate ceiling (30%, about 2x the validated 15%) guarding the
assumption against future drift.

**Tradeoffs.** This took real analysis time versus just filling and
moving on. Assuming MCAR without checking would have looked identical in
the short term, and only would have surfaced as a problem if the
assumption were wrong and nobody caught it.

**Lessons learned.**
- "Missing" isn't one category. Test why before choosing how to handle
  it, the two questions can have different, sometimes contradictory
  answers.
- Keep the evidence trail of what was imputed (the indicator column)
  even when current evidence says it's not informative. That conclusion
  is dataset-specific and time-bound, not a permanent fact.
- A validated threshold is only valid at the rate it was validated
  against. Write that down as an explicit ceiling instead of leaving it
  as an assumption someone has to remember.

---

## Entry 3: Extending the same analysis to `customer_age` and `distance_from_home`

**Observation.** These two columns (12% and 10% missing) were flagged by
the design brief as not yet analyzed, a caution against pattern-matching
"also has moderate missingness" onto the same MCAR treatment used for
`velocity_score`.

**Analysis.** Ran the same methodology: fraud-rate split, categorical
driver crosstabs, continuous-driver quartile splits, cross-column
missingness correlation. Both came back flat across every driver tested,
fraud rate flat to lower in the missing group for both, and near-zero
missingness correlation with any other column.

**Decision.** Same MCAR conclusion, same treatment (median, indicator,
ceiling), but reached through the same evidence-gathering process, not
by assuming the earlier verdict carried over.

**Tradeoffs.** None beyond the time cost of running the analysis twice
more instead of once.

**Lessons learned.**
- A caution like "don't assume, verify" only means something if it's
  actually followed the second and third time, not just the first. The
  temptation to skip the recheck grows exactly when the pattern looks
  most familiar.

---

## Entry 4: The remaining 9 columns, and formalizing the ceiling rule

**Observation.** 9 more columns had real missingness (2%-9%) and no
policy yet. Some of them (`hour_of_day`, `device_type`,
`is_first_transaction`, `is_weekend`, `store_type`) are
categorical/binary, not continuous, so median doesn't apply.

**Analysis.** Same four checks, but formalized into a runnable script
(`scripts/analyze_missingness.py`) instead of one-off interactive
queries, plus an actual two-proportion z-test for the fraud-rate split
instead of eyeballing a percentage gap. All 9 came back MCAR (highest
p-value 0.079 on `num_items`, still above the 0.05 threshold; highest
cross-column missingness correlation 0.037 across all 12 analyzed
columns).

**Decision.** Continuous columns got median imputation, matching the
first three. Categorical/binary columns got mode imputation instead, the
same default `SimpleImputer(strategy="most_frequent")` already used
downstream in `build_features.py`'s categorical pipeline, just made
explicit and rate-guarded at the validation stage instead of happening
quietly later. Also wrote the missing-rate ceiling down as an explicit
rule instead of an ad hoc judgment call each time: the smallest multiple
of 5 that's at least 2x the validated missing rate.

**Tradeoffs.** The three hand-written `ImputationGuardRules` static
methods (identical except for which spec they called) got replaced with
rules generated from the `IMPUTATION_SPECS` registry. That's less code,
but it adds a level of indirection (`_make_missing_rate_guard` builds a
closure) a reader has to understand before the individual rule names
make sense. Chose it anyway, because the alternative (nine more
hand-written, copy-pasted static methods) was the bigger cost: `rules.py`
could silently drift out of sync with `imputation.py`'s registry as it
grew.

**Lessons learned.**
- Once a pattern repeats a third time with zero variation in the logic
  (only in the data it closes over), that's the signal to generalize it,
  not before, when you don't yet know if the repetition is real or
  coincidental. Generalizing after two real cases (`velocity_score`, then
  `customer_age` and `distance_from_home` together), rather than trying
  to predict the abstraction upfront, kept the earlier entries honest
  about what evidence actually justified.
- A convention followed inconsistently (ceilings that were "roughly
  double, eyeballed" for the first three columns) turns into technical
  debt the moment a fourth column needs one. Write the rule down as a
  formula once it needs to apply more than twice.
- "Documented for repeatability" means a runnable script, not just
  prose. Prose describing a methodology goes stale silently; a script
  either still runs or visibly breaks.

---

## Entry 5: The Great Expectations bug — a test suite that proved nothing about the path that mattered

**Observation.** While confirming the Validation stage was genuinely
done, `python -m src.validation.validate_data`, the actual command the
Makefile and any future CI would run, was run by hand. It failed. Every
test in the suite was green.

**Analysis.** `run_great_expectations()` called `gx.from_pandas(df)`, a
pre-1.0 Great Expectations API. The installed version was 1.17.1, which
doesn't expose it. This had presumably been broken since the GX
integration was first written against a different installed version. It
stayed invisible because every single test that called
`run_validation()` passed `include_great_expectations=False`, so the one
code path that mattered most for "does this actually work in
production" had zero test coverage while everything else was thoroughly
tested.

**Decision.** Migrated to the GX 1.x Data Context / ExpectationSuite /
Batch API. Then, more importantly, added a test that calls
`run_validation()` on its true default
(`include_great_expectations=True`, no override), plus direct pass/fail
tests for `run_great_expectations()` itself.

**Tradeoffs.** None really, this was a pure bug fix. The only real cost
was the time between when it broke and when it got noticed, which came
down entirely to the coverage gap.

**Lessons learned.**
- A green test suite only proves what it actually exercises. If every
  test disables the one thing you're worried about, you have zero signal
  on it, no matter how many other tests pass. This is probably the
  single most important lesson in this log so far.
- The way to catch this class of bug is to run the real command, not
  just the tests. CI would have caught it too, once it exists (see the
  open items below), but manual verification against the actual
  entrypoint is what caught it here, and it's worth making a habit
  before calling any stage "done."
- When a test suite passes a flag to disable a subsystem at every call
  site, ask why that subsystem doesn't get exercised anywhere else. If
  the honest answer is "convenience" rather than "this genuinely can't
  be tested," that's a coverage gap wearing a disguise.

---

## Entry 6: Hardening Training — imbalance-aware metrics, and what they revealed

**Observation.** The project's stated purpose (`CLAUDE.md`) is to
demonstrate handling of rare-event classification, which forces real
choices around evaluation metrics that a balanced-class toy problem
never would. The training stage as it stood computed `accuracy` first,
`f1_score(..., average="weighted")`, and no PR-AUC, none of which are
good defaults for an imbalanced problem, and the model had no
`class_weight` set.

**Analysis.** With `class_weight` unset, `RandomForestClassifier`
optimizes as if false negatives and false positives on the tiny fraud
class cost the same as on the large non-fraud class. That doesn't error,
it just quietly trains a worse fraud detector while looking completely
normal. `average="weighted"` on F1 blends both classes' F1 scores
proportional to their support, so on a binary problem where one class is
about 90% of the data, the number reported ends up dominated by how well
the model does on the class nobody actually cares about, which dilutes
exactly the signal the metric exists to surface. And ROC-AUC alone is
known to look optimistic under class imbalance: a huge population of
easy true negatives makes the curve look better than the model's real
usefulness. PR-AUC (average precision) is the standard companion metric
that doesn't have this blind spot, and CLAUDE.md names it explicitly.

**Decision.** Set `class_weight="balanced"` (reweighted once on the full
training set, the standard first choice; `"balanced_subsample"` is the
per-bootstrap-tree variant, unnecessary complexity at this scale of
imbalance, worth stating explicitly rather than leaving it as a silent
default). Switched precision/recall/F1 to `pos_label=1,
average="binary"`, the fraud class specifically. Added `pr_auc`
(`average_precision_score`) to the metric set. Added regression tests for
both changes: one confirming `class_weight` is actually configured, one
confirming `f1` differs from what a `"weighted"` average would have
produced, proving the fix actually changes behavior and doesn't just
avoid crashing.

**What this revealed.** Running the hardened metrics against the real
trained model on real data:

```
precision=0.0000 recall=0.0000 f1=0.0000 pr_auc=0.1114 roc_auc=0.5083 accuracy=0.8971
```

The model predicts zero fraud cases, ever, at the default threshold.
Every predicted probability across the entire 1,400-row test set topped
out at 0.39, never crossing 0.5. A ROC-AUC of 0.508 is indistinguishable
from a coin flip. Feature importances are nearly uniform across every
column (0.07-0.15 for numerics, about 0.01 each for one-hot
categoricals), meaning the model found no feature more useful than any
other, which is what "no real signal" looks like in a trained model.
Checked independently against the raw data: every numeric feature's
correlation with `is_fraud` is below 0.05 in magnitude. This is a
property of the dataset, not a bug in this change,
`class_weight="balanced"` is confirmed active on the saved model; the
classifier genuinely has nothing learnable to weight.

The old metric set (`accuracy=0.897`, which reads as "89.7% correct,"
alongside a `weighted` F1 that would have scored suspiciously close to
that same number) would have hidden this completely. A model that
predicts "not fraud" for every single row scores 89.7% accuracy on a
dataset that's 10.3% fraud, and a weighted F1 mostly just reflects
performance on the 89.7% majority class it's also getting right by doing
nothing. It's close to the textbook example of why accuracy is the wrong
headline metric for this kind of problem, and it took hardening the
metrics to actually see it, rather than just say it.

**Tradeoffs.** None on the implementation side. But this finding changes
what "harden Training" can mean: fixing the evaluation lens doesn't fix
what it's now correctly showing. `class_weight="balanced"` alone wasn't
enough to produce a usable model on this feature set. That's a modeling
or feature-signal problem, not a metrics problem, and it's outside this
entry's scope.

**Lessons learned.**
- The whole point of choosing the right metric is that it might tell you
  something you don't want to hear. If hardening a metric never changes
  the story, it probably wasn't doing anything.
- Accuracy on an imbalanced problem isn't just less informative than
  precision and recall, it can actively mislead in the specific
  direction of hiding a non-functional model, since "predict the
  majority class always" is a free, high-accuracy strategy that requires
  learning nothing.
- A model that trains without error and produces plausible-shaped output
  (probabilities between 0 and 1, predictions in {0,1}) hasn't been
  shown to work. "Ran successfully" and "learned something real" are
  different claims, and only the right evaluation can tell them apart.
- This isn't a blocker resolved, it's a blocker found. The honest next
  step is investigating why (weak features as engineered, a
  fundamentally hard synthetic label, a model class mismatch, or
  something else) before treating any later stage as safe to build on
  top of this model's output.

---

## Entry 7: Investigating the signal finding — and treating it as an architecture decision, not an implementation one

**Observation.** Entry 6 found a model with no predictive power and
flagged "why" as an open item. Four more targeted checks were run
against the raw data to distinguish between the possible causes: a
preprocessing bug, a leakage issue, a model or hyperparameter problem, or
a property of the dataset itself.

**Analysis.**
- Mutual information (catches nonlinear or non-monotonic relationships
  Pearson correlation is blind to): essentially zero for every feature
  (max 0.0053, most exactly 0.0000).
- Decile fraud-rate spread per numeric feature: flat 8-13% across every
  bin of every column, no feature has a high-risk zone, linear or not.
- Pairwise interaction check (`is_first_transaction` x `device_type`):
  still flat within the noise band of the smaller cells, ruling out "no
  single feature matters, but a combination does," which a tree ensemble
  should otherwise be able to exploit if it existed.
- Row-order gap analysis on the label: mean gap between consecutive
  fraud rows was 9.71, against a theoretical expectation of 9.71 under
  independent Bernoulli(0.103) sampling, a near-exact match, with no
  clustering, periodicity, or drift across the file.

Together these rule out a pipeline bug (the raw file was checked
directly, upstream of any code written for this project) and point to a
specific, falsifiable hypothesis: `is_fraud` was very likely generated
independently of the other 12 columns.

**Decision.** This finding was surfaced to the user, who relayed it to
ChatGPT, the project's designated owner of architecture decisions (see
`CLAUDE.md`'s multi-agent division of labor). ChatGPT's call: reject
Dataset v1 for supervised learning rather than either silently training
against it and reporting misleading numbers, or reverse-engineering the
labels to make a model "work." The investigation and its evidence got
kept, not discarded, a rejected dataset with a documented reason is
itself evidence of a working data-quality gate.

Two mechanical pieces followed from there:
- `scripts/assess_dataset_signal.py` formalizes the four ad hoc checks
  into a runnable, repeatable script (matching the same "documented for
  repeatability" bar as `scripts/analyze_missingness.py`), writing
  structured evidence to `artifacts/data_quality/signal_analysis.json`.
- `docs/DATASET_ASSESSMENT.md` has the narrative verdict and reasoning,
  including why this isn't a Great Expectations or core validation rule:
  "structurally sound" and "useful for the modeling task" are different
  claims, and collapsing them into one PASS/FAIL would erase which
  question actually failed.

What didn't happen: writing a Dataset v2 generator. ChatGPT explicitly
flagged that as its own architecture decision, which features should
influence fraud, how strongly, what interactions, what target
prevalence, how much irreducible noise, and cautioned against jumping
straight to implementation before that's decided. Per the same division
of labor that governs this whole project, architecture decisions aren't
mine to make unilaterally, so that boundary got respected rather than
worked around.

**Tradeoffs.** Building the generator now would have kept momentum, but
at the cost of embedding an unreviewed set of modeling assumptions
(which features matter, how much) directly into what becomes the
project's canonical dataset, exactly the kind of design decision that's
supposed to get scrutiny before implementation, not after.

**Lessons learned.**
- Investigating why is its own skill, separate from noticing that
  something is wrong. Each of the four checks here got picked because it
  could rule out a different class of explanation (bug vs. leakage vs.
  hyperparameters vs. genuine absence of signal), not just because more
  checks are better. A vague "let me look into it" produces a worse
  answer than a checklist built around "what would each possible cause
  look like, and how would I tell them apart?"
- Not every problem uncovered while implementing is an implementation
  problem to solve. Recognizing that this specific finding crossed from
  "bug to fix" into "design decision to make," and routing it to the
  role responsible for that decision instead of picking a direction
  unilaterally, is itself the correct engineering behavior the project's
  division of labor is supposed to produce.
- Evidence of a rejected option isn't waste. Deleting a dead end and
  moving on would have thrown away exactly the artifact that proves the
  pipeline's data-quality gate works.

---

## Entry 8: `build_features.py`'s `SimpleImputer` — keep, as defense-in-depth

**Observation.** Entry 7 flagged `build_features.py`'s `SimpleImputer`
steps (median for numerics, most-frequent for categoricals, inside the
sklearn `ColumnTransformer`) as provably redundant: the Validation stage
(`src/validation/imputation.py`) already fills every column with an
evidence-backed, per-column policy and guarantees zero nulls reach
`VALIDATED_DATASET_PATH`. On the current pipeline, `SimpleImputer.fit()`
never sees a null and is a no-op in practice.

**Decision. Keep it.** Two reasons:
1. It's a different trust boundary than validation, at negligible cost.
   `build_features.py` loads `VALIDATED_DATASET_PATH` directly
   (`load_validated_data`) and doesn't re-run `src/validation`'s checks.
   If that file is ever read from a stale run, edited by hand, produced
   by a future pipeline path that skips validation, or a new column gets
   added to `NUMERIC_FEATURES`/`CATEGORICAL_FEATURES` without a matching
   `ImputationSpec`, the feature stage would otherwise call `.fit()` on a
   column with unexpected nulls and raise deep inside sklearn instead of
   failing predictably. A no-op imputer costs nothing measurable at this
   dataset's size; removing it trades a negligible runtime cost for a
   silent assumption ("upstream always ran and always covers every
   column") that isn't enforced anywhere in this file.
2. It matches the file's own stated design decisions. The module
   docstring already lists median imputation for numerics, robust
   against outliers in production data, as a deliberate choice
   independent of whatever Validation does upstream. `build_features.py`
   is written to hold up on its own, not only when combined with
   Validation's current behavior.

**Documentation added.** `build_preprocessor()`'s docstring in
`src/features/build_features.py` now states explicitly that the imputers
are a defense-in-depth guard against an already-validated input, not the
primary null-handling mechanism (that's Validation's job, see
`src/validation/imputation.py` and `docs/MISSINGNESS_ANALYSIS.md`), so a
future reader doesn't mistake it for dead code again without checking
this entry.

**What would change the answer.** If this imputer step ever fires in
practice (logged or observed filling a real null on production data),
that's a signal the Validation stage's guarantee has been bypassed
somewhere and needs investigating directly, not a reason to keep the
`SimpleImputer` doing double duty.

---

## Entry 9: Final decision — stay on Dataset v1, reject PaySim as a pivot

**Observation.** ChatGPT proposed PaySim, a well-known public synthetic
fraud dataset (6.36M rows, 0.13% fraud rate), as a Dataset v2 candidate.
Direct inspection confirmed it has real signal `fraud_raw.csv` never
had: fraud occurs only in `TRANSFER`/`CASH_OUT` transactions (0%
elsewhere), plus a genuine `amount` decile spread.

**Analysis.** PaySim's schema shares no columns with `fraud_raw.csv`,
has zero missingness (which makes the twelve MCAR-backed imputation
policies in `src/validation/imputation.py` inapplicable), and carries far
more extreme class imbalance (0.13% vs. 10.3%). Adopting it isn't a
dataset swap, it's the same scope of rework Entry 1 did for churn to
fraud, applied again: `SchemaRules`, `FeatureRules`, `feature_config.py`,
and training's imbalance handling would all need rebuilding around a
different problem shape.

**Decision.** Keep Dataset v1. The rewrite PaySim requires is a
legitimate future project, not a fix for this one. Pivoting to it now
would mean abandoning a finished, working pipeline to restart a
structurally different one for the sake of a "real" model result. Full
comparison, including what each path costs and produces, is in
`docs/DATASET_ASSESSMENT.md`'s "Decision (resolved)" section.

**Tradeoffs.** This closes off ever training a working fraud classifier
on this repo's current data, that ceiling was already set by Entry 7's
signal assessment, and this decision just confirms it's being accepted
rather than worked around. In exchange, the project stays finished and
demonstrable today instead of turning into a second half-built pipeline.

**Lessons learned.**
- A dataset with a stronger property (real signal) isn't automatically
  the right move. It can cost more than it's worth if adopting it means
  rebuilding work that already runs correctly on a different structure.
- Framing the deliverable honestly (a pipeline that correctly identified
  its data couldn't support the task, not a working classifier) is what
  makes staying on the "worse" dataset defensible instead of a
  compromise quietly glossed over.

---

## Entry 10: Containerizing the inference API — a smooth build, and what almost looked like a bug but wasn't

**Observation.** Built `Dockerfile`, `prometheus.yml`, and
`docker-compose.yml` for the already-working `src/serving/` API, then
verified the full stack (API, Prometheus, Grafana) end to end, including
proving a Grafana panel actually reflects live traffic rather than just
rendering. Worth logging honestly what did and didn't go wrong, per this
document's own principle: entries record what actually happened, not a
manufactured story to fill the template.

**Analysis.** Nothing here broke in a way that needed a real fix. Three
things were still worth noting:

1. Both `docker build` and the first `docker compose up` ran past this
   session's command timeouts (5 minutes, then 3 minutes) and had to be
   backgrounded. Not a bug -- `requirements.txt` pulls in mlflow, sklearn,
   pandas, and matplotlib, and the first `docker compose up` also had to
   pull the `prom/prometheus` and `grafana/grafana` images cold. The
   final image is 1.32GB. Slow, not broken, but worth knowing in advance
   rather than being surprised by it.
2. Right after `docker compose up` reported the `api` container as `Up`,
   `docker compose logs api` came back completely empty for a moment --
   which looks exactly like a silent startup failure. It wasn't: uvicorn
   and the manifest-hash-check logging simply hadn't been written yet at
   the exact instant `docker compose ps` was checked. Rechecking a few
   seconds later showed the same clean startup sequence confirmed earlier
   in the standalone container test. Worth flagging specifically because
   this is the kind of race that could send someone chasing a startup bug
   that was never there.
3. No `.dockerignore` existed yet, so the first build would have sent the
   full build context, including `.venv/` (hundreds of MB), to the Docker
   daemon even though the `Dockerfile` never copies it in. Added one
   before building, scoped to the same kind of thing `.gitignore` already
   excludes plus `tests/`, `docs/`, and `.git/`.

**Decision.** Verified "the Grafana panel actually moves" the same way
the metric itself gets read, not by trusting that it should: queried
`predictions_total` through Grafana's own datasource-proxy API
(`/api/datasources/proxy/uid/.../api/v1/query`, the exact path the panel
uses) before and after sending five real `/predict` requests through the
running compose stack, and confirmed `0 -> 5` after the next scrape
interval. Did the same for the latency panel's query
(`histogram_quantile(0.5, rate(prediction_latency_seconds_bucket[5m]))`),
which returned a real, non-zero value once traffic existed. A dashboard
screenshot would have shown a render; querying the same path Grafana
itself queries proves the data behind it is real and moving.

**Tradeoffs.** None worth noting -- this genuinely was the straightforward
case. The Dockerfile, compose file, and Prometheus config all worked on
the first attempt against the design from the earlier plan, largely
because the inference API underneath had already been built and
manually verified in the previous session before any of this started.

**Lessons learned.**
- Not every entry needs a dramatic failure to be worth writing. A clean
  build is itself useful evidence that the design work done beforehand
  (the plan, the manifest-hash check, the already-tested API) paid off --
  logging that plainly is more honest than inventing friction that wasn't
  there.
- A container reporting "Up" and a container's application actually
  having logged anything are two different moments, sometimes by a
  couple of seconds. Don't read an empty log immediately after "Up" as a
  failure signal without rechecking.
- When verifying a dashboard shows real data, query through the same
  path the dashboard itself uses (Grafana's proxy, not a shortcut through
  the data source directly) -- it's the difference between "the numbers
  exist somewhere" and "the panel would actually show this."

---

## Entry 11: Adding Trivy to CI — a real CRITICAL finding, and a real breaking change from fixing it

**Observation.** `.github/workflows/ci.yml` ran install and tests but had
no container security scan, despite `docs/PROJECT_SCOPE.md` naming Trivy
explicitly as part of this project's CI. Added a build-then-scan step and
ran it locally first against the real `fraud-api:local` image before
trusting it in CI.

**Analysis.** The scan wasn't theoretical -- it found 46 real HIGH/CRITICAL
vulnerabilities with available fixes, split across OS packages (3, from
the `python:3.12-slim` base image) and Python dependencies (43, from
`requirements.txt`). One was worth acting on immediately:
`mlflow==3.12.0` carries CVE-2026-64849, a CRITICAL unauthenticated SSRF
in webhook delivery (`_validate_webhook_url` bypassed via an unvalidated
path), fixed in `3.15.0`. `docker build` succeeding had told us nothing
about this -- the image ran, the tests passed, and none of that surfaced
a known, published, actively-exploitable vulnerability sitting in a
pinned dependency.

**Decision.** Scoped the CI gate to `severity: HIGH,CRITICAL` with
`exit-code: 1` and `ignore-unfixed: true`. Scanning every severity sounds
more thorough but produces a list long enough that a team learns to
ignore the check entirely; gating only on the two severities worth
actually stopping a deploy over keeps the check meaningful.
`ignore-unfixed` skips vulnerabilities with no available patch, since
failing a build over something nobody can act on today just trains
people to bypass the gate rather than fix anything.

Then upgraded `mlflow`/`mlflow-skinny`/`mlflow-tracing` from `3.12.0` to
`3.15.2` (latest, past the `3.15.0` fix line) to close the CRITICAL
finding. This broke `configure_mlflow()` immediately: `mlflow>=3.15`
hard-refuses the filesystem tracking backend
(`MLflowException: The filesystem tracking backend ... is in maintenance
mode`) that every training run and test in this project depends on --
the exact deprecation warning every test run had already been printing
(`FutureWarning: The filesystem tracking backend ... is deprecated as of
February 2026`) had hardened into a real error. Fixed by setting
`MLFLOW_ALLOW_FILE_STORE=true` in `configure_mlflow()` rather than adding
a SQLite tracking backend: this project is deliberately local-first with
no infra beyond what solves a real problem
(`docs/PROJECT_SCOPE.md`), and the filesystem backend still works fine at
this project's scale -- opting into the documented escape hatch keeps
that positioning intact instead of quietly growing a new dependency to
route around a warning.

**Tradeoffs.** Upgrading a pinned dependency to fix one CVE risks pulling
in unrelated breaking changes, which is exactly what happened here, just
not in the vulnerable code path itself. Verified the fix didn't silently
change anything else: 80/80 tests pass, and a real end-to-end training
run produces the identical, previously-documented metrics
(`roc_auc=0.5083`, same deterministic model) -- the upgrade only changed
what CVE-2026-64849 and the tracking-backend deprecation do, nothing
about the model or pipeline's actual behavior.

**Lessons learned.**
- A CI check that never fires against real content isn't verified, it's
  assumed. Running Trivy locally against the actual built image, not
  just wiring the YAML and trusting it, is what turned "this should
  catch vulnerabilities" into "this caught 46 real ones, one of them
  worth fixing today."
- Fixing a security finding by bumping a version is not automatically
  safe just because the CVE itself is unrelated to your code path. Treat
  a dependency upgrade for a CVE fix with the same verification bar as
  any other change: run the real tests, run the real pipeline, and
  compare output to what was true before.
- A scanner surfacing more than you can act on right now (46 findings,
  one CRITICAL) is normal, not a sign to fix everything at once. Fixing
  the one with real severity and an available patch, and leaving the
  rest as a known, visible backlog rather than either ignoring the scan
  or trying to zero it out in one pass, is the sustainable version of
  this practice.

---

## Entry 12: Triaging the remaining 42 HIGH findings — 41 fixed, 1 genuinely blocked

**Observation.** Entry 11 fixed the one CRITICAL finding (`mlflow`) and
left 42 HIGH findings open across 3 OS packages and 8 Python packages,
tracked as an open item rather than ignored. Triaged all of them: checked
what depends on each vulnerable package, upgraded what could be upgraded
safely, and verified the result against the real image, not just against
`pip`'s dependency resolver saying yes.

**Analysis.** None of the 8 Python packages are imported directly by this
project's own code -- `GitPython`, `sqlparse`, and `starlette` come in
through `mlflow-skinny`; `aiohttp` and `cryptography` through `mlflow`;
`pillow` through `matplotlib`; `mistune` through `great_expectations`;
`pyasn1` through `google-auth`. That matters for risk: an upgrade here
can only break something by breaking *their* usage of the package, not
ours directly, but it can still break behavior we depend on through them
(this is exactly what happened with `starlette`, see below). The 3 OS
findings (`libssl3t64`, `openssl`, `openssl-provider-legacy`) were all one
Debian security update the base image's snapshot hadn't picked up yet --
not something `pip` touches at all.

Upgrading `cryptography` to its lowest reported fix version
(`>=48.0.1`) without a ceiling immediately caught a real conflict `pip`
itself flagged: it resolved to `50.0.1`, but `mlflow==3.15.2` requires
`cryptography<50`. Re-pinned to `49.0.0` -- still past every fix boundary
Trivy reported (`48.0.1`), still inside mlflow's declared constraint.

`starlette` was the one requiring real verification rather than trusting
`pip`: FastAPI declares `starlette>=0.46.0` with no upper bound, so `pip`
happily jumped `0.52.1 -> 1.6.0`, a major version bump. Tested against
both the in-process `TestClient` (all 6 serving tests pass) and a real
running `uvicorn` server hit with actual `curl` requests, since
`TestClient` runs in-process and can mask ASGI-server-level breakage that
only shows up against a real server. Both came back clean, and produced
the identical `fraud_probability: 0.18` every prior real-server test in
this project has produced for the same input.

**Decision.** Upgraded 7 of 8 Python packages
(`GitPython`, `aiohttp`, `pillow`, `mistune`, `pyasn1`, `sqlparse`,
`starlette`) plus `cryptography` (capped at `<50` per mlflow's own
constraint) in `requirements.txt`. Added `RUN apt-get update && apt-get
upgrade -y` to the `Dockerfile`, right after `FROM`, to close the 3 OS
findings -- the base image's own security patches, not anything `pip`
manages.

One finding stays open on purpose: `cryptography`'s remaining CVE
(CVE-2026-69247) is fixed in `50.0.0`, but `mlflow==3.15.2` pins
`cryptography<50`. Forcing `50.0.0` would break `pip check` and risk
mlflow's actual behavior for a dependency this project doesn't call
directly -- not a trade worth making today. This is a real, verified
constraint, not a shortcut: confirmed via `pip install
"cryptography>=48.0.1,<50"` and `pip check` reporting "No broken
requirements found" at `49.0.0`, versus a real conflict at `50.0.1`.

**Tradeoffs.** Rebuilt the image and reran the full test suite, a real
training run was not needed here (none of these 8 packages touch
training/feature logic), but the serving path specifically needed the
real-server check given the `starlette` major bump. Final scan: **44
findings -> 1**, and that 1 is a documented, upstream-blocked wait rather
than an unexamined gap.

**Lessons learned.**
- "Nothing in `src/` imports this package" lowers the risk of an
  upgrade, it doesn't eliminate it. `starlette` proved that: FastAPI's
  behavior on top of it is exactly the thing that could have broken, and
  did need real verification (TestClient plus a real server), not an
  assumption that "we don't call it directly" meant "safe to skip
  testing."
- When `pip`'s resolver picks a version that conflicts with another
  package's declared constraint, that's a signal to worry about, not a
  warning to scroll past. `pip check` after every batch upgrade is what
  caught the `cryptography` conflict before it became a runtime problem
  instead of an install-time one.
- A security backlog doesn't have to hit zero to be honest. One
  documented, upstream-blocked finding with a clear reason is a
  completely different thing from 42 unexamined ones -- the goal was
  triage, not a fabricated clean scan.

---

## Entry 13: Grafana state that didn't survive a container recreate, and `/predict` with no rate limit

**Observation.** An independent review session flagged three real gaps:
the live Grafana dashboard showed one panel while `docs/MONITORING.md`
documented two, Grafana was still running on the image's default
`admin`/`admin` login, and `/predict` had no rate limiting at all --
anyone (or anything mis-firing in a loop) could hit the model endpoint
as fast as the network allowed.

**Analysis.** The dashboard drift traced back to how it was built in the
first place (Entry 10): a live call against Grafana's HTTP API, against
a container with no persisted volume for `/var/lib/grafana`. That's not
a bug in the dashboard JSON, it's a bug in where the state lived --
every field in it, including the admin password, existed only in one
specific container's writable layer and was gone the moment that
container was recreated instead of just restarted. The default-password
finding was the same root cause wearing a different hat: nothing in
`docker-compose.yml` ever set a real password, so Grafana fell back to
its own default on every fresh container. `/predict` having no rate
limit wasn't a drift problem, just a genuine gap -- the endpoint does a
real model call per request and had nothing between it and the network.

**Decision.** Replaced the one-off API build with provisioning-as-code:
`grafana-provisioning/datasources/prometheus.yml` and
`grafana-provisioning/dashboards/{dashboard-provider.yml,fraud-api.json}`,
mounted read-only into the container and matched exactly to what
`MONITORING.md` already (correctly) described -- same uid (`ahr7db`),
same two panels -- so the fix is "make the doc true" rather than
"rewrite the doc." Added a named `grafana-data` volume so
`/var/lib/grafana` survives a `docker compose down`/`up` even without
provisioning. Moved the admin password into `.env` (gitignored, real
value generated with `secrets.token_urlsafe`) with `.env.example`
committed as the template, and `docker-compose.yml` reading it via
`${GRAFANA_ADMIN_PASSWORD:?set GRAFANA_ADMIN_PASSWORD in .env}` --
missing the var fails the compose file outright instead of silently
falling back to Grafana's own default, which is the failure mode that
caused this finding in the first place.

For rate limiting, added `slowapi` (`Limiter` keyed on
`get_remote_address`, since this API has no auth layer to key on
instead) and capped `/predict` specifically at `10/minute` --
`/health` and `/metrics` stay unthrottled since Prometheus scrapes
`/metrics` on its own schedule and shouldn't be able to trip a limit
meant for external callers. Added `test_predict_is_rate_limited`
(10 requests succeed, the 11th gets a 429) plus an autouse
`_reset_rate_limiter` fixture, since `Limiter`'s request counts live in
module-level storage on `serving.app`, not per-`TestClient` -- without
the reset, whichever test happened to run after enough accumulated
`/predict` calls would start failing on traffic that wasn't its own.

**Tradeoffs.** `10/minute` is a judgment call, not a measured production
number -- there's no real traffic pattern to size it against yet. It's
picked to be generous enough not to interfere with a demo or the test
suite while still being tight enough to actually trigger a 429 in a
quick manual check, which is what this project needs it to prove right
now. Ran the full suite after both changes: 81/81 (80 prior +
`test_predict_is_rate_limited`).

**Lessons learned.**
- State built via a live API call against a container is exactly as
  durable as that specific container -- if the fix the first time is
  "call the API again," the actual bug (no persisted volume, no
  provisioning) is still there waiting for the next `docker compose
  down`. Provisioning-as-code turns "reproduce it if you notice it's
  gone" into "it can't be gone."
- `${VAR:?message}` in compose files is worth reaching for over a plain
  `${VAR}` any time the fallback behavior (silently using nothing, or
  an image's own default) is itself the security problem you're trying
  to close.
- A rate limiter implemented as a module-level singleton needs the same
  test-isolation attention as any other shared global -- it doesn't
  announce itself as shared state the way a database connection does,
  but it behaves like one across a test session.

---

## Open items (tracked here, not yet actioned)

- **`cryptography` CVE-2026-69247** (HIGH, fixed in `50.0.0`) stays open:
  `mlflow==3.15.2` pins `cryptography<50`. Revisit once mlflow relaxes
  that constraint or this project's mlflow dependency changes.
- **`docs/GOVERNANCE.md`.** Per `LOCAL_COMPLETION_GUIDE.md`'s Governance
  and Operations phase, this doesn't exist yet. `MONITORING.md` now
  exists and describes real, provisioned, verified state (Entry 13).
- **`10/minute` on `/predict` is unvalidated against real traffic.**
  Picked as a reasonable demo/test value, not measured against any
  actual usage pattern -- revisit if this project ever serves real
  traffic.
