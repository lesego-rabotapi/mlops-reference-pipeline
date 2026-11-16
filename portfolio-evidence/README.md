# Portfolio Evidence — Provenance Log

Every file in this folder is either (a) a screenshot the user captured and pasted directly
into the chat during live verification on `lesego-hp` (real files, timestamped by upload),
or (b) still outstanding and listed below as a gap. Nothing here is a mockup or a
recreation — if a state wasn't captured as a real file, it's listed as missing, not faked.

## Captured and verified (real files, in this folder)

| File | What it shows | How it was produced |
|---|---|---|
| `monitoring/prometheus-predictions-total-baseline-0.png` | Prometheus query `predictions_total{instance="api:8000",job="fraud-api"}` returning `0` | User screenshot, Prometheus UI, before any successful `/predict` calls landed |
| `monitoring/prometheus-predictions-total-16-plus-terminal.png` | Same query returning `16`, alongside the PowerShell terminal transcript of the curl batches that produced those 16 successful predictions | User screenshot, taken after the curl double-quote fix (see engineering note below) |
| `monitoring/prometheus-target-down-dns-error.png` | Prometheus **Status → Target health** page showing `fraud-api` target `0/1 up`, with the real error `Get "http://api:8000/metrics": dial tcp: lookup api on 127.0.0.11:53: no such host` | User screenshot, taken immediately after the `api` container was stopped via Docker Desktop |

## Known gap: screenshots I personally drove could not be exported

During this session I had computer-use (click-tier) access to Docker Desktop on `lesego-hp` and
used it directly to stop/start the `api` container for the failure/recovery demo. I also took
several `computer_screenshot` captures myself (Docker Desktop's green "running" container list,
the restarted container going green again, Prometheus's target list returning to `1/1 up`).
**Those captures have no file-export path available to me in this session** — the
`computer_screenshot` tool renders inline for verification but has no `save_to_disk` option — so
none of them exist as portable image files. I confirmed each state visually in the moment (and
described what I saw in chat as it happened), but I am not going to claim a file exists when it
doesn't. The three files above are the only real image evidence files that exist right now.

### Still needed from the user (quick to recapture — the stack is still running)

All of these are easy to re-produce right now with `Win+Shift+S` since nothing here requires
tearing anything down:

- [ ] `infrastructure/docker-desktop-containers-running.png` — Docker Desktop container list, all three (`api-1`, `prometheus-1`, `grafana-1`) green/running
- [ ] `application/fastapi-swagger-docs.png` — `http://localhost:8000/docs` (Swagger UI showing `/health`, `/predict`, `/metrics`)
- [ ] `application/predict-response-example.png` — one clean `/predict` request/response (already have this in terminal form inside the `-16-plus-terminal` screenshot, but a dedicated Swagger "Try it out" screenshot reads better for the portfolio)
- [ ] `monitoring/prometheus-target-up-healthy.png` — Target health page showing `fraud-api` `1/1 up` (the recovered/healthy state — this was confirmed live in chat but never captured as a file)
- [ ] `monitoring/grafana-dashboard-fraud-detection-board.png` — the Grafana "Fraud Detection Board" dashboard (`uid=ahr7db`), logged in, showing the live "Total Predictions Served" panel
- [ ] `ci-cd/github-actions-workflow-runs.png` — the Actions tab on `github.com/lesego-rabotapi/mlops-reference-pipeline` showing real run history (green checks or otherwise) — I never got API/gh-CLI access to verify this directly; this screenshot is the only way to close that gap
- [ ] `ci-cd/trivy-scan-output.png` (optional but valuable) — the Trivy step's log output from inside a GitHub Actions run, if one exists — closes the "did Trivy actually run and pass" gap since it could not be replicated in the cloud sandbox (registry egress blocked) or independently confirmed on `lesego-hp`

None of these block the written deliverables (demo script, LinkedIn post, portfolio entry,
final report) — those are done below using only what's already verified — but they materially
strengthen the evidence folder and should be captured before recording the demo video.
