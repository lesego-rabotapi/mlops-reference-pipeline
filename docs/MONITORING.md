# Monitoring

## Why this document exists

`docs/LOCAL_COMPLETION_GUIDE.md`'s Governance and Operations phase asks
for a document explaining the local Prometheus metrics and how they'd
map to a cloud deployment. Everything below describes metrics that were
actually verified working end to end in this session, not a planned
design — see `docs/ENGINEERING_LOG.md`, Entry 10, for the verification
process itself.

## What's actually exposed

`src/serving/main.py` defines two real metrics, using `prometheus_client`:

| Metric | Type | What it measures |
|---|---|---|
| `predictions_total` | Counter | Total `/predict` requests served, incremented once per successful prediction |
| `prediction_latency_seconds` | Histogram | Wall-clock time spent inside the `/predict` handler's model call, timed with `PREDICTION_LATENCY.time()` around the `.transform()`/`.predict_proba()` call |

Both are served at `GET /metrics` in Prometheus's text exposition format
via `generate_latest()` — no hand-rolled formatting.

## How Prometheus scrapes it

`prometheus.yml` (repo root) defines one scrape job:

```yaml
scrape_configs:
  - job_name: fraud-api
    static_configs:
      - targets: ["api:8000"]
```

`api:8000` is `docker-compose.yml`'s service-name DNS, not `localhost` —
this only resolves inside the compose network, which is why the target
shows as reachable in Prometheus's own UI but not if you tried to curl
that hostname from your host machine directly. Scrape interval is 15s
(`prometheus.yml`'s `global.scrape_interval`).

## The Grafana dashboard

Provisioned as code (`grafana-provisioning/`), not clicked together by
hand and not built via a one-off API call against a container's ephemeral
state — see `docs/ENGINEERING_LOG.md` for the entry documenting why the
API-built version didn't survive a container recreate, and the fix. On
every `docker compose up`, Grafana reads
`grafana-provisioning/datasources/prometheus.yml` and
`grafana-provisioning/dashboards/fraud-api.json` and reproduces one data
source (Prometheus, `http://prometheus:9090`) and one dashboard
(`Fraud API`, uid `ahr7db`) with two panels:

- **Predictions Total** — `predictions_total`
- **Prediction Latency (p50)** — `histogram_quantile(0.5, rate(prediction_latency_seconds_bucket[1m]))`

## Proof this actually works, not just renders

A dashboard panel rendering is not the same claim as a dashboard panel
showing real data. This was verified directly, twice, in two separate
sessions:

1. **First verification** (`ENGINEERING_LOG.md`, Entry 10): queried
   `predictions_total` through Grafana's own datasource-proxy endpoint
   (`/api/datasources/proxy/uid/<uid>/api/v1/query` — the exact path the
   panel itself queries, not a shortcut through Prometheus directly)
   before and after sending 5 real `/predict` requests. Result: `0 -> 5`.
2. **Second verification, a day later**: re-ran the same latency query
   and got `NaN` — not a bug, but `rate()` over a 5-minute window
   correctly reporting "no data," since the container had been running
   14 hours with no recent traffic. Sent 5 fresh requests, re-queried,
   got a real value (`0.0194s` p50). `predictions_total` read `10` at
   that point — the original 5 plus the new 5, confirming the counter
   persisted correctly across a 14-hour gap rather than resetting.

That `NaN` result matters as much as the non-`NaN` ones: it's evidence
the metric is actually being computed live from real scrape data, not
cached or faked. A metric that can never show "no data" isn't measuring
anything real.

## Inspecting it yourself

```bash
docker compose up
```
Grafana's admin credentials are no longer the default — they're read
from `.env` (copy `.env.example`, set a real `GRAFANA_ADMIN_PASSWORD`
before first run; `docker-compose.yml` refuses to start Grafana without
it). Then:
- `http://localhost:8000/health` — API alive
- `http://localhost:9090/targets` — confirm `fraud-api` shows `UP`
- `http://localhost:3000` (login with the credentials from your `.env`)
  — the `Fraud API` dashboard; `/predict` is rate-limited to 1/minute
  per client (see `src/serving/main.py`), so send one `curl` request,
  wait for the window to reset, and repeat to watch the panels move

## Cloud mapping

Per `docs/ARCHITECTURE.md`'s AWS mapping table: Prometheus + Grafana
maps to **CloudWatch Metrics + Dashboards** — same scrape-and-visualize
model, CloudWatch replaces Prometheus as the metrics backend. Concretely,
`predictions_total` and `prediction_latency_seconds` would become custom
CloudWatch metrics published from the FastAPI service (via the
CloudWatch embedded metric format or a small publisher), and the Grafana
dashboard's two panels would become a CloudWatch dashboard with the same
two widgets.

## What isn't monitored here, honestly

- **No alerting.** `docs/ARCHITECTURE.md` explicitly excludes Alertmanager
  — without a continuously running service and someone on-call, alert
  routing config isn't testable or meaningful at this project's scale.
- **No model or data drift monitoring.** The metrics here cover service
  health (is it up, how fast, how much traffic) — not whether the
  model's predictions are still valid against incoming data. Given
  `docs/DATASET_ASSESSMENT.md`'s finding that this model has no real
  predictive signal to begin with, drift monitoring on top of it would
  be measuring the wrong thing; this is a gap worth closing only once
  the model itself is one worth monitoring for drift.
- **No business-level metrics** (e.g., fraud rate over time, false
  positive/negative tracking against ground truth) — this would require
  a feedback loop from real labeled outcomes, which doesn't exist in a
  local, no-live-traffic deployment.
