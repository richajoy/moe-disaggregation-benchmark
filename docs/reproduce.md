# Reproduce

Back to the [overview](../README.md).

## Re-derive

**Re-derive every number from the published data** (Python 3 standard library; matplotlib only for the charts):

```bash
python3 scripts/derive.py              # rankings, frontiers, TABLES.md; self-checks SLO-ok and ITL p90 against data/requests/
python3 scripts/charts.py              # figures/<node>/*.png
python3 scripts/anim_slo_frontier.py   # figures/anim/anim-slo-frontier.svg
```

## Re-run the benchmark

**Re-run the benchmark** on a single Kubernetes GPU node with the llm-d 1.0 router installed:

1. Start the in-cluster load generator: `kubectl apply -f scripts/aiperf-pod-llmd.yaml` (edit the host results path), then copy `scripts/sweep3.sh` into the pod.
2. Aggregated and P/D sweeps: `REPO_DIR=$PWD scripts/queue.sh agg-dp4 agg-4x-tp1 pd-1p3d-tp1 pd-2p2d-tp1 pd-2p1d-dtp2-bs16 pd-2p1d-dtp2-bs64`. P/D manifests come from [`deploy/gen-pd.py`](../deploy/gen-pd.py) (`DECODE_TP=2 python3 deploy/gen-pd.py 2 1 --block-size=64`), aggregated ones from [`deploy/render.sh`](../deploy/render.sh).
3. Transfer knobs, KV levers and agentic load: [`scripts/knobs.sh`](../scripts/knobs.sh), [`scripts/offload.sh`](../scripts/offload.sh), [`scripts/agentic.sh`](../scripts/agentic.sh). Router profiles for selective P/D and flow control are in [`deploy/router/`](../deploy/router).
4. Server-side signals: run [`scripts/scrape.sh`](../scripts/scrape.sh) beside each sweep, then [`scripts/layers.py`](../scripts/layers.py) (stage timings), [`scripts/nixl.py`](../scripts/nixl.py) (transfer statistics) and [`scripts/hits.py`](../scripts/hits.py) (prefix hits).
5. Turn each run directory into published form: `python3 scripts/build_points.py --node h100 --gpus 4 --config <name> <run-dir>`, then `python3 scripts/derive.py`.

Environment placeholders: `REPO_DIR` (this repo on the node), `BENCH_RESULTS` (host results directory, default `/var/lib/bench-results`), `HF_CACHE` (host Hugging Face cache, default `/var/lib/hf-cache`), and a `hf-token-secret` Secret for the tokenizer download.

## Limits

- One MoE model; one node per GPU type; synthetic, fixed-length, closed-loop load (concurrency is requests in flight, not users).
- Single runs except the H100 block-64 P/D layout; differences under about 10% are ties. The concurrency ladder (16, 64, 256, 512, 1,024) is coarse.
- **P/D pods never shard experts**; the best aggregated layouts do. The verdict is "TP1/TP2 P/D vs expert-sharded aggregated".
- FP8 KV output quality was not checked.
- The A100 runs used a 2,048-token prefill budget against 8,192 on H100.
- Dollar figures appear only for H100, at an assumed $2.00/GPU-hour reference rate ([method 6](00-method.md#6-cost)).

Full list: [what was not measured](08-not-measured.md).
