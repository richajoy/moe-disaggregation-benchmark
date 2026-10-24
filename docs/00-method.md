# How the benchmark works

> **What you get from this page:** exactly what was run, how every number is computed, what "good" means for each metric and who set that bar, and which caveats apply to every other page.

## The answer first

- **One model, two nodes, never mixed.** Qwen3-30B-A3B (a mixture-of-experts model: 30.5B parameters, about 3B active per token) in BF16 on vLLM 0.32.0, served on one 4x H100 SXM node and, separately, one 8x A100 SXM node. H100 and A100 numbers are never compared with each other.
- **Closed-loop, fixed-length load.** AIPerf keeps `c` requests in flight (c = 16, 64, 256, 512, 1,024) with zero think time, on four shapes: prompt-heavy (4,000 tokens in, 200 out), chat (1,000 in, 500 out), prefix (shared 2,000-token prefixes plus 300 unique tokens, 200 out) and agentic (multi-turn sessions).
- **One SLO everywhere:** a request is "good" if its time to first token (TTFT) is at most 2 s **and** its mean inter-token latency (ITL) is at most 50 ms. **Goodput** is good requests per second per GPU. The headline ranking is goodput at the best load where at least 90% of requests are good.
- **p90 is the headline percentile.** Latency tables show p50 / **p90** / p99. Streaming speed (tok/s per user) is `1000 / ITL p90`, the slow tail.
- **Warm-up requests are excluded** from every percentile, SLO share and goodput number. Every value was recomputed from the per-request records, which ship in [`data/requests/`](../data/requests).
- **Every GPU counts.** Per-GPU numbers divide by all GPUs the layout uses, prefill and decode together.

## Evidence

### 1. Hardware and software

| Item | H100 node | A100 node |
|---|---|---|
| GPUs | 4x H100 80GB SXM (HBM3), NVLink | 8x A100 80GB SXM4, NVSwitch |
| Engine | vLLM 0.32.0 (`vllm/vllm-openai:v0.32.0`), PyTorch 2.13 / CUDA 13.0, NIXL 1.5.0 | same image |
| Chunked-prefill budget vLLM chose | **8,192 tokens per step** | **2,048 tokens per step** (vLLM's GPU-dependent default; not set by the scripts) |
| Router | llm-d 1.0 router (EPP plus Envoy, standalone chart), P/D routing sidecar 1.0 | same |
| Other stacks | n/a | NVIDIA Dynamo 1.5.1 (vLLM 0.28 runtime) for the router-accounting and planner tests |
| Orchestration | Kubernetes (k3s 1.34), single node | same |
| Load generator | AIPerf 0.12.0, in-cluster pod, `/v1/completions` streaming (chat endpoint for agentic) | same |

llm-d 1.0 is tested by its maintainers with vLLM 0.31.0; this benchmark used 0.32.0. The AIPerf client carries one small patch so its GPU-telemetry collector accepts a lowercase `hostname` label ([`scripts/aiperf-pod-llmd.yaml`](../scripts/aiperf-pod-llmd.yaml)); it touches telemetry only, not request timing.

**The prefill budget differs by node.** Same scripts, flags and image, but vLLM compiled the A100 arms for 2,048-token prefill chunks and the H100 arms for 8,192. Every A100 prefill-queue time and per-prefill-GPU rate is conditional on that. It is one more reason the two nodes are never compared.

### 2. The model and why KV capacity drives most results

The KV cache stores keys and values for every token a request has seen. Its size per token is fixed by the model:

```
KV bytes per token = layers x 2 (K and V) x KV heads x head dim x bytes per value
                   = 48 x 2 x 4 x 128 x 2 (BF16) = 98,304 bytes (half that in FP8)
```

A 4,000-token prompt is therefore 393 MB of KV, the size vLLM's NIXL connector measured per transfer. How many tokens fit depends on how the weights are placed (vLLM startup logs, [`data/server-metrics.json`](../data/server-metrics.json)):

| Placement (H100) | KV tokens | Prompt-heavy requests that fit (4,200 tokens each) |
|---|---|---|
| One GPU holds the whole model (TP1) | 124,528 per pod | 29 |
| Experts split across 4 GPUs (`--data-parallel-size 4`, "DP4") | 542,912 per GPU | 129 per GPU |
| Two GPUs share one decode instance (TP2) | 862,480 per instance | 205 |

A TP1 pod spends about 57 GiB of its roughly 72 GiB usable memory on weights. Every P/D pod in this benchmark is TP1 or TP2, so **P/D pods never shard experts**, while the best aggregated layouts do. That confound applies to every agg vs P/D comparison here: the result is "TP1/TP2 P/D vs expert-sharded aggregated", not P/D in general.

### 3. Workloads and load

| Shape | Prompt tokens | Output tokens | What it stresses |
|---|---|---|---|
| prompt-heavy | 4,000 | 200 | prefill and KV transfer |
| chat | 1,000 | 500 | decode slots and decode KV |
| prefix | G shared 2,000-token prefixes + 300 unique (G = 128 on H100) | 200 | KV capacity and prefix reuse |
| agentic | 6 +- 1 turns per session; each turn re-sends the whole history plus 600 new tokens | 150 per turn | history reuse across turns |

Lengths are exact (`ignore_eos`, `min_tokens`). Requests per point = max(3c, 192), after a warm-up of min(c, 64) requests. Agentic points run 2c sessions with a 500 ms gap between turns and no warm-up ([`scripts/agentic.sh`](../scripts/agentic.sh)).

**Concurrency is not users.** `c` is requests in flight with zero think time. A real user who reads and thinks occupies far less than one slot.

### 4. Metrics, the SLO and the percentile rules

| Metric | Definition | What "good" looks like, and who says so |
|---|---|---|
| **TTFT** | Client time from sending a request to its first token: queueing + prefill + (P/D) KV transfer + routing | MLPerf Inference uses **2,000 ms at p99** as the conversational TTFT limit for Llama2-70B, Mixtral-8x7B and DeepSeek-R1 ([inference rules](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc)); usability research puts the "flow of thought" limit at about 1 s ([NN/g](https://www.nngroup.com/articles/response-times-3-important-limits/)). No body publishes a p50 or p90 TTFT target. |
| **ITL** | Per request: (request latency - TTFT) / (output tokens - 1). AIPerf's `inter_token_latency` | MLPerf's per-token limits (TPOT, p99) range from 200 ms (Llama2-70B, Mixtral conversational) to 80 ms (DeepSeek-R1, GPT-OSS server) and 15 to 80 ms in its interactive scenarios ([inference rules](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc)). |
| **tok/s per user** (interactivity) | `1000 / ITL p90`: how fast the slowest tenth of streams receive tokens. Excludes TTFT | InferenceX tiers: 30, 50 (default), 75, 100, 150, 200 tok/s/user; its cost calculator defaults to 35 ([InferenceX options](https://inferencex.semianalysis.com/api/v1/views/options), [interactivity](https://inferencex.semianalysis.com/glossary/interactivity)). InferenceX computes its agentic P90 interactivity the same way, as 1000 / p90 ITL ([P90 interactivity](https://inferencex.semianalysis.com/glossary/p90-interactivity)). |
| **SLO** | TTFT <= 2 s **and** the request's mean ITL <= 50 ms (50 ms is 20 tok/s) | The 2 s matches MLPerf's conversational TTFT bound. The 50 ms is this benchmark's choice: stricter than MLPerf's 80 to 200 ms server bounds, looser than its interactive ones. InferenceX declines to set a single SLO ([SLO](https://inferencex.semianalysis.com/glossary/service-level-objective)). |
| **SLO-ok** | Share of measured requests that met the SLO | DistServe treats at least 90% attainment as the goodput bar ([DistServe](https://arxiv.org/abs/2401.09670)); MLPerf requires 99% at its percentile. |
| **goodput** | req/s x SLO-ok / GPUs the layout uses | **Headline:** best goodput over the concurrency ladder among points with SLO-ok >= 90%. **Secondary:** peak goodput at any attainment (can sit where 1 in 4 requests fail; always shown with its SLO-ok). |
| **output tok/s per GPU** | Output tokens per second / all GPUs of the layout | Output only; input tokens are not counted. InferenceX headlines total (input + output) tok/s per GPU and, for its P/D output-only figure, divides by decode GPUs only ([code](https://github.com/SemiAnalysisAI/InferenceX/blob/main/inferencex-e2e/infx/results/fixed_sequence.py)). This benchmark divides by every GPU, which never flatters P/D. |
| **eff** (effective concurrency) | req/s x mean request latency / c (Little's law) | 1.0 means the load was held. Below about 0.9 the point's last requests drained with fewer than c in flight, which slightly understates req/s. Such points are kept and marked. |

**Percentiles.** Linear interpolation over per-request values of the measured phase (the convention AIPerf uses; the rebuilt p50 and p99 match AIPerf's own to floating-point precision). ITL percentiles are over per-request mean ITL.

**Two frontiers.** A throughput-vs-interactivity frontier answers "how much output per GPU can I get if every user streams at least X tok/s". The usual version ignores TTFT. This benchmark always shows two, pooled over every configuration of one node and one shape:
- **TTFT-blind:** the most output tok/s/GPU at or above each tok/s/user floor.
- **SLO-respecting:** the same, using only points with TTFT p90 <= 2 s. In this data set it picks the same point as an "SLO-ok >= 90%" filter at every floor.

### 5. Warm-up exclusion: what changed and why it matters

AIPerf tags each request record with its phase (`warmup` or `profiling`). The earlier analysis of these runs counted warm-up records in SLO-ok while taking req/s from the measured phase only. Rebuilding all 142 original points from the raw records reproduces the old values exactly with warm-up included, and gives these changes with warm-up excluded (33 of the 140 points with a warm-up are unchanged):

| Direction | Points | Why |
|---|---|---|
| SLO-ok **falls** | 89 | Warm-up requests run on an empty engine, so at high load they meet the SLO more often than the measured requests |
| SLO-ok **rises**, by up to 17 points | 18 | At c64 prompt-heavy, the warm-up fires 64 simultaneous 4,000-token prompts at a cold engine, which queue and miss; this hit P/D layouts hardest (A100 A2 P/D c64: 71.9% to 81.2%; H100 block-64 P/D c64: 89.1% to 92.7%) |

The effect on rankings is real: with warm-up excluded, the H100 prompt-heavy goodput gap between the best aggregated and best P/D layouts shrinks from 17% to 10% (peak), and at 90% attainment P/D moves ahead by 5%. Rule used everywhere: **warm-up excluded**. Both values are kept in the points files (`slo_ok`, `slo_ok_incl_warmup`).

### 6. Cost

Dollar figures appear only for H100, as an assumption: **$2.00 per GPU-hour**, the H100 "Rent - 3 Year Commit" input InferenceX publishes ([options API](https://inferencex.semianalysis.com/api/v1/views/options), [tokens per dollar](https://inferencex.semianalysis.com/glossary/tokens-per-dollar)). Formula (InferenceX's, [cost per million tokens](https://inferencex.semianalysis.com/glossary/cost-per-million-tokens)):

```
$ per million output tokens = $/GPU-hour x 1,000,000 / (3,600 x output tok/s per GPU)
```

Output tokens only, so prompt-heavy work looks expensive by construction. Any other rate scales linearly. **No citable A100 rate was found**, so A100 pages give the formula and no dollar values.

### 7. Validity checks

- **The KV really moved over NVLink.** Every P/D pod ran with `UCX_PROTO_INFO=y`; the protocol logs show `cuda_ipc` zero-copy for GPU reads (H100: 15,000+ transfers, 0 failures). P/D pods see all GPUs and pin their engine with `CUDA_VISIBLE_DEVICES`, because CUDA IPC needs the reading process to see the peer GPU; with one GPU per pod, UCX silently falls back to TCP. This is a benchmark setup, not a production pattern: pods that request no GPUs are invisible to the scheduler.
- **Effective concurrency** is shown on every point. It falls to 0.82 to 0.89 at long-queue points because of the fixed-count drain.
- **Replication.** One configuration (H100 P/D 2P(TP1):1D(TP2), block 64) was run twice; the runs agree within 1% on req/s and on goodput. Everything else is a single run. Differences under about 10% are treated as ties.

### 8. Caveats that apply to every page

- Single runs, except the replicated block-64 P/D configuration.
- The concurrency ladder is coarse (16, 64, 256, 512, 1,024), so the true capacity at 90% attainment lies between rungs.
- P/D pods never shard experts; the best aggregated layouts do.
- FP8 KV runs used default, uncalibrated scales; output quality was not checked.
- The A100 runs used a 2,048-token prefill budget against 8,192 on H100.
- Synthetic, fixed-length, closed-loop traffic. Real traffic has length variance and think time.
- The MoE kernel ran on vLLM's default config on H100 (no tuned file for this shape); running the H200-tuned file on H100 changed fused-MoE kernel time by -6.4% to +1.6%.

## What to do with it

- **Read every number with its conditions:** node, shape, concurrency, percentile and SLO-ok. The tables carry all five.
- **Re-rank for your own SLO** with [`scripts/derive.py`](../scripts/derive.py): change the 2 s / 50 ms bounds or the 90% bar, and the rankings and frontiers rebuild from [`data/requests/`](../data/requests).
- **Do not compare our tok/s/GPU with a published leaderboard number** unless the token accounting (output vs total, all GPUs vs decode GPUs), percentile (median vs p90) and sequence lengths match.
- **Size KV first.** The capacity table above predicts most results on the other pages before any run.
