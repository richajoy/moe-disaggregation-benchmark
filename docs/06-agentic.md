# Agentic multi-turn load: who waits, and how much history gets reused

> **What you get from this page:** how aggregated and disaggregated layouts behave when every turn re-sends a growing conversation, why almost none of that history was reused by default, and the two fixes that worked (prefix-aware routing, prefill-side CPU offload), with their limits.

Terms: **session** = one multi-turn conversation. **History reuse** = share of all prompt tokens (history plus new input) served from cached KV instead of recomputed. **Reuse ceiling** = the most a perfect cache could reuse for this workload.

**Not AgentX.** This is a short synthetic multi-turn loop (6 +- 1 turns, prompts up to about 5,900 tokens), not the SemiAnalysis AgentX trace replay, whose median request carries 142,016 input tokens ([AgentX methodology](https://inferencex.semianalysis.com/agentx/methodology)). For a larger MoE model (Qwen3-235B-A22B on 8x H100, 8 turns, prompts up to 8,200 tokens) see the companion [MoE agentic serving benchmark](https://github.com/richajoy/moe-agentic-serving-benchmark).

## The answer first

| Question | Answer | Conditions |
|---|---|---|
| H100: aggregated or P/D for agentic? | **Aggregated DP4: 6.81 vs 3.31 req/s/GPU of goodput** at c256 (72.4% vs 34.5% SLO-ok). Same throughput (37.7 vs 38.4 req/s), opposite failure modes | P/D used FP8 KV, agg BF16. Neither reaches 90% SLO-ok at any measured load. Single runs |
| How much history did they reuse? | **9.6% (agg DP4) and 1.6% (P/D prefill pods)** against a ceiling of about 76% | Whole-arm vLLM counters |
| A100: what fixed the reuse? | **Prefix-aware routing over 8 TP1 replicas: 62.7% reuse, 4.31 req/s/GPU goodput at 98.2% SLO-ok** (c256), the only agentic point on either node above 90% | vs 48.2% reuse and 84.0% SLO-ok behind a plain Service (same pods; block size also differs, 64 vs 16) |
| What helped P/D? | **Prefill-side CPU offload**: prefill CPU-tier hits 30.5% / 20.4%, output +29% / +21%, TTFT p90 -48% / -19% at c256 / c1024 | A100 P/D 4P:2D; single runs |
| At c1024? | Nothing works. Up to about 9M tokens of live history meet about 1.2M tokens of KV | Every layout below 6% SLO-ok on both nodes |

## Evidence

### 1. Workload

AIPerf on the chat endpoint (it carries multi-turn history only there), `scripts/agentic.sh`:

| Parameter | Value |
|---|---|
| Sessions | 2 x concurrency (512 at c256, 2,048 at c1024) |
| Turns per session | 6 +- 1 |
| New input per turn | 600 tokens; the whole history is re-sent every turn |
| Output per turn | 150 tokens, fixed |
| Gap between turns | 500 ms (simulated tool call) |
| Measured prompt growth | 600, 1,352, 2,104, ... 5,863 tokens: +752 per turn (600 new + 150 previous output + 2 template tokens) |

**The reuse ceiling.** A 6-turn session sends 600 + 1,352 + 2,104 + 2,856 + 3,608 + 4,360 = 14,880 prompt tokens; turns 2 to 6 can reuse 750 + 1,502 + 2,254 + 3,006 + 3,758 = 11,270 of them: **75.7%**. Effective concurrency is 0.79 to 0.88 by design, because sessions idle during the tool gap.

### 2. H100: same throughput, opposite failure modes

| H100, agentic | agg DP4, c256 | P/D 2P(TP1):1D(TP2) block 64 FP8 KV, c256 | agg DP4, c1024 | P/D, c1024 |
|---|---|---|---|---|
| req/s | 37.66 | 38.40 | 41.01 | 40.16 |
| output tok/s/GPU | 1,412 | 1,440 | 1,538 | 1,506 |
| TTFT p50 / **p90** / p99 ms | 1,459 / **2,709** / 3,511 | 2,867 / **5,898** / 6,731 | 5,892 / **13,525** / 16,223 | 19,627 / **30,301** / 32,199 |
| ITL p50 / **p90** / p99 ms | 26.5 / **39.2** / 46.5 | 13.8 / **16.6** / 20.9 | 118.1 / **120.6** / 128.4 | 13.7 / **27.2** / 40.4 |
| SLO-ok | **72.4%** | 34.5% | 5.9% | 1.9% |
| goodput req/s/GPU | **6.81** | 3.31 | 0.61 | 0.19 |
| history reuse | 9.6% (both loads) | prefill pods 1.6% | | |

Client values: `agentic-agg-dp4`, `agentic-pd-2p1d-dtp2-bs64-fp8kv` in [`data/h100/`](../data/h100), now including c256 rebuilt from the per-request records. Output tokens only; agentic token counts in other benchmarks are often dominated by cached input tokens, so do not compare tok/s/GPU across agentic benchmarks without matching the accounting.

- **Agg DP4's streams slow down.** Each engine step carries every stream's next token plus chunks of incoming prompts; with almost no reuse, every rank is prefilling most of the time (ITL p90 121 ms at c1024).
- **P/D's turns wait.** The decode only decodes (ITL p90 16.6 ms at c256), but every turn re-prefills its full history on two TP1 prefill pods, so TTFT grows. For an agent, TTFT per turn is the latency the user feels, because the next tool call cannot start until the turn's output arrives.

### 3. Why reuse was near zero

- **Not the chat template.** A single-pod control (8 sessions x 4 turns, no eviction possible) reused 64.7% against a computed ideal of 65.2%.
- **Agg DP4: scatter.** vLLM's internal data-parallel load balancer places each turn by load, not by session, so a session's next turn usually lands on a rank that does not hold its history. Random placement over 4 ranks would give about 76% / 4 = 19%; measured 9.6%. The working set (about 768k tokens at c256) fits in the 2.17M-token pool, so capacity is not the cause.
- **P/D prefill pods: eviction plus missing output KV.** The prefill tier takes in about 115k prompt tokens per second at c256, about 57.5k per pod, so a pod's 249,568 FP8 KV tokens turn over in about 4.3 s, while a session's next turn arrives about 5.5 s later. And the previous turn's 150 output tokens were generated on the decode pod, so the prefill tier can never reuse them.

### 4. A100: two fixes tested

Same workload on 8x A100 ([`data/a100/`](../data/a100), `agentic-*`; hit rates from vLLM counters, [`data/server-metrics.json`](../data/server-metrics.json)):

| Layout (router) | c | output tok/s/GPU | TTFT p50 / **p90** ms | ITL p50 / **p90** ms | SLO-ok | goodput req/s/GPU | Prefix reuse (GPU tier) | Prefill CPU-tier hit |
|---|---|---|---|---|---|---|---|---|
| 8x TP1, Kubernetes Service, block 16 | 256 | 605 | 580 / **1,426** | 36.1 / **51.0** | 84.0% | 3.39 | 48.2% | - |
| | 1024 | 493 | 20,456 / **34,412** | 91.7 / **97.0** | 2.3% | 0.07 | 1.7% (434 preemptions) | - |
| AG1 agg DP8, vLLM internal balancer | 256 | 379 | 2,274 / **5,681** | 51.4 / **79.8** | 12.4% | 0.31 | 24.0% | - |
| | 1024 | 330 | 5,594 / **17,156** | 334.5 / **371.9** | 0.2% | 0.005 | 17.8% | - |
| **AG2 8x TP1, llm-d approximate prefix routing, block 64** | 256 | **658** | 261 / **909** | 37.8 / **44.4** | **98.2%** | **4.31** | **62.7%** | - |
| | 1024 | 505 | 18,740 / **32,306** | 103.1 / **107.0** | 4.7% | 0.16 | 2.1% (338 preemptions) | - |
| AG3 P/D 4P(TP1):2D(TP2), block 64 | 256 | 480 | 4,835 / **10,879** | 18.1 / **23.6** | 31.4% | 1.00 | prefill 10.5%, decode 56.3% | 0% |
| | 1024 | 462 | 34,155 / **50,620** | 19.2 / **29.6** | 1.6% | 0.05 | prefill 0.5%, decode 4.5% | 0% |
| AG4 = AG3 + 100 GiB CPU offload per pod | 256 | 621 | 2,576 / **5,624** | 21.2 / **28.9** | 40.5% | 1.68 | prefill 9.1%, decode 57.4% | **30.5%** |
| | 1024 | 561 | 25,440 / **41,082** | 23.1 / **35.3** | 1.9% | 0.07 | prefill 0.5%, decode 4.5% | **20.4%** |

- **Fix 1, prefix-aware routing, works while sessions fit in KV.** Against the same eight TP1 pods behind a plain Service, it raised reuse from 48.2% to 62.7% (of a 75.7% ceiling), TTFT p90 from 1.43 to 0.91 s, output from 605 to 658 tok/s/GPU (+9%) and SLO-ok from 84.0% to 98.2%. Block size also differed (64 vs 16), so the gain is not routing alone.
- **Fix 2, prefill-side CPU offload, is the only lever that helped at c1024.** It recovered 30.5% and 20.4% of prefill tokens from host memory, for +29% and +21% output and TTFT p90 10.9 to 5.6 s and 50.6 to 41.1 s, at a cost of about 5 to 6 ms of ITL p90. It cannot fix the structural gap: the previous turn's output KV still lives only on the decode.
- **At c1024 routing cannot help.** 2,048 sessions hold up to about 9M tokens of history against about 1.2M tokens of KV across 8 TP1 pods, so history is evicted before the next turn on any routing.

### Caveats

- Single runs; c256 and c1024 only.
- H100 P/D used FP8 KV and agg used BF16 (FP8 favors P/D on capacity; output quality not checked).
- Reuse figures are whole-tier prompt-token counters, not per-session measurements.
- AG4 also configured the offload tier on the decode pods; its ITL cost was not attributed.
- Effective concurrency 0.79 to 0.88 by design, so req/s is not comparable with the fixed-shape pages.

## What to do with it

1. **Route by session or prefix** whenever history is re-sent each turn; neither vLLM's internal DP balancer nor a plain Service keeps a session on the replica that holds its history.
2. **Size KV for the live history, not one request:** the working set is sessions x average history length. When it exceeds the pool, add capacity (FP8 KV, CPU offload, more replicas); better routing cannot help.
3. **In P/D, put a CPU offload tier behind the prefill pods**, or keep a session's history where its next prefill runs.
4. **Pick the failure mode on purpose.** If users watch tokens stream, P/D keeps ITL low; if agents wait for whole turns, aggregated gives the lower TTFT.
