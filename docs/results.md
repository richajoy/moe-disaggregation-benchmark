# Results at a glance

Every headline number in one place, with its conditions. Back to the [overview](../README.md).

## Summary

**On one node, no prefill/decode (P/D) layout beat the best aggregated layout by more than run noise, and aggregated won chat by 2 to 3x.** P/D's large leads on the usual throughput-vs-interactivity chart exist only at loads where TTFT p90 (time to first token, 90th percentile) is 7 to 57 s; filter those points out and the best P/D throughput drops by a third (1,368 to about 915 output tok/s/GPU on H100). The settings that moved results most were about KV: more KV capacity (decode on two GPUs, FP8 KV plus CPU offload), a cheaper KV transfer (`--block-size 64`), and a router that spreads load instead of pinning it to the pods that hold a prefix. Scope: the P/D pods here never split experts across GPUs while the best aggregated layouts do, so the verdict is "TP1/TP2 P/D vs expert-sharded aggregated" ([limits](reproduce.md#limits)).

Everything below is ranked by **goodput**: requests per second per GPU that met **TTFT <= 2 s and mean ITL (inter-token latency) <= 50 ms**, at the best load where at least 90% of requests met it. Latency is **p90** unless stated. Warm-up requests are excluded. H100 and A100 are never compared.

Notation: **cN** = N requests in flight (closed loop, not users). **SLO-ok** = share of requests that met both bounds. **agg DP4** = one vLLM server on 4 GPUs with the MoE experts split across them; **agg 4x TP1** = four independent 1-GPU replicas. **2P(TP1):1D(TP2)** = two 1-GPU prefill instances feeding one 2-GPU decode instance. Everything else: [glossary](glossary.md).

## Headline results

| Question | Answer | Conditions |
|---|---|---|
| H100 chat: aggregated or P/D? | **Aggregated, 3.2x.** agg DP4 7.86 vs best P/D 2.48 req/s/GPU | 4x H100, 1,000 in / 500 out. agg DP4 at c256: TTFT p90 1.28 s, ITL p90 15.5 ms, 100% SLO-ok. Single runs. P/D with FP8 KV reaches 5.62 at 89.7% SLO-ok, just under the bar; counted, the lead is 1.4x |
| H100 prompt-heavy: aggregated or P/D? | **Tie.** P/D 2P(TP1):1D(TP2) block 64 4.24 vs agg DP4 4.04 req/s/GPU (+5%, inside run noise) | 4,000 in / 200 out, both at c64. P/D replicated (4.23; block 128, single run: 4.26); P/D TTFT p90 1.80 s vs 1.35 s. At any attainment agg DP4 peaks higher, 4.65 at 74.9% SLO-ok |
| H100: how much does the usual chart overstate P/D? | **+49.6%** (1,368 vs 915 output tok/s/GPU) | Prompt-heavy, at >= 35 tok/s per user. The 1,368 point has TTFT p90 33.6 s and 1.5% SLO-ok; 915 is the replicated P/D point at c64 (the single-run block-128 point gives 919, +48.9%) |
| A100: how much does the usual chart overstate P/D? | **+46.8%** (424 vs 289 output tok/s/GPU) | Prompt-heavy, at >= 35 tok/s per user. The 424 point has TTFT p90 56.5 s and 0.9% SLO-ok |
| A100: aggregated or P/D? | **Eight independent TP1 replicas: 2.0x on chat** (2.19 vs 1.09), 1.7x on prompt-heavy (1.40 vs 0.81); prompt-heavy peaks tie (1.40 vs 1.38) | 8x A100, single runs, 2,048-token prefill budget |
| One flag for heterogeneous P/D? | **`--block-size 64`: goodput 1.72 to 4.24 req/s/GPU (2.5x)** | H100 2P(TP1):1D(TP2), prompt-heavy, >= 90% SLO-ok. At c256 (past the SLO for both): 1.86x req/s (13.87 to 25.82), ITL p90 73.4 to 17.0 ms; replicated. The same descriptor effect shows on A100 (ITL p90 37.1 to 18.5 ms at c256) |
| Prefix traffic that overflows GPU KV? | **FP8 KV + CPU offload: goodput +48%** (10.02 to 14.85 req/s/GPU at c256, both >= 90% SLO-ok) | H100 agg 4x TP1, 128 x 2,000-token shared prefixes. At c1024 (overload): +116% req/s (45.40 to 97.87), TTFT p90 17.7 to 6.2 s, SLO-ok 8.5% to 56.2%. FP8 output quality not checked |
| Autoscaling behind a prefix-aware router? | **The 6 new pods served 0 requests**; a load-only router gave each 908 to 1,242 and cut burst TTFT p99 from 1,197 to 164 ms | A100, llm-d KEDA path, 2 to 8 pods, one run per arm |

Every number above traces to [`data/`](../data/); the derivation is [`scripts/derive.py`](../scripts/derive.py) and its output [`data/derived/TABLES.md`](../data/derived/TABLES.md).

## The key picture

<p align="center">
<img src="../figures/anim/anim-slo-frontier.svg" width="100%" alt="Animated scatter for 4x H100, Qwen3-30B-A3B, prompt-heavy traffic: output tokens per second per GPU against tokens per second per user (1000 / ITL p90) for agg DP4, agg 4x TP1 and P/D 2P(TP1):1D(TP2) block 64. The TTFT-blind frontier's best point at or above 35 tok/s/user is P/D at c1024, 1,368 tok/s/GPU, with TTFT p90 33.6 s and 1.5% of requests meeting the SLO. Points with TTFT p90 above 2 s turn hollow; the frontier rebuilt from filled points peaks at P/D c64, 915 tok/s/GPU, TTFT p90 1.8 s, 92.7% SLO-ok: the blind view is 49.6% higher.">
</p>

> <sub><i>Rank layouts on points that meet the SLO; the TTFT-blind frontier credits P/D with throughput no SLO-bound user receives.</i></sub>

<p align="center">
<img src="../figures/anim/anim-pd-request-layers.svg" width="100%" alt="Animated path of one prompt-heavy request through llm-d P/D on H100 (2 TP1 prefill pods, 1 TP2 decode, block 64, c=256): prefill queue 4,778 ms, prefill compute 359 ms, NIXL KV pull 27 ms, decode queue 52 ms, then a token every 15.9 ms (server-side means). The prefill queue is 86% of the 5,553 ms mean TTFT.">
</p>

> <sub><i>Once the transfer is fixed, P/D's wait sits in the prefill queue: TTFT, not streaming speed, is what P/D trades away.</i></sub>

## How this compares with common bars

There is no published reference for this model or for A100: InferenceX has no Qwen3-30B-A3B and no A100 data. The bars below are what common benchmarks treat as good; they are **indicative and cross-model**, not a ranking. MLCommons does not allow comparing results with MLPerf results, so only MLPerf's limit values are cited, as a common bar.

H100 values below; the A100 page reads its own results against the same bars ([04](04-a100-aggregated-vs-disaggregated.md)).

| Metric | Common bar (source) | H100, at the best SLO-meeting points | Reading |
|---|---|---|---|
| TTFT | 2,000 ms at p99, MLPerf conversational limit for Llama2-70B, Mixtral-8x7B, DeepSeek-R1 ([rules](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc)) | Chat agg DP4 c256: p90 1.28 s, p99 1.64 s. Prompt-heavy P/D c64: p90 1.80 s, p99 2.47 s | Chat clears the bar at p99; long prompts clear it at p90 only |
| Per-token latency | 200 ms (Mixtral conversational) to 80 ms (DeepSeek-R1, GPT-OSS server) per output token at p99, MLPerf | ITL p99 15.7 ms (chat), 13.8 ms (prompt-heavy P/D) | Streaming is never the binding limit at these points; TTFT is |
| Streaming speed | InferenceX tiers 30, 50 (default), 75, 100, 150, 200 tok/s/user; calculator default 35 ([options](https://inferencex.semianalysis.com/api/v1/views/options)) | 64.5 tok/s/user (chat), 73.1 (prompt-heavy P/D), both as 1000 / ITL p90 | Above the default 50 tier |
| SLO attainment | At least 90% (DistServe, [paper](https://arxiv.org/abs/2401.09670)); MLPerf requires 99% at its percentile | Headline ranking requires >= 90%; the peak-goodput points sit at 74.9% (prompt-heavy) and 76.2% (chat) | Ask which attainment a "max goodput" number assumed |
| GPU accounting for P/D | InferenceX divides P/D output throughput by decode GPUs only ([code](https://github.com/SemiAnalysisAI/InferenceX/blob/main/inferencex-e2e/infx/results/fixed_sequence.py)) | All GPUs of the layout, prefill and decode | Decode-only accounting would show 2x higher numbers for 2P(TP1):1D(TP2) |

## What to do with it

1. **Serve an MoE model of this size aggregated first,** with experts split across GPUs (`--data-parallel-size`) on H100, or as independent single-GPU replicas on A100. Test P/D only for long prompts, and only with decode at TP2 or more, `--block-size 64` and enough prefill GPUs for your TTFT budget.
2. **Never pick a layout from a TTFT-blind frontier.** Filter by TTFT p90 or SLO attainment first. On prompt-heavy traffic, on both nodes, the blind frontier's top point was a P/D point with TTFT p90 of 34 to 57 s, about 1.5x what any SLO-meeting point delivered.
3. **Add KV capacity before anything else:** FP8 KV (after an accuracy check) and CPU offload when prefixes overflow GPU memory.
4. **Check new-pod uptake after every scale-out** if your router does prefix affinity.
5. **Re-rank for your own target:** change the SLO bounds or the 90% bar in [`scripts/derive.py`](../scripts/derive.py); the per-request records in [`data/requests/`](../data/requests) rebuild every table.
