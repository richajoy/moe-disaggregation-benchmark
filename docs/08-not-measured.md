# What this benchmark did not measure

> **What you get from this page:** the limits of every result here, ranked by how much they could change a conclusion, so you know which numbers to re-measure on your own setup before acting on them.

## The answer first

The results hold for **one MoE model, one node per GPU type, synthetic closed-loop traffic, mostly single runs, and P/D layouts that never shard experts**. The biggest open question is the fair P/D test: expert-sharded prefill and decode against an expert-parallel aggregated layout. Until that runs, "P/D did not beat aggregated" means "TP1/TP2 P/D did not beat expert-sharded aggregated on one node".

## Evidence: the gaps, ranked

| # | Not measured | Why it matters | What it could change |
|---|---|---|---|
| 1 | **Expert-sharded P/D** (DP or EP inside the prefill and decode pools) | Every P/D pod here holds the full MoE weights on 1 or 2 GPUs and keeps a quarter of the KV an expert-sharded GPU keeps | The agg vs P/D ranking on both nodes |
| 2 | **Open-loop traffic** (Poisson arrivals at a request rate) and **length variance** | `c` is requests in flight with zero think time, not users; every layout saturates by c1024. MLPerf's server scenario is open-loop Poisson | Where each layout's 90% capacity lies; tail latency under bursts |
| 3 | **Replication** | Only the H100 block-64 P/D layout ran twice (within 1%). Differences under about 10% are treated as ties | The H100 prompt-heavy and A100 prompt-heavy orderings |
| 4 | **Intermediate loads** | The ladder is 16, 64, 256, 512, 1,024. The 90%-attainment capacity lies between rungs (A100 A2 goes from 100% to 81% SLO-ok between c16 and c64) | Headline goodput values by up to the gap between rungs |
| 5 | **FP8 KV output quality** | Several of the best numbers use FP8 KV (prefix +116%, P/D chat 3.0x goodput, H100 agentic P/D) with default, uncalibrated scales | Whether those configurations are usable at all |
| 6 | **Other prefill/decode ratios** | H100 3P:1D was cancelled (a single TP1 decode would starve); A100 6P:1D was not run | The best P/D layout for prompt-heavy traffic |
| 7 | **Cross-node KV transfer** (RDMA) | Single node only; all transfers used NVLink (`cuda_ipc`) | Transfer cost and the block-size effect at scale |
| 8 | **Same prefill budget on both nodes** | vLLM chose 2,048-token prefill chunks on A100 and 8,192 on H100 | A100 prefill-queue times and per-prefill-GPU rates |
| 9 | **Tuned thresholds** | llm-d flow control ran on default thresholds; the selective-P/D threshold and `explorationProbability` were tested at one value each | Whether those features are net positive |
| 10 | **A tuned MoE kernel config for H100** | vLLM 0.32.0 ships none for this expert shape on H100; the H200 file changed kernel time by -6.4% to +1.6% | A few percent of decode step time |
| 11 | **Cost beyond a reference rate** | One assumed H100 rate, output tokens only, no idle or utilization accounting; no citable A100 rate | Any $/M comparison |
| 12 | **Per-step batch composition** | The DP8 interference explanation (prefill chunks stalling decode across expert-sharded ranks) is a likely mechanism, not traced | Why A0 loses to A1 on long prompts |

**Resolved here, not a gap any more:** the per-request records for every client point are published in [`data/requests/`](../data/requests), so SLO-ok, percentiles and goodput are reproducible from this repository. Warm-up requests are excluded throughout ([method 5](00-method.md#5-warm-up-exclusion-what-changed-and-why-it-matters)). Server-side signals (NIXL, KV usage, prefix hits, autoscaling samples) are transcribed in [`data/server-metrics.json`](../data/server-metrics.json); the raw scrapes are not published.

## What to do with it

1. **Before adopting a layout from these pages, re-run its row on your model and traffic** with intermediate concurrencies and at least two runs.
2. **Run an accuracy check before any FP8 KV deployment.**
3. **Treat the agg vs P/D verdict as conditional on weight placement.** If your P/D framework can shard experts in both pools, measure it.
4. **Use open-loop load for capacity planning**; closed-loop concurrency sweeps show the shape of the tradeoff, not your production headroom.
