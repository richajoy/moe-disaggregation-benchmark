# MoE disaggregation benchmark

**Aggregated or prefill/decode-disaggregated serving for a mixture-of-experts model? Ranked by requests that actually meet a latency target.**

[![vLLM 0.32.0](https://img.shields.io/badge/vLLM-0.32.0-1f6feb)](https://github.com/vllm-project/vllm)
[![llm-d 1.0](https://img.shields.io/badge/llm--d-1.0-eb6834)](https://github.com/llm-d/llm-d)
[![NVIDIA Dynamo 1.5.1](https://img.shields.io/badge/NVIDIA%20Dynamo-1.5.1-76B900)](https://github.com/ai-dynamo/dynamo)
[![Model Qwen3-30B-A3B](https://img.shields.io/badge/model-Qwen3--30B--A3B%20(MoE)-6250d6)](https://huggingface.co/Qwen/Qwen3-30B-A3B)
[![Hardware](https://img.shields.io/badge/hardware-4x%20H100%20%7C%208x%20A100-52514e)](docs/00-method.md#1-hardware-and-software)
[![Per-request data](https://img.shields.io/badge/data-286k%20per--request%20records-1baf7a)](data/)
[![License Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

## Two takeaways

1. **On one node, aggregated serving won.** On chat it served 3.2x more requests per second per GPU within target than the best disaggregated layout (4x H100); on long prompts the two tied. 8x A100 points the same way.
2. **The usual throughput-vs-speed chart overstates disaggregation by about 50%.** Its best points make users wait over 30 s for the first token. Rank layouts only on points that meet your latency target.

<sub>Target: first token within 2 s and at most 50 ms per token, for at least 90% of requests.</sub>

<p align="center">
<img src="figures/anim/anim-slo-frontier.svg" width="100%" alt="Animated scatter for 4x H100, Qwen3-30B-A3B, prompt-heavy traffic: output tokens per second per GPU against tokens per second per user (1000 / ITL p90) for agg DP4, agg 4x TP1 and P/D 2P(TP1):1D(TP2) block 64. The TTFT-blind frontier's best point at or above 35 tok/s/user is P/D at c1024, 1,368 tok/s/GPU, with TTFT p90 33.6 s and 1.5% of requests meeting the SLO. Points with TTFT p90 above 2 s turn hollow; the frontier rebuilt from filled points peaks at P/D c64, 915 tok/s/GPU, TTFT p90 1.8 s, 92.7% SLO-ok: the blind view is 49.6% higher.">
</p>

> <sub><i>Drop the points that miss the first-token target and the best disaggregated throughput falls from 1,368 to 915 tokens/s per GPU.</i></sub>

<p align="center">
<img src="figures/anim/anim-pd-request-layers.svg" width="100%" alt="Animated path of one prompt-heavy request through llm-d P/D on H100 (2 TP1 prefill pods, 1 TP2 decode, block 64, c=256): prefill queue 4,778 ms, prefill compute 359 ms, NIXL KV pull 27 ms, decode queue 52 ms, then a token every 15.9 ms (server-side means). The prefill queue is 86% of the 5,553 ms mean TTFT.">
</p>

> <sub><i>Disaggregation streams tokens smoothly, but the wait moves to the prefill queue: first-token time is what it trades away.</i></sub>

<p align="center">
<img src="figures/anim/anim-kv-levers.svg" width="100%" alt="Animated comparison of four aggregated 4x TP1 arms on H100 under overflowing prefix traffic. GPU KV per pod is 124,528 tokens in BF16 and 249,568 in FP8 against a 256k-token working set; two arms add a 100 GiB CPU tier. Prefix reuse is 30.9%, 76.8%, 55.5% and 76.8%. At c1024 throughput is 45.4, 53.3, 82.7 and 97.9 req/s, with TTFT p90 17.7, 14.9, 7.9 and 6.2 s and ITL p90 26.1, 23.1, 38.2 and 33.2 ms.">
</p>

> <sub><i>When shared prefixes overflow GPU memory, FP8 KV plus CPU offload carries twice the BF16 throughput.</i></sub>

## Go deeper

| Page | What you get |
|---|---|
| [Results at a glance](docs/results.md) | Every headline number with its conditions, and how it compares with MLPerf limits and InferenceX tiers |
| [H100: aggregated vs disaggregated](docs/01-h100-aggregated-vs-disaggregated.md) | The ranking, both frontiers, where requests wait |
| [KV transfer and block size](docs/02-kv-transfer-block-size.md) | The one flag that made disaggregation 2.5x better |
| [KV capacity levers](docs/03-kv-capacity-levers.md) | FP8 KV, CPU offload, flow control, each with its cost |
| [A100: aggregated vs disaggregated](docs/04-a100-aggregated-vs-disaggregated.md) | Why eight small replicas beat one large server |
| [Routing and prefix hits](docs/05-routing-and-prefix-hits.md) | When prefix-aware routing pays, and when it creates a hotspot |
| [Agentic load](docs/06-agentic.md) · [Autoscaling](docs/07-autoscaling.md) | Who waits under multi-turn load; why new pods sat idle |
| [Method](docs/00-method.md) · [Not measured](docs/08-not-measured.md) · [Glossary](docs/glossary.md) · [Reproduce](docs/reproduce.md) | How it was measured, what it does not show, and how to re-run it |

**Re-rank for your own target:** change the latency bounds in [`scripts/derive.py`](scripts/derive.py); the 286k per-request records in [`data/requests/`](data/requests) rebuild every table.

Related: [moe-agentic-serving-benchmark](https://github.com/richajoy/moe-agentic-serving-benchmark) (agent sessions on a 235B MoE, 8x H100) · [llm-inference-performance](https://github.com/richajoy/llm-inference-performance) (which serving knob moves which metric, from first principles).

## License

[Apache-2.0](LICENSE).
