# Glossary

> **What you get from this page:** a plain-language definition of every term, notation and metric used in this benchmark, with the convention this repository uses where several exist.

## The answer first

Four ideas carry the whole benchmark: **TTFT** (how long a user waits for the first token), **ITL** (the gap between streamed tokens), the **SLO** that combines them (TTFT <= 2 s and mean ITL <= 50 ms), and **goodput** (requests per second per GPU that met the SLO). Everything else describes a layout, a workload or a cause.

## Evidence: definitions

### Layouts and parallelism

| Term | Meaning |
|---|---|
| **Aggregated (agg)** | One engine does both prompt processing (prefill) and token generation (decode) for each request |
| **P/D, disaggregated** | Prefill and decode run on separate GPU pools; the prompt's KV cache is copied from the prefill pod to the decode pod |
| **Prefill** | Processing the whole prompt in one pass to build its KV cache and produce the first token. Compute-bound |
| **Decode** | Generating output tokens one step at a time, each step reading the weights and the KV cache. Memory-bandwidth-bound |
| **TP1, TP2** | Tensor parallel size: how many GPUs share one copy of the model. A TP1 pod holds all weights on one GPU |
| **DP4, DP8** | One vLLM server with `--data-parallel-size 4` (or 8): attention replicated per GPU, MoE experts split across GPUs, an internal load balancer spreads requests |
| **MoE (mixture of experts)** | A model whose feed-forward layers are many "experts", of which each token uses a few. Qwen3-30B-A3B has 128 experts per layer and activates 8 (about 3B of 30.5B parameters per token), but all experts must sit in GPU memory |
| **2P(TP1):1D(TP2)** | A P/D layout: 2 prefill instances of 1 GPU each, 1 decode instance on 2 GPUs. 4 GPUs in total |
| **1P:3D, 2P:2D** | P/D layouts with all instances TP1 |
| **4P:2D, 2P:3D (A100)** | 4 or 2 TP1 prefill GPUs; 2 or 3 TP2 decode instances. 8 GPUs in total |
| **A0 to A4** | The A100 arms: A0 agg DP8, A1 agg 8x TP1, A2 P/D 4P:2D block 64, A3 P/D 2P:3D block 64, A4 = A2 at block 16 |
| **Knob arm** | A 3-GPU 1P:2D TP1 P/D setup used to vary one transfer setting at a time (pull/push, block size, prefill budget, selective P/D) |
| **Expert sharding** | Splitting each MoE expert's weights across GPUs (DP or expert parallelism) so each GPU holds a fraction and keeps more memory for KV |

### KV cache and transfer

| Term | Meaning |
|---|---|
| **KV cache** | The keys and values a model stores for every token seen so far, so later tokens need not recompute them. 98,304 bytes per token for this model in BF16, 49,152 in FP8 |
| **KV tokens (capacity)** | How many tokens of KV fit in a pod's spare GPU memory; sets how many requests run at once |
| **Preemption** | vLLM evicts a running request's KV when memory runs out and recomputes it later. In P/D the decode recomputes the prompt locally |
| **KV block, block size** | The unit (16 tokens by default; 64 here where noted) in which vLLM stores and copies KV |
| **NIXL** | NVIDIA's transfer library, used by vLLM's P/D connectors |
| **Descriptor** | One `(address, length, device)` piece of a transfer. The CPU builds and posts each one, so many small descriptors cost CPU time |
| **Pull / push** | Pull: the decode reads the prefill's blocks (vLLM's default `NixlConnector`). Push: the prefill writes into blocks the decode reserved |
| **Heterogeneous TP** | Prefill and decode at different TP sizes (TP1 to TP2); each decode rank reads half of every block's KV heads |
| **`cuda_ipc`** | The UCX transport for GPU-to-GPU copies inside one node over NVLink/NVSwitch |
| **FP8 KV** | Storing KV in 8-bit floats: half the bytes, twice the tokens. Output quality must be checked |
| **CPU offload** | A host-memory tier behind GPU KV (vLLM `OffloadingConnector`); evicted blocks can be loaded back instead of recomputed |
| **MultiConnector** | vLLM's way to combine connectors, here NIXL (P/D transfer) and CPU offload |
| **Chunked prefill budget** | The most prompt tokens vLLM processes per engine step (`max_num_batched_tokens`): 8,192 on the H100 runs, 2,048 on A100 |

### Routing and scaling

| Term | Meaning |
|---|---|
| **EPP** | llm-d's endpoint picker: the router that chooses a pod for each request through filters and scorers |
| **Prefix-aware routing** | Sending a request to the pod believed to cache its prompt prefix. *Approximate*: the router remembers its own decisions. *Precise*: the router learns from the engines' KV events |
| **Prefix-affinity filter** | An llm-d router stage that drops every pod not holding at least 80% of a prompt's prefix blocks when any pod does |
| **Prefix hit (rate)** | Share of prompt tokens served from cached KV. *Predicted* = the router's estimate; *actual* = the engine's counters |
| **Flow control** | llm-d feature that holds requests at the router when the pool looks saturated |
| **Selective P/D** | Disaggregate only requests whose uncached prompt is at least a threshold (2,048 tokens here); shorter prompts prefill on the decode pod |
| **KEDA, HPA** | Kubernetes autoscaling: KEDA turns a metric (router in-flight requests) into a desired replica count that the HPA applies |
| **Planner** | NVIDIA Dynamo's autoscaler |
| **Cold start** | Time from a new pod starting to serving traffic |

### Load and workloads

| Term | Meaning |
|---|---|
| **c16 ... c1024** | Concurrency: requests in flight at once, closed loop (a new request starts as soon as one finishes, no think time). Not users |
| **ISL / OSL** | Input (prompt) and output sequence length in tokens |
| **prompt-heavy** | 4,000 tokens in, 200 out |
| **chat** | 1,000 in, 500 out |
| **prefix (G)** | G shared 2,000-token prefixes plus 300 unique tokens, 200 out |
| **agentic** | Multi-turn sessions (6 +- 1 turns, +600 new tokens per turn, history re-sent, 150 out per turn, 500 ms between turns) |
| **Warm-up** | The first min(c, 64) requests of each point; excluded from every metric here |
| **eff (effective concurrency)** | req/s x mean request latency / c. Below about 0.9 the point drained at the end; its req/s is slightly understated |
| **Drain** | With a fixed request count, the last requests run with fewer than c in flight |

### Metrics

| Term | Meaning |
|---|---|
| **TTFT** | Time to first token, as the client sees it: queueing + prefill + (P/D) KV transfer + routing |
| **ITL** | Inter-token latency of one request: (request latency - TTFT) / (output tokens - 1). Also called TPOT |
| **p50 / p90 / p99** | Median, 90th and 99th percentile over the measured requests of one point. **p90 is the headline percentile here** |
| **tok/s per user (interactivity)** | `1000 / ITL p90`: streaming speed of the slowest tenth of requests. Excludes TTFT. Not AIPerf's per-user p90, which is the fast tail |
| **output tok/s per GPU** | Output tokens per second divided by every GPU of the layout (prefill and decode) |
| **SLO** | Service level objective. Here: TTFT <= 2 s and the request's mean ITL <= 50 ms |
| **SLO-ok** | Share of measured requests that met the SLO |
| **Goodput** | req/s x SLO-ok / GPUs. Headline: the best such value over the ladder among points with SLO-ok >= 90%. "Peak goodput" drops the 90% floor |
| **Frontier, TTFT-blind** | For each tok/s-per-user floor, the most output tok/s/GPU any point reaches, ignoring TTFT |
| **Frontier, SLO-respecting** | The same, using only points with TTFT p90 <= 2 s |
| **Overstatement** | How much the TTFT-blind frontier exceeds the SLO-respecting one at the same floor |
| **$/M output tokens** | $/GPU-hour x 1,000,000 / (3,600 x output tok/s per GPU). H100 only, at an assumed $2.00/GPU-hour |
| **Stage timings** | Mean time per request in each server stage (prefill queue, prefill, decode queue, NIXL transfer, decode) from vLLM's histograms |
| **Residual** | Client TTFT minus the prefill and decode pods' own TTFTs: router, sidecar, network and measurement-window effects |

### Notation

| You will see | It means |
|---|---|
| "X to Y" | Before, then after one change, same node and load |
| "points" | Percentage points (68.8% - 63.0% = +5.8 points) |
| `config-name` in code font | A file in `data/<node>/<config-name>.points.json` |

## What to do with it

Read every number on the other pages as a sentence: *node, shape, concurrency, percentile, SLO-ok*. If one of the five is missing, the number is not comparable with another.
