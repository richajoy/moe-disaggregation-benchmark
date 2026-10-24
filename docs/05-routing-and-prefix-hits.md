# Routing: when prefix-aware routing pays, and how far router hit predictions drift

> **What you get from this page:** a rule for when prefix-aware routing is worth its risk, measured on 8x A100 TP1 replicas with four llm-d router profiles; the failure mode that can put a whole burst on one pod; and how far the llm-d and Dynamo routers' predicted cache hits drift from what the engines actually deliver. A100 only.

Terms: **prefix-aware routing** sends a request to the pod the router believes already caches its prompt prefix. **Hit rate** = share of prompt tokens served from cached KV. **G** = number of distinct 2,000-token shared prefixes in the workload. **EPP** = llm-d's endpoint picker (the router). **Predicted vs actual hits**: the router routes on its prediction; the engine's counters report what really hit.

## The answer first

- **Prefix routing pays only when the shared prefixes do not fit in one pod's KV but do fit in the pool.** With 128 prefixes (256k tokens against 158,336 tokens per pod), llm-d's approximate prefix routing beat round robin at c1024 by **+67% output tok/s** (12,701 vs 7,606), TTFT p90 5.0 s vs 16.7 s, 0 preemptions vs 270, hit rate 81.2% vs 38.2%. Two runs per arm.
- **When prefixes fit everywhere, it loses.** With 16 prefixes, round robin already reaches 82.6% of an 86.2% hit ceiling, and prefix routing was **25% slower** at c1024 (11,117 vs 14,726 tok/s; ITL p90 111 vs 65 ms) because it piles prefix groups onto a few pods.
- **One shared prefix in a burst can land on a single pod.** In one of two identical runs, the approximate-prefix profile sent all 1,024 burst requests to one of eight pods: 2,424 vs 13,581 output tok/s for round robin (-82%), TTFT p90 61.7 s vs 1.4 s.
- **Router predictions drift under eviction.** llm-d's precise index predicted 68.8% hits where the engines delivered 63.0% (+5.8 points) when pods were evicting; with FP8 KV the eviction stopped and the gap fell to +0.4. Dynamo's router slightly under-predicted (-2.0 to -4.4 points at BF16).
- **llm-d's precise (KV-event) mode needs a warm-up after every router restart:** cold, it lost 13 to 16 points of hit rate; warm, it landed within 0.6 points of the ideal with zero index errors.

## Evidence

### 1. Setup

| Item | Value |
|---|---|
| Model servers | 8 TP1 replicas, vLLM 0.32.0, prefix caching on, block 64, KV events on in every arm (the same 8 pods serve every arm); 158,336 KV tokens per pod |
| Router | llm-d 1.0 router, one plugin profile per arm; router restarted and every prefix cache reset before each run ("cold" protocol) |
| Workload | G prefixes x 2,000 tokens + 300 unique tokens, 200 output tokens; requests = max(3c, 192); same seed for every arm |
| Hit ceiling | 1,984 of 2,301 tokens are cacheable in 64-token blocks: **86.2%** |

| Arm | Profile | What it does |
|---|---|---|
| R0 | plain Kubernetes Service | balances connections; knows nothing about prefixes |
| R1 | queue + KV-utilization scorers, max-score picker | picks the least loaded pod by its metrics |
| R2 | approximate prefix producer + prefix-cache-affinity filter + token-load scorer (guide defaults) | remembers which pod it sent each prefix to and keeps sending it there |
| R3 | precise prefix producer fed by the engines' KV events | learns what each pod actually cached |

### 2. Results

Mean over two runs for c1024 and for G128 c256; one run for G16 c256. Client values from [`data/a100/routing-runs.json`](../data/a100/routing-runs.json) (TTFT and ITL percentiles recomputed from the per-request records).

| Point | Arm | Output tok/s (8 GPUs) | TTFT p50 / **p90** ms | ITL **p90** ms | Engine hit rate | Preemptions |
|---|---|---|---|---|---|---|
| G16 c256 | R0 / R1 / **R2** / R3 | 7,449 / 7,411 / **8,395** / 7,624 | R2: 190 / **1,342** (R0: 272 / 2,003) | R2 30.7 (R0 30.3) | 71.9 / 71.9 / 80.0 / 76.2% | 0 everywhere |
| G16 c1024 | **R0** / R1 / R2 / R3 | **14,726** / 14,371 / 11,117 / 12,727 | R0: 616 / **4,070**; R2: 663 / **4,949** | R0 65.0, R2 **111.2** | 82.6 / 82.6 / 85.7 / 83.7% | 0 everywhere |
| G128 c256 | R0 / R1 / **R2** / R3 | 5,655 / 6,194 / **7,679** / 7,014 | R2: 197 / **2,276** (R0: 519 / 2,975) | R2 33.6 (R0 42.7) | 26.4 / 25.6 / 68.3 / 57.3% | 0 everywhere |
| G128 c1024 | R0 / R1 / **R2** / R3 | 7,606 / 7,773 / **12,701** / 11,309 | R0: 8,528 / **16,655**; R2: 472 / **4,958** | R0 81.4, R2 86.1 | 38.2 / 39.2 / 81.2 / 77.5% | 270 / 266 / 0 / 59 |

Hit rates and preemptions: vLLM counters summed over the 8 pods.

### 3. When prefix routing pays, from first principles

| | G16 | G128 |
|---|---|---|
| Distinct prefix tokens | 32k | 256k |
| Round robin: every pod sees every prefix | 32k per pod, fits in 158,336 | **256k per pod, does not fit**: evicted before reuse |
| Prefix routing: prefixes sharded across 8 pods | about 4k per pod | about 32k per pod, fits |

- **G128: routing decides whether the cache works at all.** Round robin evicts a prefix before it returns to the same pod; hits stay at 26 to 39% and at c1024 preemptions push TTFT p90 to 16.7 s. Prefix routing gives each pod about 16 prefixes; hits reach 68 to 81%.
- **G16: there is nothing to win.** Every pod already holds every prefix. Prefix routing can add at most about 3 points of hit rate and loses more than that to load imbalance.

### 4. The hotspot: the affinity filter does not let go

The approximate-prefix profile keeps only "sticky" pods: those whose prefix score (matched blocks / total blocks) is at least 0.80. The first pod to serve a prefix becomes sticky for it and stays sticky, because the router's index records its own routing decisions. The only release is a load gate that compares *estimated* TTFT, computed as in-flight **uncached prompt tokens** / an assumed prefill throughput. With about 300 uncached tokens per request, the estimate stays near zero however many requests are decoding or queued on the hot pod. (Mechanism from reading the llm-d router 1.0 source; the filter's decision counter was not captured.)

**Burst test: one shared 2,000-token prefix, c512, 1,024 requests, 500 output tokens**

| Arm | Per-pod requests | TTFT p50 / **p90** / p99 ms | ITL p50 / **p90** ms | Output tok/s | Preemptions |
|---|---|---|---|---|---|
| R0 round robin | 114 to 140 on all 8 | 726 / **1,358** / 1,675 | 34.7 / **36.5** | 13,581 | 0 |
| R2 defaults, run 1 | **1,024 on 1 pod, 0 on 7** | 56,572 / **61,704** / 78,600 | 94.5 / **113.9** | **2,424** | 286 |
| R2 defaults, run 2 | 242 to 271 on 4 pods, 0 on 4 | 661 / **2,831** / 3,555 | 48.7 / **53.5** | 9,663 | 0 |
| R2 calibrated prefill rate, run 1 | 123 to 132 on all 8 | 358 / **1,368** / 1,741 | 36.0 / **39.8** | 13,233 | 0 |
| R2 calibrated prefill rate, run 2 | 339 to 345 on 3 pods, 0 on 5 | 1,150 / **3,844** / 4,981 | 58.3 / **60.1** | 8,353 | 0 |
| R3 precise | 491 on 1 pod, 61 to 132 on the others | 1,649 / **44,774** / 48,892 | 35.7 / **76.2** | 5,432 | 69 |

- **The lock-in width varies between identical runs** (1, 4, 8 and 3 pods across four runs with the same seed): whichever pods cache the prefix in the first moments become the only candidates.
- **The gate did not fire even at 56.6 s median TTFT**, far above its 18,000 ms penalty, because the hot pod's cost is decode and queueing, which the estimate does not count.
- **Calibrating the prefill rate is not a fix.** It rescales a near-zero estimate; at G16 c1024 the calibrated profile was worse (8,624 tok/s, busiest pod 2.05x the mean).
- **R1 herds too.** Concurrent requests read the same metrics snapshot and an argmax picker sends them to the same pod: TTFT p50 1.8 to 2.5x worse than round robin at three of four points.

The same filter starves newly scaled pods under autoscaling: [07](07-autoscaling.md).

### 5. Precise mode is a cold-start artifact

Every matrix run was cold by protocol (router restarted, prefix caches reset). In a warm repeat (no restart, new prefixes each run):

| Setup | Arm | Runs | Engine hit rate | Ideal | KV-event index errors |
|---|---|---|---|---|---|
| 2 pods, G16, c64 | R3 warm | 3 | 83.4 / 84.1 / 84.1% | 84.0% | 0 |
| 2 pods, G16, c64 | R2 warm | 3 | 82.2 / 83.3 / 82.5% | 84.0% | n/a |
| 8 pods, G128, c256 | R3 warm | 2 | 71.2 / 71.2% | 71.8% | 0 |
| 8 pods, G128, c256 | R3 cold | 3 | 58.6 / 55.9 / 56.7% | 71.8% | 25 to 333 key-not-found, 8 to 17 incomplete replays per run |

Ideal = 86.2% x (N - G) / N (one cold miss per prefix group). Cold runs lose about 13 to 16 points while the index re-syncs; whether the router restart or the cache reset causes it was not isolated.

### 6. Predicted vs actual hits, llm-d vs Dynamo

Two TP1 pods per stack, running side by side on disjoint GPUs; a hit-accounting comparison, **not a performance comparison**. llm-d: vLLM 0.32, block 64, R3 precise profile. Dynamo 1.5.1: its vLLM 0.28 runtime, block 16, `--router-mode kv`. Workload: 2,000-token prefix + 300 unique, c256, 832 requests per cell, measured pass after a warm pass with the same prompts (so hit rates above the single-pass ceiling are expected). One run per cell.

| Stack | KV | G | Engine hit rate | Router-predicted | Gap (points) |
|---|---|---|---|---|---|
| Dynamo | BF16 | 128 | 44.3% | 42.3% | -2.0 |
| Dynamo | BF16 | 16 | 83.8% | 79.4% | -4.4 |
| Dynamo | FP8 | 128 | 68.7% | 66.0% | -2.7 |
| Dynamo | FP8 | 16 | 92.2% | 93.2% | +1.0 |
| llm-d | BF16 | 128 | 63.0% | 68.8% | **+5.8** |
| llm-d | BF16 | 16 | 84.7% | 85.2% | +0.6 |
| llm-d | FP8 | 128 | 79.2% | 79.5% | +0.4 |
| llm-d | FP8 | 16 | 97.3% | 97.5% | +0.2 |

Source: [`data/server-metrics.json`](../data/server-metrics.json) (`prefix_hit_accounting_a100_two_pods`).

- **llm-d over-predicts when pods evict.** At BF16 G128 the 256k-token working set exceeds 2 x 158k of KV. With FP8 (2 x 309k tokens) it fits, and the gap collapses. The gap tracks eviction, not FP8: the index keeps blocks the engine has already dropped, or learns of the drop late.
- **Dynamo under-predicts slightly**, the safer direction. One plausible reason: it credits only full 16-token blocks matched at routing time, while the engine also hits blocks stored after routing (not checked).
- **FP8 KV raises actual hits in both stacks** (G128: Dynamo 44.3% to 68.7%, llm-d 63.0% to 79.2%) by halving eviction. FP8 KV ran without errors on A100 in both vLLM versions; its output quality was not checked.

### Caveats

- Short runs (20 to 70 s); TTFT p99 at c256 rests on few samples. Two runs per c1024 and G128 c256 point, one for G16 c256.
- Cold protocol for every matrix run; long-lived routers may behave differently, especially R3.
- vLLM 0.32.0 is newer than the 0.31.0 llm-d 1.0 is tested with.
- The two-stack accounting test uses different engine versions and block sizes; it says nothing about which stack is faster.

## What to do with it

1. **Compute the prefix working set first.** If the distinct shared-prefix tokens fit in one replica's KV, a balanced Service gets the hits for free. If they exceed one replica but fit in the pool, prefix routing was worth +67% throughput here.
2. **Do not run the affinity filter on defaults for single-prefix bursts** (one system prompt, agent frameworks). Use a load-only profile for those, or validate the lock-in width under your real burst shape.
3. **Warm the precise index after every router restart**, or schedule restarts at low traffic.
4. **Plan capacity from engine counters** (`vllm:prefix_cache_hits_total / queries_total`), not from the router's predicted hit rate, whenever pods evict.
5. **Use FP8 KV as the cheapest capacity lever** (1.85 to 1.95x tokens per pod here) after an accuracy check on your model.
