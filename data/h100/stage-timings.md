# H100 stage timings: where each P/D request spends its time

Mean milliseconds per request, per measured point, from vLLM histograms scraped every 5 s on every pod
(`scripts/scrape.sh`) and differenced over each point's window (`scripts/layers.py`). Client TTFT mean is AIPerf's.

| column | meaning |
|---|---|
| P queue | time a request waits on a prefill pod before its prefill starts |
| P prefill | prefill compute on the prefill pod |
| P server TTFT | prefill pod's own TTFT (queue + prefill) |
| D queue (incl. KV wait) | time on the decode pod before its first decode step, including waiting for the KV to arrive |
| NIXL xfer | KV transfer time measured by vLLM's NIXL connector on the decode side |
| D server TTFT | decode pod's own TTFT |
| D decode total | decode time for the whole output |
| ITL (server) | server-side inter-token latency (mean) |
| residual | client TTFT - P server TTFT - D server TTFT: router, routing sidecar, network and histogram-window effects; can be negative |

Configuration names match `data/h100/*.points.json`.

### pd-2p2d-tp1  prefill pods=2 decode/agg pods=2  (ms, mean per request)
| shape | c | P queue | P prefill | P server TTFT | D queue (incl. KV wait) | NIXL xfer | D server TTFT | D decode total | ITL (server) | client TTFT mean | residual (router+sidecar+net) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| prompt-heavy | 16 | 9.7 | 130.8 | 219.6 | 55.6 | 34.1 | 90.4 | 1,817.2 | 9.1 | 240.3 | -69.7 |
| prompt-heavy | 64 | 353.5 | 236.5 | 679.9 | 181.9 | 59.3 | 289.9 | 3,254.7 | 16.6 | 818.0 | -151.8 |
| prompt-heavy | 256 | 7,013.4 | 166.3 | 7,375.1 | 2,868.7 | 69.3 | 3,201.5 | 3,491.1 | 17.5 | 10,702.7 | 126.1 |
| prompt-heavy | 512 | 20,135.8 | 141.8 | 20,640.0 | 3,386.0 | 74.3 | 3,546.2 | 3,637.7 | 18.3 | 24,595.5 | 409.3 |
| prompt-heavy | 1024 | 47,852.4 | 142.3 | 48,651.1 | 3,665.9 | 75.1 | 3,748.2 | 3,697.5 | 18.6 | 53,529.2 | 1,130.0 |
| chat | 16 | 0.0 | 68.5 | 93.9 | 42.5 | 24.3 | 70.6 | 4,265.8 | 8.6 | 181.1 | 16.7 |
| chat | 64 | 30.0 | 151.9 | 230.4 | 82.8 | 40.2 | 134.2 | 6,683.9 | 13.4 | 379.8 | 15.3 |
| chat | 256 | 226.2 | 147.1 | 447.3 | 1,366.2 | 44.3 | 1,728.1 | 11,657.1 | 23.3 | 2,204.0 | 28.5 |
| chat | 512 | 2,680.5 | 146.9 | 2,949.2 | 10,548.3 | 39.7 | 11,354.0 | 11,787.9 | 23.5 | 14,342.5 | 39.2 |
| chat | 1024 | 25,239.3 | 117.4 | 25,534.2 | 12,219.6 | 38.9 | 12,545.3 | 11,731.0 | 23.6 | 38,177.4 | 97.9 |

### pd-2p1d-dtp2-bs16  prefill pods=2 decode/agg pods=1  (ms, mean per request)
| shape | c | P queue | P prefill | P server TTFT | D queue (incl. KV wait) | NIXL xfer | D server TTFT | D decode total | ITL (server) | client TTFT mean | residual (router+sidecar+net) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| prompt-heavy | 16 | 22.1 | 215.5 | 278.3 | 146.2 | 102.4 | 276.4 | 1,748.7 | 8.7 | 587.6 | 32.9 |
| prompt-heavy | 64 | 493.0 | 315.0 | 925.8 | 345.9 | 175.8 | 760.2 | 3,693.7 | 17.9 | 1,741.4 | 55.3 |
| prompt-heavy | 256 | 1,408.5 | 262.7 | 1,898.6 | 2,183.4 | 351.4 | 3,509.7 | 12,628.0 | 60.6 | 5,469.8 | 61.5 |
| prompt-heavy | 512 | 14,711.6 | 253.6 | 15,357.0 | 3,046.3 | 299.9 | 3,849.4 | 12,478.9 | 62.7 | 19,322.3 | 115.9 |
| prompt-heavy | 1024 | 40,680.4 | 222.7 | 41,583.6 | 3,292.6 | 228.3 | 3,831.4 | 12,254.0 | 61.2 | 46,408.8 | 993.7 |
| chat | 16 | 0.0 | 74.8 | 98.9 | 51.3 | 26.6 | 89.0 | 3,730.9 | 7.5 | 207.6 | 19.7 |
| chat | 64 | 65.1 | 196.2 | 319.6 | 197.7 | 92.9 | 374.2 | 5,759.4 | 11.5 | 718.0 | 24.2 |
| chat | 256 | 588.8 | 254.2 | 984.3 | 476.3 | 177.0 | 1,114.6 | 9,370.2 | 19.7 | 2,153.0 | 54.2 |
| chat | 512 | 1,357.3 | 254.9 | 1,848.3 | 964.2 | 372.4 | 2,302.3 | 14,730.3 | 29.8 | 4,233.1 | 82.4 |
| chat | 1024 | 4,129.6 | 137.6 | 4,495.7 | 11,127.8 | 198.9 | 12,500.1 | 41,355.1 | 81.0 | 17,268.9 | 273.1 |

### pd-2p1d-dtp2-bs64-run2  prefill pods=2 decode/agg pods=1  (ms, mean per request)
| shape | c | P queue | P prefill | P server TTFT | D queue (incl. KV wait) | NIXL xfer | D server TTFT | D decode total | ITL (server) | client TTFT mean | residual (router+sidecar+net) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| prompt-heavy | 64 | 344.3 | 201.8 | 642.2 | 46.0 | 26.4 | 100.9 | 2,566.7 | 13.3 | 623.0 | -120.1 |
| prompt-heavy | 256 | 4,777.5 | 358.8 | 5,390.3 | 52.3 | 26.9 | 113.2 | 3,131.7 | 15.9 | 5,553.2 | 49.7 |
| prompt-heavy | 512 | 12,112.3 | 361.0 | 12,881.1 | 53.0 | 26.3 | 114.4 | 3,169.3 | 16.0 | 13,138.8 | 143.4 |
| prompt-heavy | 1024 | 26,913.4 | 362.3 | 27,992.9 | 54.3 | 28.1 | 117.2 | 3,224.6 | 16.2 | 28,345.7 | 235.6 |
| chat | 64 | 62.0 | 202.0 | 316.0 | 151.7 | 73.8 | 363.4 | 5,733.2 | 11.5 | 704.8 | 25.4 |
| chat | 256 | 588.7 | 256.1 | 997.1 | 646.4 | 226.7 | 1,385.1 | 9,444.1 | 19.5 | 2,436.4 | 54.2 |
| chat | 512 | 1,318.8 | 259.8 | 1,815.3 | 730.6 | 219.3 | 1,938.2 | 14,612.6 | 29.8 | 3,828.9 | 75.3 |
| chat | 1024 | 4,589.9 | 156.4 | 4,962.4 | 11,816.1 | 212.4 | 12,860.5 | 40,445.4 | 81.8 | 17,935.9 | 113.0 |

### pd-2p1d-dtp2-bs64-fp8kv  prefill pods=2 decode/agg pods=1  (ms, mean per request)
| shape | c | P queue | P prefill | P server TTFT | D queue (incl. KV wait) | NIXL xfer | D server TTFT | D decode total | ITL (server) | client TTFT mean | residual (router+sidecar+net) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chat | 256 | 352.1 | 235.3 | 719.7 | 196.3 | 49.0 | 439.7 | 8,512.0 | 17.5 | 1,221.0 | 61.5 |
| chat | 512 | 1,345.0 | 266.4 | 1,818.7 | 543.9 | 192.3 | 1,415.6 | 13,126.9 | 26.0 | 3,311.6 | 77.3 |
| chat | 1024 | 1,454.7 | 241.6 | 1,941.1 | 883.6 | 152.2 | 1,782.4 | 38,170.0 | 74.5 | 3,927.3 | 203.8 |
| prompt-heavy | 256 | 5,068.5 | 360.4 | 5,681.7 | 45.6 | 21.8 | 101.4 | 2,741.7 | 13.9 | 5,849.6 | 66.5 |
| prompt-heavy | 512 | 12,416.2 | 362.2 | 13,172.9 | 46.4 | 20.8 | 101.7 | 2,792.6 | 14.1 | 13,432.4 | 157.8 |
| prompt-heavy | 1024 | 26,953.2 | 363.6 | 28,022.8 | 47.3 | 22.3 | 104.1 | 2,824.3 | 14.2 | 28,762.6 | 635.7 |

