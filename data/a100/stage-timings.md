# A100 stage timings: where each request spends its time

Same method and columns as [`../h100/stage-timings.md`](../h100/stage-timings.md): mean milliseconds per request from
vLLM histograms scraped every 5 s on every pod, differenced over each point's window (`scripts/layers.py`). For the
aggregated arms (A0, A1) there is no prefill pod, so only the decode/agg columns are filled. Configuration names match
`data/a100/*.points.json`. These numbers are never compared with H100.

### a0-agg-dp8  prefill pods=0 decode/agg pods=1  (ms, mean per request)
| shape | c | P queue | P prefill | P server TTFT | D queue (incl. KV wait) | NIXL xfer | D server TTFT | D decode total | ITL (server) | client TTFT mean | residual (router+sidecar+net) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| prompt-heavy | 16 | - | - | - | 129.2 | - | 831.4 | 3,474.5 | 17.7 | 759.0 | - |
| prompt-heavy | 64 | - | - | - | 848.1 | - | 1,728.8 | 5,222.4 | 26.5 | 1,531.8 | - |
| prompt-heavy | 256 | - | - | - | 3,162.9 | - | 3,629.9 | 16,786.7 | 82.6 | 3,666.7 | - |
| prompt-heavy | 512 | - | - | - | 5,087.3 | - | 6,536.8 | 34,249.5 | 172.1 | 6,802.5 | - |
| prompt-heavy | 1024 | - | - | - | 25,622.8 | - | 27,478.7 | 64,853.7 | 325.9 | 28,151.8 | - |
| chat | 16 | - | - | - | 6.6 | - | 340.5 | 6,461.5 | 12.9 | 345.6 | - |
| chat | 64 | - | - | - | 97.4 | - | 642.6 | 7,650.6 | 15.3 | 655.0 | - |
| chat | 256 | - | - | - | 706.9 | - | 1,164.8 | 11,119.0 | 22.2 | 1,238.4 | - |
| chat | 512 | - | - | - | 1,270.0 | - | 2,026.6 | 17,328.3 | 34.7 | 2,073.6 | - |
| chat | 1024 | - | - | - | 2,584.5 | - | 3,380.8 | 29,586.7 | 59.3 | 3,578.3 | - |


### a1-agg-8x-tp1  prefill pods=0 decode/agg pods=8  (ms, mean per request)
| shape | c | P queue | P prefill | P server TTFT | D queue (incl. KV wait) | NIXL xfer | D server TTFT | D decode total | ITL (server) | client TTFT mean | residual (router+sidecar+net) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| prompt-heavy | 16 | - | - | - | 13.6 | - | 407.1 | 2,277.4 | 11.5 | 401.0 | - |
| prompt-heavy | 64 | - | - | - | 347.4 | - | 775.5 | 4,260.5 | 20.7 | 782.5 | - |
| prompt-heavy | 256 | - | - | - | 1,676.7 | - | 2,264.5 | 11,522.6 | 58.1 | 2,368.4 | - |
| prompt-heavy | 512 | - | - | - | 11,594.5 | - | 12,826.3 | 13,541.1 | 67.5 | 13,331.5 | - |
| prompt-heavy | 1024 | - | - | - | 35,335.4 | - | 36,521.7 | 13,895.5 | 69.5 | 37,283.0 | - |
| chat | 16 | - | - | - | 0.0 | - | 145.3 | 5,000.9 | 10.0 | 150.0 | - |
| chat | 64 | - | - | - | 48.3 | - | 298.4 | 7,473.1 | 15.0 | 305.9 | - |
| chat | 256 | - | - | - | 281.9 | - | 641.0 | 11,968.9 | 24.4 | 656.8 | - |
| chat | 512 | - | - | - | 598.7 | - | 1,017.2 | 17,000.8 | 34.3 | 1,049.2 | - |
| chat | 1024 | - | - | - | 3,882.8 | - | 4,486.4 | 25,805.1 | 51.8 | 4,536.8 | - |


### a2-pd-4p2d-dtp2-bs64  prefill pods=4 decode/agg pods=2  (ms, mean per request)
| shape | c | P queue | P prefill | P server TTFT | D queue (incl. KV wait) | NIXL xfer | D server TTFT | D decode total | ITL (server) | client TTFT mean | residual (router+sidecar+net) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| prompt-heavy | 16 | 27.0 | 271.5 | 335.7 | 53.8 | 22.9 | 107.1 | 2,105.3 | 10.6 | 409.4 | -33.5 |
| prompt-heavy | 64 | 766.2 | 401.1 | 1,288.1 | 52.0 | 26.1 | 120.4 | 3,010.5 | 15.4 | 1,174.2 | -234.3 |
| prompt-heavy | 256 | 9,208.9 | 462.9 | 9,866.5 | 57.2 | 25.8 | 126.8 | 3,275.0 | 16.5 | 10,175.0 | 181.7 |
| prompt-heavy | 512 | 20,676.5 | 464.1 | 21,455.4 | 57.5 | 25.4 | 126.8 | 3,366.2 | 16.9 | 22,451.2 | 868.9 |
| prompt-heavy | 1024 | 44,999.8 | 466.3 | 46,072.6 | 58.5 | 25.6 | 127.6 | 3,329.6 | 16.7 | 47,209.3 | 1,009.1 |
| chat | 16 | 0.4 | 105.7 | 163.3 | 24.0 | 8.4 | 60.7 | 5,072.3 | 10.2 | 196.4 | -27.7 |
| chat | 64 | 109.1 | 169.2 | 344.3 | 30.5 | 9.7 | 83.0 | 7,429.7 | 14.9 | 384.6 | -42.7 |
| chat | 256 | 436.7 | 175.8 | 702.8 | 52.1 | 12.6 | 181.5 | 11,461.9 | 23.2 | 939.6 | 55.3 |
| chat | 512 | 893.9 | 173.6 | 1,209.2 | 71.8 | 15.3 | 879.1 | 15,327.6 | 30.6 | 2,314.3 | 225.9 |
| chat | 1024 | 1,840.3 | 166.9 | 2,212.7 | 11,793.5 | 126.4 | 13,299.4 | 17,255.9 | 34.6 | 16,352.8 | 840.6 |


### a3-pd-2p3d-dtp2-bs64  prefill pods=2 decode/agg pods=3  (ms, mean per request)
| shape | c | P queue | P prefill | P server TTFT | D queue (incl. KV wait) | NIXL xfer | D server TTFT | D decode total | ITL (server) | client TTFT mean | residual (router+sidecar+net) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| prompt-heavy | 16 | 51.4 | 277.5 | 381.0 | 37.6 | 19.9 | 82.5 | 1,892.5 | 9.5 | 481.9 | 18.4 |
| prompt-heavy | 64 | 4,242.9 | 464.7 | 4,883.6 | 40.1 | 20.7 | 88.4 | 2,067.1 | 10.4 | 4,863.9 | -108.1 |
| prompt-heavy | 256 | 21,099.7 | 466.4 | 21,910.6 | 40.8 | 21.0 | 91.2 | 2,154.7 | 10.9 | 23,333.9 | 1,332.1 |
| prompt-heavy | 512 | 45,185.2 | 468.3 | 46,229.6 | 42.6 | 21.4 | 92.9 | 2,199.8 | 11.1 | 48,154.9 | 1,832.4 |
| prompt-heavy | 1024 | 94,052.8 | 469.3 | 95,644.1 | 43.3 | 21.5 | 94.7 | 2,241.0 | 11.3 | 97,762.0 | 2,023.2 |
| chat | 16 | 4.6 | 106.2 | 141.2 | 24.4 | 9.3 | 58.8 | 4,629.4 | 9.3 | 216.0 | 16.0 |
| chat | 64 | 296.7 | 185.6 | 569.6 | 32.0 | 10.2 | 78.0 | 6,538.9 | 13.1 | 543.1 | -104.5 |
| chat | 256 | 936.1 | 174.1 | 1,242.2 | 47.5 | 12.6 | 128.5 | 9,455.1 | 19.0 | 1,420.8 | 50.1 |
| chat | 512 | 2,362.4 | 236.4 | 2,839.2 | 60.4 | 14.0 | 233.2 | 12,225.6 | 24.6 | 3,178.7 | 106.2 |
| chat | 1024 | 12,097.8 | 260.7 | 12,786.4 | 67.4 | 15.2 | 380.7 | 12,876.1 | 25.8 | 13,584.0 | 417.0 |


