#!/usr/bin/env python3
"""NIXL transfer stats from scrape.sh output (decode pods pull KV, so the consumer side carries the histograms).
  nixl.py <metrics-dir> [<metrics-dir> ...]
Per run: totals over the run plus per-30 s windows: transfers/s, mean xfer ms, mean post ms, post share,
MB/transfer, descriptors/transfer, effective GB/s per transfer, and mean KV-fetch stage gauges.
"""
import collections, pathlib, re, sys
LINE = re.compile(r'^(\d+) (\S+) (vllm:[a-z_]+)(\{[^}]*\})? (\S+)$')
KEYS = ["vllm:nixl_xfer_time_seconds_sum", "vllm:nixl_xfer_time_seconds_count", "vllm:nixl_post_time_seconds_sum",
        "vllm:nixl_bytes_transferred_sum", "vllm:nixl_num_descriptors_sum", "vllm:nixl_num_failed_transfers_total"]
def load(d):
    series = collections.defaultdict(dict); gauges = collections.defaultdict(list)
    for f in pathlib.Path(d).glob("*.prom"):
        for line in open(f):
            m = LINE.match(line.strip())
            if not m: continue
            ts, pod, name, labels, val = int(m[1]), m[2], m[3], m[4] or "", float(m[5])
            if name in KEYS: series[(pod, name)][ts] = series[(pod, name)].get(ts, 0) + val
            if name == "vllm:num_requests_kv_fetch_by_stage" or name in ("vllm:num_requests_waiting", "vllm:num_requests_running", "vllm:kv_cache_usage_perc"):
                gauges[(pod, name, labels)].append((ts, val))
    return series, gauges
def delta(s, a, b):
    ks = sorted(s); va = max([s[k] for k in ks if k <= a] or [0]); vb = max([s[k] for k in ks if k <= b] or [0]); return vb - va
for d in sys.argv[1:]:
    series, gauges = load(d)
    pods = sorted({p for p, _ in series if any(v for v in series[(p, "vllm:nixl_xfer_time_seconds_count")].values())})
    print(f"## {pathlib.Path(d).name}  consumer pods: {', '.join(pods)}")
    tsall = sorted({t for (p, n), s in series.items() for t in s})
    if not tsall: print("no data"); continue
    t0, t1 = tsall[0], tsall[-1]
    def agg(a, b):
        tot = {k: sum(delta(series[(p, k)], a, b) for p in pods) for k in KEYS}
        n = tot["vllm:nixl_xfer_time_seconds_count"]
        if n <= 0: return None
        return dict(n=n, rate=n / max(b - a, 1), xfer_ms=1e3 * tot["vllm:nixl_xfer_time_seconds_sum"] / n,
                    post_ms=1e3 * tot["vllm:nixl_post_time_seconds_sum"] / n,
                    mb=tot["vllm:nixl_bytes_transferred_sum"] / n / 1e6, desc=tot["vllm:nixl_num_descriptors_sum"] / n,
                    fail=tot["vllm:nixl_num_failed_transfers_total"])
    T = agg(t0 - 1, t1)
    if T:
        print(f"total: transfers={T['n']:.0f} mean xfer {T['xfer_ms']:.1f} ms, post {T['post_ms']:.1f} ms ({100*T['post_ms']/T['xfer_ms']:.0f}%), "
              f"{T['mb']:.0f} MB/transfer, {T['desc']:.0f} descriptors/transfer, {T['mb']/T['xfer_ms']:.2f} GB/s per transfer, failed={T['fail']:.0f}")
    print("| window start | transfers/s | xfer ms | post ms | post % | MB/xfer | desc/xfer | GB/s/xfer | D waiting (sum) | D running (sum) | KV fetch in progress |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for a in range(t0, t1, 30):
        w = agg(a, a + 30)
        if not w: continue
        def gmean(name, lab=None):
            vals = [v for (p, n, l), pts in gauges.items() if n == name and p in pods and (lab is None or lab in l) for (t, v) in pts if a <= t < a + 30]
            per = len(pods) or 1
            return sum(vals) / max(len(vals) / per, 1) if vals else float('nan')
        print(f"| +{a - t0}s | {w['rate']:.2f} | {w['xfer_ms']:.1f} | {w['post_ms']:.1f} | {100*w['post_ms']/w['xfer_ms']:.0f} | {w['mb']:.0f} | {w['desc']:.0f} | {w['mb']/w['xfer_ms']:.2f} | "
              f"{gmean('vllm:num_requests_waiting'):.1f} | {gmean('vllm:num_requests_running'):.1f} | {gmean('vllm:num_requests_kv_fetch_by_stage', 'in_progress'):.1f} |")
