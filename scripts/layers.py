#!/usr/bin/env python3
"""Per-layer latency attribution for one P/D (or agg) run, per concurrency point.
  layers.py <metrics-dir> <driver-log> <run-label> [<aiperf-run-dir>]
metrics-dir: scrape.sh output (5 s samples of vLLM histograms per pod). driver-log: the log whose '== HH:MM:SS shape ... c=N'
lines mark point starts (queue.log, knobs.log, ext.log, rep.log, offload.log). Means are histogram sum/count deltas
inside each point window. Layers (mean per request):
  prefill pod: queue -> prefill compute  (server TTFT on prefill = queue + prefill)
  decode pod : queue (includes waiting for the remote KV, see V5) -> NIXL xfer -> decode (per-token ITL)
  client     : AIPerf TTFT; residual = client TTFT - prefill server TTFT - decode server TTFT (router + sidecar + network)
"""
import collections, datetime, json, pathlib, re, sys
mdir, dlog, label = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
adir = pathlib.Path(sys.argv[4]) if len(sys.argv) > 4 else None
LINE = re.compile(r'^(\d+) (\S+) (vllm:[a-z_]+)(\{[^}]*\})? (\S+)$')
S = collections.defaultdict(dict)   # (pod, metric) -> {ts: value} summed over labels
for f in mdir.glob("*.prom*"):
    op = __import__("gzip").open if f.suffix == ".gz" else open
    for line in op(f, "rt"):
        m = LINE.match(line.strip())
        if not m or not (m[3].endswith("_sum") or m[3].endswith("_count")): continue
        k = (m[2], m[3]); S[k][int(m[1])] = S[k].get(int(m[1]), 0) + float(m[5])
pods = sorted({p for p, _ in S}); role = lambda p: "prefill" if "-p" in p.split("-")[1][:1] or p.startswith("pd-p") else ("decode" if p.startswith("pd-d") else "agg")
day = datetime.datetime.fromtimestamp(min(min(v) for v in S.values()), datetime.timezone.utc).date()
starts = []
inrun = False
for line in open(dlog):
    if "results -> " in line: inrun = line.strip().endswith(label)
    m = re.match(r'^== (\d\d:\d\d:\d\d) (\S+) .* c=(\d+)', line)
    if m and inrun:
        t = datetime.datetime.combine(day, datetime.time.fromisoformat(m[1]), datetime.timezone.utc).timestamp()
        if starts and t < starts[-1][0]: t += 86400
        starts.append((t, m[2], int(m[3])))
def mean(pods_, name, a, b):
    s = c = 0
    for p in pods_:
        for suf, acc in (("_sum", "s"), ("_count", "c")):
            ser = S.get((p, name + suf), {}); ks = sorted(ser)
            va = max([ser[k] for k in ks if k <= a] or [0]); vb = max([ser[k] for k in ks if k <= b] or [0])
            if acc == "s": s += vb - va
            else: c += vb - va
    return 1e3 * s / c if c > 0 else None
P = [p for p in pods if role(p) == "prefill"]; D = [p for p in pods if role(p) == "decode"] or [p for p in pods if role(p) == "agg"]
F = lambda x: "-" if x is None else f"{x:,.1f}"
print(f"### {label}  prefill pods={len(P)} decode/agg pods={len(D)}  (ms, mean per request)")
print("| shape | c | P queue | P prefill | P server TTFT | D queue (incl. KV wait) | NIXL xfer | D server TTFT | D decode total | ITL (server) | client TTFT mean | residual (router+sidecar+net) |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for i, (t, shape, c) in enumerate(starts):
    a = t + 10; b = (starts[i + 1][0] if i + 1 < len(starts) else max(max(v) for v in S.values())) - 5
    pq, pp, pt = (mean(P, n, a, b) for n in ("vllm:request_queue_time_seconds", "vllm:request_prefill_time_seconds", "vllm:time_to_first_token_seconds")) if P else (None,)*3
    dq, dx, dt, dd, itl = (mean(D, n, a, b) for n in ("vllm:request_queue_time_seconds", "vllm:nixl_xfer_time_seconds", "vllm:time_to_first_token_seconds", "vllm:request_decode_time_seconds", "vllm:inter_token_latency_seconds"))
    ct = None
    if adir:
        pj = next(iter(adir.glob(f"{shape}/c{c}/profile_export_aiperf.json")), None)
        if pj:
            j = json.load(open(pj)); stack = [j]
            while stack:
                o = stack.pop()
                if isinstance(o, dict):
                    if isinstance(o.get("time_to_first_token"), dict): ct = o["time_to_first_token"].get("avg"); break
                    stack.extend(o.values())
                elif isinstance(o, list): stack.extend(o)
    res = (ct - (pt or 0) - (dt or 0)) if (ct is not None and P) else None
    print(f"| {shape} | {c} | {F(pq)} | {F(pp)} | {F(pt)} | {F(dq)} | {F(dx)} | {F(dt)} | {F(dd)} | {F(itl)} | {F(ct)} | {F(res)} |")
