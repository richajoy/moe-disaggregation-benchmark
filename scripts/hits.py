#!/usr/bin/env python3
"""Engine prefix-cache hit rate per run from scrape.sh files: GPU tier (vllm:prefix_cache_hits/queries) and
connector/CPU tier (vllm:external_prefix_cache_hits/queries), deltas over the whole scrape.
  hits.py <metrics-dir> [...]"""
import collections, pathlib, re, sys, gzip
L = re.compile(r'^(\d+) (\S+) (vllm:[a-z_]+)(\{[^}]*\})? (\S+)$')
for d in sys.argv[1:]:
    s = collections.defaultdict(dict)
    for f in pathlib.Path(d).glob("*.prom*"):
        op = gzip.open if f.suffix == ".gz" else open
        for line in op(f, "rt"):
            m = L.match(line.strip())
            if m and m[3] in ("vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total", "vllm:external_prefix_cache_hits_total",
                              "vllm:external_prefix_cache_queries_total", "vllm:num_preemptions_total"):
                s[(m[2], m[3])][int(m[1])] = float(m[5])
    tot = collections.Counter()
    for (pod, name), ser in s.items():
        ks = sorted(ser); tot[name] += ser[ks[-1]] - ser[ks[0]]
    g = tot["vllm:prefix_cache_hits_total"] / tot["vllm:prefix_cache_queries_total"] if tot["vllm:prefix_cache_queries_total"] else float("nan")
    e = tot["vllm:external_prefix_cache_hits_total"] / tot["vllm:external_prefix_cache_queries_total"] if tot["vllm:external_prefix_cache_queries_total"] else float("nan")
    print(f"{pathlib.Path(d).name}: GPU-tier hit {g:.1%} ({tot['vllm:prefix_cache_hits_total']:.0f}/{tot['vllm:prefix_cache_queries_total']:.0f} tokens); "
          f"external/CPU-tier hit {e:.1%} ({tot['vllm:external_prefix_cache_hits_total']:.0f}/{tot['vllm:external_prefix_cache_queries_total']:.0f}); preemptions {tot['vllm:num_preemptions_total']:.0f}")
