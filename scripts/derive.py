#!/usr/bin/env python3
"""Every derived number in the docs comes from here: goodput rankings, frontiers, per-point tables, self-checks.

  python3 scripts/derive.py            # writes data/derived/{goodput,frontiers}.json and data/derived/TABLES.md

Inputs: data/configs.json, data/<node>/*.points.json, data/requests/<node>/*.csv.gz (stdlib only).

Rules
  - SLO: TTFT <= 2,000 ms and that request's mean ITL <= 50 ms. SLO-ok counts measured requests only (warm-up excluded).
  - goodput = req/s x SLO-ok / GPUs the layout uses (every prefill and decode GPU for P/D).
  - Headline ranking ("goodput at 90%"): best goodput over the concurrency ladder among points where at least 90%
    of requests met the SLO (the DistServe attainment convention). Secondary ranking ("peak goodput"): best goodput
    at any attainment. The ladder is coarse (16, 64, 256, 512, 1,024), so true capacity lies between rungs.
  - Interactivity (tok/s/user) = 1000 / ITL p90, the slow tail.
  - Frontiers pool every non-knob configuration of one node and one shape. For each interactivity floor:
      unconstrained   = most output tok/s/GPU at or above the floor (the usual TTFT-blind throughput-vs-interactivity view)
      slo             = same, only points with TTFT p90 <= 2 s (the headline, SLO-respecting view)
      slo_ok_90       = same, only points where >= 90% of requests met the SLO
      ttft_p50_2s     = same, only points with TTFT p50 <= 2 s (looser; shown for comparison)
  - The 3-GPU knob arms are excluded from rankings and frontiers (different GPU count, two points each).
  - Cost (H100 only): $/M output tokens = REF_USD_PER_GPU_HR x 1e6 / (3600 x output tok/s/GPU), with the reference
    rate $2.00/GPU-hour (InferenceX "Rent - 3 Year Commit" input for H100). An assumption, not a quote. No A100 rate.
"""
import csv, gzip, io, json, math, os, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
D = ROOT / "data"
FLOORS = [20, 35, 50, 75, 100]
# Re-rank for your own target: SLO_TTFT_MS=1000 SLO_ITL_MS=30 MIN_SLO_OK=0.95 python3 scripts/derive.py
SLO_TTFT = float(os.environ.get("SLO_TTFT_MS", 2000))
SLO_ITL = float(os.environ.get("SLO_ITL_MS", 50))
MIN_SLO_OK = float(os.environ.get("MIN_SLO_OK", 0.9))
DEFAULT_SLO = (SLO_TTFT, SLO_ITL) == (2000.0, 50.0)
REF_USD_PER_GPU_HR = 2.00
CFG = {c["config"]: c for c in json.load(open(D / "configs.json"))}


def load():
    pts = []
    for node in ("h100", "a100"):
        for f in sorted((D / node).glob("*.points.json")):
            for p in json.load(open(f)):
                p["label"] = CFG[p["config"]]["label"]
                p["family"] = CFG[p["config"]]["family"]
                pts.append(p)
    return pts


def selfcheck(pts):
    """Recompute SLO-ok and ITL p90 from the per-request files. With the default SLO they must match the points files;
    with a custom SLO (env SLO_TTFT_MS / SLO_ITL_MS) SLO-ok and goodput are replaced by the recomputed values."""
    from build_points import pct  # same percentile definition
    worst = 0.0
    for node in ("h100", "a100"):
        for f in sorted((D / "requests" / node).glob("*.csv.gz")):
            cfg = f.name[:-len(".csv.gz")]
            rows = list(csv.DictReader(io.TextIOWrapper(gzip.open(f), newline="")))
            for p in (q for q in pts if q["config"] == cfg):
                m = [r for r in rows if r["shape"] == p["shape"] and int(r["c"]) == p["c"] and r["phase"] == "m"]
                ok = sum(1 for r in m if float(r["ttft_ms"]) <= SLO_TTFT and (r["itl_ms"] == "" or float(r["itl_ms"]) <= SLO_ITL))
                itl = [float(r["itl_ms"]) for r in m if r["itl_ms"] != ""]
                worst = max(worst, abs(pct(itl, 90) - p["itl_p90_ms"]) / p["itl_p90_ms"])
                if DEFAULT_SLO:
                    worst = max(worst, abs(ok / len(m) - p["slo_ok"]))
                else:
                    p["slo_ok"] = ok / len(m)
                    p["goodput_req_s_gpu"] = p["req_s"] * p["slo_ok"] / p["gpus"]
    return worst


def usd(p):
    return REF_USD_PER_GPU_HR * 1e6 / (3600 * p["out_tok_s_gpu"]) if p["node"] == "h100" and p["out_tok_s_gpu"] else None


def brief(p):
    return None if p is None else dict(config=p["config"], c=p["c"], out_tok_s_gpu=p["out_tok_s_gpu"],
                                       tok_s_user_p90=p["tok_s_user_p90"], ttft_p50_ms=p["ttft_p50_ms"],
                                       ttft_p90_ms=p["ttft_p90_ms"], slo_ok=p["slo_ok"], eff_conc=p["eff_conc"])


def frontiers(pts):
    out = {}
    for node, shape in sorted({(p["node"], p["shape"]) for p in pts}):
        pool = [p for p in pts if p["node"] == node and p["shape"] == shape and p["family"] != "knob"]
        rows = []
        for fl in FLOORS:
            def best(pred):
                ok = [p for p in pool if p["tok_s_user_p90"] >= fl and pred(p)]
                return max(ok, key=lambda p: p["out_tok_s_gpu"]) if ok else None
            a, s = best(lambda p: True), best(lambda p: p["ttft_p90_ms"] <= SLO_TTFT)
            rows.append(dict(floor=fl, unconstrained=brief(a), slo=brief(s),
                             slo_ok_90=brief(best(lambda p: p["slo_ok"] >= MIN_SLO_OK)),
                             ttft_p50_2s=brief(best(lambda p: p["ttft_p50_ms"] <= SLO_TTFT)),
                             overstatement=(a["out_tok_s_gpu"] / s["out_tok_s_gpu"] - 1) if a and s else None))
        out[f"{node}/{shape}"] = rows
    return out


def goodput(pts, min_slo=0.0):
    out = {}
    for node, shape in sorted({(p["node"], p["shape"]) for p in pts}):
        best = {}
        for p in pts:
            if p["node"] == node and p["shape"] == shape and p["family"] != "knob" and p["slo_ok"] >= min_slo:
                if p["config"] not in best or p["goodput_req_s_gpu"] > best[p["config"]]["goodput_req_s_gpu"]:
                    best[p["config"]] = p
        out[f"{node}/{shape}"] = sorted(best.values(), key=lambda p: -p["goodput_req_s_gpu"])
    return out


F = lambda x, n=0: "-" if x is None else f"{x:,.{n}f}"
P = lambda x: "-" if x is None else f"{100 * x:.1f}%"


def lat(p, k):
    return f"{F(p[k + '_p50_ms'], 0 if k == 'ttft' else 1)} / **{F(p[k + '_p90_ms'], 0 if k == 'ttft' else 1)}** / {F(p[k + '_p99_ms'], 0 if k == 'ttft' else 1)}"


def ranktable(L, gp):
    for key, rows in gp.items():
        L += [f"### {key}", ""]
        if not rows:
            L += ["No configuration had a point where at least 90% of requests met the SLO.", ""]
            continue
        L += ["| rank | configuration | goodput req/s/GPU | at c | SLO-ok | TTFT p50 / p90 / p99 ms | ITL p50 / p90 / p99 ms | out tok/s/GPU | eff |",
              "|---|---|---|---|---|---|---|---|---|"]
        for i, p in enumerate(rows, 1):
            L.append(f"| {i} | {p['label']} (`{p['config']}`) | **{p['goodput_req_s_gpu']:.2f}** | {p['c']} | {P(p['slo_ok'])} | {lat(p, 'ttft')} | {lat(p, 'itl')} | {F(p['out_tok_s_gpu'])} | {p['eff_conc']:.2f} |")
        L.append("")


def tables(pts, gp, gp90, fr):
    L = ["# Derived tables", "",
         "Generated by `scripts/derive.py` from `data/`. Do not edit by hand.", "",
         "Conventions: latency columns are p50 / **p90** / p99 in ms over measured requests (warm-up excluded); "
         "tok/s/user = 1000 / ITL p90 (slow tail); SLO = TTFT <= 2 s and per-request mean ITL <= 50 ms; "
         "goodput = req/s x SLO-ok / GPUs; eff = effective concurrency (Little's law) / requested concurrency. "
         "H100 $/M output tokens at a $2.00/GPU-hour reference rate (assumption; see docs/00-method.md section 6). "
         "H100 and A100 are separate tables and are never compared.", ""]
    L += ["## 1. Goodput at 90% attainment (headline ranking)", "",
          "Best point per configuration among points where at least 90% of requests met the SLO.", ""]
    ranktable(L, gp90)
    L += ["## 1b. Peak goodput at any attainment", "",
          "Best point per configuration with no attainment floor. A peak here can sit where many requests miss the SLO; read the SLO-ok column.", ""]
    ranktable(L, gp)
    L += ["## 2. Frontiers: TTFT-blind vs SLO-respecting", "",
          "Most output tok/s/GPU at or above each interactivity floor (tok/s/user = 1000 / ITL p90). "
          "'TTFT-blind' ignores TTFT; 'SLO-respecting' keeps only points with TTFT p90 <= 2 s.", ""]
    for key, rows in fr.items():
        L += [f"### {key}", "", "| floor (tok/s/user) | TTFT-blind best | its TTFT p90 / SLO-ok | SLO-respecting best | its TTFT p90 / SLO-ok | TTFT-blind overstates by |",
              "|---|---|---|---|---|---|"]
        for r in rows:
            a, s = r["unconstrained"], r["slo"]
            fa = f"{F(a['out_tok_s_gpu'])} (`{a['config']}` c{a['c']})" if a else "none"
            fs = f"{F(s['out_tok_s_gpu'])} (`{s['config']}` c{s['c']})" if s else "none"
            ta = f"{F(a['ttft_p90_ms'])} ms / {P(a['slo_ok'])}" if a else "-"
            ts = f"{F(s['ttft_p90_ms'])} ms / {P(s['slo_ok'])}" if s else "-"
            ov = "-" if r["overstatement"] is None else f"{100 * r['overstatement']:+.1f}%"
            L.append(f"| >= {r['floor']} | {fa} | {ta} | {fs} | {ts} | {ov} |")
        L.append("")
    L += ["## 3. Every measured point", ""]
    for cfg in CFG:
        ps = sorted((p for p in pts if p["config"] == cfg), key=lambda p: (p["shape"], p["c"]))
        if not ps:
            continue
        c0 = CFG[cfg]
        L += [f"### `{cfg}`: {c0['label']}", "",
              f"{c0['node'].upper()}, {c0['gpus']} GPUs; router: {c0['router']}; KV {c0['kv_dtype'].upper()}, block {c0['block_size']}"
              + (f"; {c0['note']}" if c0["note"] else "") + ".", "",
              "| shape | c | eff | req/s | out tok/s/GPU | tok/s/user (p90) | TTFT p50 / p90 / p99 ms | ITL p50 / p90 / p99 ms | SLO-ok | SLO-ok incl. warm-up | goodput req/s/GPU |"
              + (" $/M out tok at $2.00/GPU-h |" if c0["node"] == "h100" else ""),
              "|---|---|---|---|---|---|---|---|---|---|---|" + ("---|" if c0["node"] == "h100" else "")]
        for p in ps:
            L.append(f"| {p['shape']} | {p['c']} | {p['eff_conc']:.2f} | {p['req_s']:.2f} | {F(p['out_tok_s_gpu'])} | {p['tok_s_user_p90']:.1f} | {lat(p, 'ttft')} | {lat(p, 'itl')} | {P(p['slo_ok'])} | {P(p['slo_ok_incl_warmup'])} | {p['goodput_req_s_gpu']:.2f} |"
                     + (f" {usd(p):.3f} |" if c0["node"] == "h100" else ""))
        L.append("")
    return "\n".join(L)


def main():
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    pts = load()
    worst = selfcheck(pts)
    assert worst < 1e-6, f"self-check failed: {worst}"
    gp, gp90, fr = goodput(pts), goodput(pts, MIN_SLO_OK), frontiers(pts)
    if not DEFAULT_SLO or MIN_SLO_OK != 0.9:
        print(f"custom target: TTFT <= {SLO_TTFT:g} ms, ITL <= {SLO_ITL:g} ms, attainment >= {MIN_SLO_OK:g}; headings in TABLES.md still name the defaults")
    (D / "derived").mkdir(exist_ok=True)
    G = lambda g: {k: [dict(brief(p), goodput_req_s_gpu=p["goodput_req_s_gpu"], slo_ok_incl_warmup=p["slo_ok_incl_warmup"]) for p in v] for k, v in g.items()}
    json.dump({"at_90pct_attainment": G(gp90), "peak_any_attainment": G(gp)}, open(D / "derived" / "goodput.json", "w"), indent=1)
    json.dump(fr, open(D / "derived" / "frontiers.json", "w"), indent=1)
    open(D / "derived" / "TABLES.md", "w").write(tables(pts, gp, gp90, fr) + "\n")
    print(f"{len(pts)} points; self-check max delta {worst:.1e}; wrote data/derived/")


if __name__ == "__main__":
    main()
