#!/usr/bin/env python3
"""Turn AIPerf run directories into one points file and one compact per-request file per configuration.

  build_points.py --node h100 --gpus 4 --config agg-dp4 RUN_DIR [RUN_DIR ...]

RUN_DIR is a sweep directory written by scripts/sweep3.sh or scripts/agentic.sh: <shape>/c<N>/profile_export_aiperf.json
plus the per-request export, either <shape>/c<N>/profile_export.jsonl[.gz] or a records-jsonl.tgz at the run root.

Writes (stdlib only):
  data/<node>/<config>.points.json        one record per (shape, concurrency)
  data/requests/<node>/<config>.csv.gz    one row per request: shape, c, phase, ttft_ms, itl_ms, isl, osl, e2e_ms

Rules (see docs/00-method.md):
  - Warm-up requests are excluded. AIPerf tags every record with metadata.benchmark_phase ("warmup" or "profiling");
    only "profiling" records count toward SLO-ok and the percentiles.
  - SLO-ok = share of measured requests with TTFT <= SLO_TTFT_MS (2,000) and that request's mean ITL <= SLO_ITL_MS (50).
  - Percentiles are numpy-style linear interpolation over per-request values (the convention AIPerf uses for its own
    p50 / p99, which this reproduces exactly). ITL per request is AIPerf's inter_token_latency:
    (request latency - TTFT) / (output tokens - 1).
  - tok_s_user_p90 = 1000 / ITL p90: the slow-tail streaming speed (the InferenceX convention), not AIPerf's
    per-user p90, which is the fast tail.
  - req/s, output tok/s and mean request latency come from AIPerf's summary for the measured phase.
"""
import argparse, csv, gzip, io, json, math, os, pathlib, tarfile

SLO_TTFT = float(os.environ.get("SLO_TTFT_MS", 2000))
SLO_ITL = float(os.environ.get("SLO_ITL_MS", 50))
ROOT = pathlib.Path(__file__).resolve().parent.parent


def pct(vals, q):
    """numpy.percentile(..., method='linear') without numpy."""
    v = sorted(vals)
    if not v:
        return None
    k = (len(v) - 1) * q / 100.0
    f, c = math.floor(k), math.ceil(k)
    return v[f] if f == c else v[f] + (v[c] - v[f]) * (k - f)


def parse(lines):
    out = []
    for line in lines:
        try:
            r = json.loads(line)
        except ValueError:
            continue
        m, md = r.get("metrics", r), r.get("metadata", {})
        val = lambda k: (m.get(k) or {}).get("value") if isinstance(m.get(k), dict) else m.get(k)
        out.append(dict(phase=md.get("benchmark_phase"), ttft=val("time_to_first_token"), itl=val("inter_token_latency"),
                        isl=val("input_sequence_length"), osl=val("output_sequence_length"), e2e=val("request_latency")))
    return out


def records(run, shape, c):
    d = run / shape / f"c{c}"
    for name in ("profile_export.jsonl.gz", "profile_export.jsonl"):
        f = d / name
        if f.exists():
            with (gzip.open(f, "rt") if name.endswith(".gz") else open(f)) as fh:
                return parse(fh)
    tgz = run / "records-jsonl.tgz"
    if tgz.exists():
        with tarfile.open(tgz) as tf:
            for mem in tf.getmembers():
                if mem.name.lstrip("./") == f"{shape}/c{c}/profile_export.jsonl":
                    return parse(io.TextIOWrapper(tf.extractfile(mem)))
    raise SystemExit(f"no per-request export for {run}/{shape}/c{c}")


def summary(prof, key):
    m = prof.get(key)
    return m.get("avg") if isinstance(m, dict) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", required=True, choices=["h100", "a100"])
    ap.add_argument("--gpus", required=True, type=int, help="GPUs the layout uses (all prefill + decode GPUs for P/D)")
    ap.add_argument("--config", required=True)
    ap.add_argument("runs", nargs="+", type=pathlib.Path)
    a = ap.parse_args()
    pts, rows = [], []
    for run in a.runs:
        for pj in sorted(run.glob("*/c*/profile_export_aiperf.json"), key=lambda p: (p.parts[-3], int(p.parts[-2][1:]))):
            shape, c = pj.parts[-3], int(pj.parts[-2][1:])
            prof = json.load(open(pj))
            rec = [r for r in records(run, shape, c) if r["ttft"] is not None]
            meas = [r for r in rec if r["phase"] == "profiling"]
            warm = [r for r in rec if r["phase"] == "warmup"]
            ok = lambda r: r["ttft"] <= SLO_TTFT and (r["itl"] is None or r["itl"] <= SLO_ITL)
            itl = [r["itl"] for r in meas if r["itl"] is not None]
            ttft = [r["ttft"] for r in meas]
            rps, out, e2e = summary(prof, "request_throughput"), summary(prof, "output_token_throughput"), summary(prof, "request_latency")
            slo = sum(map(ok, meas)) / len(meas)
            p = dict(config=a.config, node=a.node, gpus=a.gpus, shape=shape, c=c,
                     n_measured=len(meas), n_warmup=len(warm),
                     eff_conc=rps * e2e / 1000 / c, req_s=rps, out_tok_s=out, out_tok_s_gpu=out / a.gpus,
                     ttft_p50_ms=pct(ttft, 50), ttft_p90_ms=pct(ttft, 90), ttft_p99_ms=pct(ttft, 99),
                     itl_p50_ms=pct(itl, 50), itl_p90_ms=pct(itl, 90), itl_p99_ms=pct(itl, 99),
                     tok_s_user_p90=1000 / pct(itl, 90),
                     slo_ok=slo, slo_ok_incl_warmup=sum(map(ok, rec)) / len(rec),
                     goodput_req_s_gpu=rps * slo / a.gpus)
            pts.append(p)
            for r in rec:
                rows.append([shape, c, "w" if r["phase"] == "warmup" else "m"] +
                            [("" if r[k] is None else f"{r[k]:.6f}") for k in ("ttft", "itl")] +
                            [("" if r[k] is None else int(r[k])) for k in ("isl", "osl")] +
                            ["" if r["e2e"] is None else f"{r['e2e']:.6f}"])
    (ROOT / "data" / a.node).mkdir(parents=True, exist_ok=True)
    json.dump(pts, open(ROOT / "data" / a.node / f"{a.config}.points.json", "w"), indent=1)
    rq = ROOT / "data" / "requests" / a.node
    rq.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(rq / f"{a.config}.csv.gz", "wb", mtime=0) as gz:
        w = csv.writer(io.TextIOWrapper(gz, newline="", write_through=True))
        w.writerow(["shape", "c", "phase", "ttft_ms", "itl_ms", "isl", "osl", "e2e_ms"])
        w.writerows(rows)
    print(f"{a.config}: {len(pts)} points, {len(rows)} requests")


if __name__ == "__main__":
    main()
