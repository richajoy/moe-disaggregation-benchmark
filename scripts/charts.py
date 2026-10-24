#!/usr/bin/env python3
"""Static charts for the docs, one hardware type per chart, from data/<node>/*.points.json.

  python3 scripts/charts.py          # writes figures/<node>/frontier-<shape>.png and latency-<shape>.png

frontier-<shape>.png  output tok/s per GPU vs tok/s per user (1000 / ITL p90), one line per configuration.
                      Filled marker = TTFT p90 <= 2 s (inside the SLO's TTFT bound); hollow = TTFT p90 > 2 s.
                      Callouts mark the TTFT-blind best point and the SLO-respecting best point at >= 35 tok/s/user.
latency-<shape>.png   TTFT p90 (log) and ITL p90 vs concurrency, with the SLO bounds drawn.
Palette: the categorical order used across this repo's figures; markers and direct labels carry identity too.
"""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.facecolor": SURF, "figure.facecolor": SURF, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "svg.fonttype": "none"})
SETS = {
    "h100": [("agg DP4", "agg-dp4"), ("agg 4x TP1", "agg-4x-tp1"), ("P/D 1P:3D TP1", "pd-1p3d-tp1"), ("P/D 2P:2D TP1", "pd-2p2d-tp1"),
             ("P/D 2P:1D(TP2) block 16", "pd-2p1d-dtp2-bs16"), ("P/D 2P:1D(TP2) block 64 (run 2)", "pd-2p1d-dtp2-bs64-run2"),
             ("P/D 2P:1D(TP2) block 64 FP8 KV", "pd-2p1d-dtp2-bs64-fp8kv")],
    "a100": [("A0 agg DP8", "a0-agg-dp8"), ("A1 agg 8x TP1", "a1-agg-8x-tp1"), ("A2 P/D 4P:2D(TP2) block 64", "a2-pd-4p2d-dtp2-bs64"),
             ("A3 P/D 2P:3D(TP2) block 64", "a3-pd-2p3d-dtp2-bs64")],
}
HW = {"h100": "4x H100 SXM", "a100": "8x A100 SXM"}
SHAPE = {"prompt-heavy": "prompt-heavy (ISL 4,000 / OSL 200)", "chat": "chat (ISL 1,000 / OSL 500)"}
FLOOR = 35


def load(node):
    return {cfg: json.load(open(ROOT / "data" / node / f"{cfg}.points.json")) for _, cfg in SETS[node]}


def drop_colliding_labels(fig, labels, keep):
    """Remove concurrency labels that would overlap a callout, the legend or an earlier label (direct labels stay legible)."""
    r = fig.canvas.get_renderer()
    placed = [k.get_window_extent(r).expanded(1.02, 1.1) for k in keep if k is not None]
    for lab in labels:
        bb = lab.get_window_extent(r)
        if any(bb.overlaps(o) for o in placed):
            lab.remove()
        else:
            placed.append(bb.expanded(1.05, 1.2))


def frontier(node, shape, data):
    fig, ax = plt.subplots(figsize=(9, 5.8))
    allp, clabels, callouts = [], [], []
    box = dict(boxstyle="round,pad=0.25", fc=SURF, ec="none", alpha=0.9)
    for i, (label, cfg) in enumerate(SETS[node]):
        ps = sorted([p for p in data[cfg] if p["shape"] == shape], key=lambda p: p["c"])
        if not ps:
            continue
        col, mk = SERIES[i % 8], MARKERS[i % 8]
        ax.plot([p["tok_s_user_p90"] for p in ps], [p["out_tok_s_gpu"] for p in ps], color=col, linewidth=1.8, alpha=0.85, label=label, zorder=2)
        for p in ps:
            ok = p["ttft_p90_ms"] <= 2000
            ax.scatter(p["tok_s_user_p90"], p["out_tok_s_gpu"], s=64, marker=mk, zorder=3, linewidths=1.8,
                       facecolors=col if ok else SURF, edgecolors=col)
            if p["c"] in (64, 1024):
                clabels.append(ax.annotate(f"c{p['c']}", (p["tok_s_user_p90"], p["out_tok_s_gpu"]), textcoords="offset points",
                                           xytext=(6, 4), fontsize=8, color=INK2, zorder=4))
            allp.append(p)
    above = [p for p in allp if p["tok_s_user_p90"] >= FLOOR]
    if above:
        blind = max(above, key=lambda p: p["out_tok_s_gpu"])
        slo = [p for p in above if p["ttft_p90_ms"] <= 2000]
        ax.axvline(FLOOR, color=INK2, linestyle=":", linewidth=1)
        ax.text(FLOOR + 1, ax.get_ylim()[1] * 0.02, f"{FLOOR} tok/s/user", color=INK2, fontsize=8, rotation=90, va="bottom")
        callouts.append(ax.annotate(f"TTFT-blind best: {blind['out_tok_s_gpu']:,.0f}\nTTFT p90 {blind['ttft_p90_ms'] / 1000:.1f} s, SLO-ok {100 * blind['slo_ok']:.1f}%",
                    (blind["tok_s_user_p90"], blind["out_tok_s_gpu"]), textcoords="offset points", xytext=(-190, -6), fontsize=8.5, color=INK,
                    bbox=box, zorder=5, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8)))
        if slo:
            s = max(slo, key=lambda p: p["out_tok_s_gpu"])
            if s is not blind:
                callouts.append(ax.annotate(f"SLO-respecting best: {s['out_tok_s_gpu']:,.0f}\nTTFT p90 {s['ttft_p90_ms'] / 1000:.1f} s, SLO-ok {100 * s['slo_ok']:.1f}%",
                            (s["tok_s_user_p90"], s["out_tok_s_gpu"]), textcoords="offset points", xytext=(24, -34), fontsize=8.5, color=INK,
                            bbox=box, zorder=5, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8)))
    ax.set_xlabel("tok/s per user = 1000 / ITL p90 (slow-tail streaming speed; excludes TTFT)")
    ax.set_ylabel("output tok/s per GPU (all GPUs of the layout)")
    ax.set_title(f"{HW[node]}, Qwen3-30B-A3B, {SHAPE[shape]}\nfilled = TTFT p90 <= 2 s, hollow = TTFT p90 > 2 s", color=INK, loc="left", fontsize=10.5)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(left=0)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.06)
    fig.tight_layout()
    drop_colliding_labels(fig, clabels, callouts + [ax.get_legend()])
    fig.savefig(ROOT / "figures" / node / f"frontier-{shape}.png", dpi=150)
    plt.close(fig)


def latency(node, shape, data):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.1), sharex=True)
    for i, (label, cfg) in enumerate(SETS[node]):
        ps = sorted([p for p in data[cfg] if p["shape"] == shape], key=lambda p: p["c"])
        if not ps:
            continue
        kw = dict(color=SERIES[i % 8], marker=MARKERS[i % 8], markersize=6, linewidth=1.8, label=label, markeredgecolor=SURF)
        axes[0].plot([p["c"] for p in ps], [p["ttft_p90_ms"] / 1000 for p in ps], **kw)
        axes[1].plot([p["c"] for p in ps], [p["itl_p90_ms"] for p in ps], **kw)
    for a in axes:
        a.set_xscale("log", base=2)
        a.set_xlabel("concurrency (requests in flight, closed loop)")
        a.set_xticks([16, 64, 256, 512, 1024])
        a.set_xticklabels(["16", "64", "256", "512", "1024"])
    axes[0].set_yscale("log")
    axes[0].axhline(2, color=INK2, linestyle="--", linewidth=1)
    axes[0].text(17, 2.15, "SLO bound 2 s", fontsize=8, color=INK2, zorder=5, bbox=dict(boxstyle="round,pad=0.15", fc=SURF, ec="none", alpha=0.85))
    axes[0].set_ylabel("TTFT p90 (s, log scale)")
    axes[0].set_title("time to first token, p90", loc="left", color=INK)
    axes[1].axhline(50, color=INK2, linestyle="--", linewidth=1)
    axes[1].text(17, 51.5, "SLO bound 50 ms (per-request mean ITL)", fontsize=8, color=INK2, zorder=5, bbox=dict(boxstyle="round,pad=0.15", fc=SURF, ec="none", alpha=0.85))
    axes[1].set_ylabel("ITL p90 (ms)")
    axes[1].set_title("inter-token latency, p90 of per-request means", loc="left", color=INK)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.suptitle(f"{HW[node]}, Qwen3-30B-A3B, {SHAPE[shape]}: latency vs load", x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(ROOT / "figures" / node / f"latency-{shape}.png", dpi=150)
    plt.close(fig)


def main():
    for node in SETS:
        (ROOT / "figures" / node).mkdir(parents=True, exist_ok=True)
        data = load(node)
        for shape in ("prompt-heavy", "chat"):
            frontier(node, shape, data)
            latency(node, shape, data)
    print("wrote", sorted(str(p.relative_to(ROOT)) for p in (ROOT / "figures").glob("*/*.png")))


if __name__ == "__main__":
    main()
