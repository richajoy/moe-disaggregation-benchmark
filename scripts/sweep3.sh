#!/usr/bin/env bash
# Concurrency sweep. Runs INSIDE the aiperf pod.
#   URL=http://h0-dp4.llm-d.svc:8000 CONCURRENCIES="16 64 256 512 1024" SHAPES="prompt-heavy chat" bash sweep3.sh <label>
# /v1/completions + ignore_eos + min_tokens -> exact lengths. Effective concurrency is checked afterwards
# from the AIPerf export (requests in flight), points below 0.9 x requested are marked invalid by the analysis.
set -euo pipefail
LABEL="${1:?usage: sweep3.sh <label>}"
URL="${URL:?set URL}"
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B}"
CONCURRENCIES="${CONCURRENCIES:-16 64 256 512 1024}"
SHAPES="${SHAPES:-prompt-heavy chat}"
METRICS="${METRICS:-$URL/metrics}"
DCGM_URL="${DCGM_URL:-http://nvidia-dcgm-exporter.gpu-operator.svc:9400/metrics}"
OUT="/results/bench/$(date +%Y%m%d-%H%M)-$LABEL"
declare -A ISL=( [prompt-heavy]=4000 [chat]=1000 [long]=7000 [prefix]=300 )
declare -A OSL=( [prompt-heavy]=200  [chat]=500  [long]=500 [prefix]=200 )
# prefix shape: PREFIX_G shared 2000-token prefixes + 300 unique tokens (working set G x 2000 tokens)
PREFIX_G="${PREFIX_G:-128}"
curl -sf "$URL/v1/models" >/dev/null || { echo "server not reachable at $URL"; exit 1; }
mkdir -p "$OUT"; echo "results -> $OUT"
for shape in $SHAPES; do
  for c in $CONCURRENCIES; do
    n=$(( c * 3 > 192 ? c * 3 : 192 ))
    echo "== $(date +%H:%M:%S) $shape ISL=${ISL[$shape]} OSL=${OSL[$shape]} c=$c n=$n"
    curl -s "$METRICS" > "$OUT/$shape-c$c.metrics.before" || true
    PFX=(); [ "$shape" = prefix ] && PFX=(--num-prefix-prompts "$PREFIX_G" --prefix-prompt-length 2000 --random-seed 42)
    aiperf profile --model "$MODEL" --url "$URL" "${PFX[@]}" \
      --endpoint-type completions --streaming \
      --concurrency "$c" --request-count "$n" --warmup-request-count $(( c < 64 ? c : 64 )) \
      --synthetic-input-tokens-mean "${ISL[$shape]}" --synthetic-input-tokens-stddev 0 \
      --output-tokens-mean "${OSL[$shape]}" --output-tokens-stddev 0 \
      --extra-inputs "max_tokens:${OSL[$shape]}" --extra-inputs "min_tokens:${OSL[$shape]}" --extra-inputs ignore_eos:true \
      --gpu-telemetry "$DCGM_URL" --server-metrics "$METRICS" \
      --artifact-dir "$OUT/$shape/c$c" --ui simple > "$OUT/$shape-c$c.log" 2>&1 || echo "aiperf rc=$? (see log)"
    curl -s "$METRICS" > "$OUT/$shape-c$c.metrics.after" || true
    grep -E 'Time to First Token|Inter Token Latency|Output Token Throughput|Request Throughput' "$OUT/$shape-c$c.log" | head -6 || true
  done
done
echo "done: $OUT"
