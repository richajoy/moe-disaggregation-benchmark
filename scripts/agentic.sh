#!/usr/bin/env bash
# Agentic load (LAST test): multi-turn sessions with growing context on the chat endpoint.
# 6 +- 1 turns per session, 600 new tokens per turn, 150-token outputs, 500 ms tool gap between turns.
# Configs: best agg (vLLM DP4) and best P/D (2P TP1 : 1D TP2, block 64, FP8 KV). Concurrency 256 and 1024.
set -uo pipefail
export KUBECONFIG="${KUBECONFIG:-/etc/rancher/k3s/k3s.yaml}"
REPO_DIR="${REPO_DIR:-$HOME/moe-disaggregation-benchmark}"   # working copy of this repo on the node
cd "$REPO_DIR"; R="${BENCH_RESULTS:-/var/lib/bench-results}/bench"
t() { echo "[$(date -u +%H:%M:%S)] $*"; }
run() { # label url metrics
  for c in 256 1024; do
    out=/results/bench/$(date +%Y%m%d-%H%M)-$1/agentic/c$c
    t "agentic $1 c=$c"
    kubectl -n llm-d exec aiperf -- bash -c "mkdir -p $out; curl -s $3 > /tmp/m.before; aiperf profile --model Qwen/Qwen3-30B-A3B --url $2 --endpoint-type chat --streaming \
      --concurrency $c --num-sessions $((c*2)) --session-turns-mean 6 --session-turns-stddev 1 --session-turn-delay-mean 500 \
      --synthetic-input-tokens-mean 600 --synthetic-input-tokens-stddev 0 --output-tokens-mean 150 --output-tokens-stddev 0 \
      --extra-inputs max_tokens:150 --extra-inputs min_tokens:150 --extra-inputs ignore_eos:true --random-seed 42 \
      --request-timeout-seconds 900 --server-metrics $3 --artifact-dir $out --ui simple > $out.log 2>&1; echo rc=\$? >> $out.log; curl -s $3 > $out.metrics.after; cp /tmp/m.before $out.metrics.before"
    kubectl -n llm-d exec aiperf -- bash -c "grep -E 'Time to First Token|Inter Token Latency|Request Throughput|Output Token Throughput \(|rc=' $out.log | head -6"
  done; }
# 2) best P/D: 2P TP1 : 1D TP2, block 64, FP8 KV (guide router, always-disaggregate)
t "deploy agentic P/D 2P:1D(TP2) bs64 fp8kv"
DECODE_TP=2 python3 deploy/gen-pd.py 2 1 --block-size=64 --kv-cache-dtype=fp8 | kubectl apply -f - >/dev/null   # no-op if already running
for _ in $(seq 1 180); do n=$(kubectl -n llm-d-pd get pods -l llm-d.ai/guide=pd-disaggregation -o jsonpath='{range .items[*]}{.status.containerStatuses[*].ready}{"\n"}{end}' | grep -cv false); [ "$n" -ge 3 ] && break; sleep 5; done
sleep 10
scripts/scrape.sh llm-d-pd llm-d.ai/guide=pd-disaggregation $R/metrics-agentic-pd-fp8 5 & S=$!
run agentic-pd-fp8 http://pd-epp.llm-d-pd.svc:80 http://pd-d0.llm-d-pd.svc:8200/metrics
kill $S
for p in $(kubectl -n llm-d-pd get pods -l llm-d.ai/guide=pd-disaggregation -o name); do kubectl -n llm-d-pd logs "$p" -c modelserver > $R/logs-agentic-pd-fp8-${p#pod/}.log 2>&1; done
for d in $(kubectl -n llm-d-pd get deploy -o name | grep -E 'pd-(p|d)[0-9]'); do kubectl -n llm-d-pd delete $d --wait=true >/dev/null 2>&1; done
while kubectl -n llm-d-pd get pods | grep -qE 'pd-(p|d)[0-9]'; do sleep 3; done
# 1) best agg: vLLM --data-parallel-size 4
t "deploy agentic agg DP4"
deploy/render.sh ag-dp4 4 "--data-parallel-size 4 --max-model-len 8192 --gpu-memory-utilization 0.90 --port 8000 --disable-access-log-for-endpoints=/health,/metrics,/v1/models" | kubectl apply -f - >/dev/null
for _ in $(seq 1 180); do kubectl -n llm-d get pods -l app=ag-dp4 -o jsonpath='{.items[0].status.containerStatuses[0].ready}' 2>/dev/null | grep -q true && break; sleep 5; done
sleep 10
scripts/scrape.sh llm-d app=ag-dp4 $R/metrics-agentic-agg-dp4 5 & S=$!
run agentic-agg-dp4 http://ag-dp4.llm-d.svc:8000 http://ag-dp4.llm-d.svc:8000/metrics
kill $S
kubectl -n llm-d logs deploy/ag-dp4 > $R/logs-agentic-agg-dp4.log 2>&1
kubectl -n llm-d delete deploy ag-dp4 svc ag-dp4 --wait=true >/dev/null 2>&1; while kubectl -n llm-d get pods | grep -q ag-dp4; do sleep 3; done
t "AGENTIC_DONE"
