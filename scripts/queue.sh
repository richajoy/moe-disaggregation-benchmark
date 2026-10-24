#!/usr/bin/env bash
# Unattended queue: each step deploys a config, waits until ready, sweeps, tears down.
#   scripts/queue.sh agg-dp4 agg-4x-tp1 pd-1p3d-tp1 pd-2p2d-tp1 pd-2p1d-dtp2-bs16 pd-2p1d-dtp2-bs64 Logs to $REPO_DIR/queue.log.
set -uo pipefail
export KUBECONFIG="${KUBECONFIG:-/etc/rancher/k3s/k3s.yaml}"
REPO_DIR="${REPO_DIR:-$HOME/moe-disaggregation-benchmark}"   # working copy of this repo on the node
cd "$REPO_DIR"
R="${BENCH_RESULTS:-/var/lib/bench-results}/bench"
t() { echo "[$(date -u +%H:%M:%S)] $*"; }
wait_ready() { # ns selector count
  for _ in $(seq 1 180); do
    n=$(kubectl -n "$1" get pods -l "$2" -o jsonpath='{range .items[*]}{.status.containerStatuses[*].ready}{"\n"}{end}' 2>/dev/null | grep -cv false)
    [ "$n" -ge "$3" ] && return 0; sleep 5; done; return 1; }
sweep() { # label url shapes concurrencies
  kubectl -n llm-d exec aiperf -- env URL="$2" METRICS="${METRICS:-$2/metrics}" SHAPES="$3" CONCURRENCIES="$4" bash /tmp/sweep3.sh "$1"; }
run_pd() { # np nd label shapes concurrencies [extra vllm args]
  local np=$1 nd=$2 label=$3 shapes=$4 cs=$5; shift 5
  t "deploy $label ${np}P:${nd}D $*"
  python3 deploy/gen-pd.py "$np" "$nd" "$@" | kubectl apply -f - >/dev/null
  wait_ready llm-d-pd llm-d.ai/guide=pd-disaggregation $((np+nd)) || { t "$label NOT READY"; kubectl -n llm-d-pd get pods; }
  sleep 10
  scripts/scrape.sh llm-d-pd llm-d.ai/guide=pd-disaggregation $R/metrics-$label 5 & S=$!
  METRICS=http://pd-d0.llm-d-pd.svc:8200/metrics sweep "$label" http://pd-epp.llm-d-pd.svc:80 "$shapes" "$cs"
  kill $S
  for p in $(kubectl -n llm-d-pd get pods -l llm-d.ai/guide=pd-disaggregation -o name); do
    kubectl -n llm-d-pd logs "$p" -c modelserver > $R/logs-$label-${p#pod/}.log 2>&1; done
  kubectl -n llm-d-pd delete deploy -l llm-d.ai/guide=pd-disaggregation --wait=true >/dev/null 2>&1
  kubectl -n llm-d-pd delete deploy $(kubectl -n llm-d-pd get deploy -o name | grep -E 'pd-(p|d)[0-9]' | sed 's|deployment.apps/||') --wait=true >/dev/null 2>&1
  while kubectl -n llm-d-pd get pods 2>/dev/null | grep -qE 'pd-(p|d)[0-9]'; do sleep 3; done
  t "done $label"
}
mkdir -p $R 2>/dev/null
VLLM_ARGS="--max-model-len 8192 --gpu-memory-utilization 0.90 --port 8000 --disable-access-log-for-endpoints=/health,/metrics,/v1/models"
run_agg() { # label ngpu replicas extra-args
  local label=$1 ngpu=$2 reps=$3; shift 3
  t "deploy $label"
  deploy/render.sh "$label" "$ngpu" "$VLLM_ARGS $*" | sed "s/  replicas: 1/  replicas: $reps/" | kubectl apply -f - >/dev/null
  wait_ready llm-d app="$label" "$reps" || t "$label NOT READY"
  sleep 10
  sweep "$label" "http://$label.llm-d.svc:8000" "prompt-heavy chat" "16 64 256 512 1024"
  for p in $(kubectl -n llm-d get pods -l app="$label" -o name); do kubectl -n llm-d logs "$p" > $R/logs-$label-${p#pod/}.log 2>&1; done
  kubectl -n llm-d delete deploy "$label" svc "$label" --wait=true >/dev/null 2>&1
  while kubectl -n llm-d get pods | grep -q "$label"; do sleep 3; done; t "done $label"; }
for step in "$@"; do
  case $step in
    agg-dp4) run_agg agg-dp4 4 1 --data-parallel-size 4 ;;
    agg-4x-tp1) run_agg agg-4x-tp1 1 4 ;;
    pd-1p3d-tp1) run_pd 1 3 pd-1p3d-tp1 "prompt-heavy chat" "16 64 256 512 1024" ;;
    pd-2p2d-tp1) run_pd 2 2 pd-2p2d-tp1 "prompt-heavy chat" "16 64 256 512 1024" ;;
    pd-2p1d-dtp2-bs16) DECODE_TP=2 run_pd 2 1 pd-2p1d-dtp2-bs16 "prompt-heavy chat" "16 64 256 512 1024" ;;
    pd-2p1d-dtp2-bs64) DECODE_TP=2 run_pd 2 1 pd-2p1d-dtp2-bs64 "prompt-heavy chat" "16 64 256 512 1024" --block-size=64 ;;
    pd-2p1d-dtp2-bs64-fp8kv) DECODE_TP=2 run_pd 2 1 pd-2p1d-dtp2-bs64-fp8kv "prompt-heavy chat" "256 512 1024" --block-size=64 --kv-cache-dtype=fp8 ;;
    *) t "unknown step $step" ;;
  esac
done
t "QUEUE_DONE"
