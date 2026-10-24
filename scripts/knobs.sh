#!/usr/bin/env bash
# Knob A/Bs on 1P:2D (GPUs 0-2). GPU 3 is left for the MoE tuner. Logs to $REPO_DIR/knobs.log.
set -uo pipefail
export KUBECONFIG="${KUBECONFIG:-/etc/rancher/k3s/k3s.yaml}"
REPO_DIR="${REPO_DIR:-$HOME/moe-disaggregation-benchmark}"   # working copy of this repo on the node
LLMD_DIR="${LLMD_DIR:-$HOME/llm-d}"      # checkout of the llm-d repo (guides/, recipes/)
cd "$REPO_DIR"
R="${BENCH_RESULTS:-/var/lib/bench-results}/bench"
t() { echo "[$(date -u +%H:%M:%S)] $*"; }
wait_ready() { for _ in $(seq 1 180); do
    n=$(kubectl -n llm-d-pd get pods -l llm-d.ai/guide=pd-disaggregation -o jsonpath='{range .items[*]}{.status.containerStatuses[*].ready}{"\n"}{end}' 2>/dev/null | grep -cv false)
    [ "$n" -ge "$1" ] && return 0; sleep 5; done; return 1; }
sweep() { kubectl -n llm-d exec aiperf -- env URL=http://pd-epp.llm-d-pd.svc:80 METRICS=http://pd-d0.llm-d-pd.svc:8200/metrics SHAPES="$2" CONCURRENCIES="$3" bash /tmp/sweep3.sh "$1"; }
teardown() {
  kubectl -n llm-d-pd delete deploy $(kubectl -n llm-d-pd get deploy -o name | grep -E 'pd-(p|d)[0-9]' | sed 's|deployment.apps/||') --wait=true >/dev/null 2>&1
  while kubectl -n llm-d-pd get pods 2>/dev/null | grep -qE 'pd-(p|d)[0-9]'; do sleep 3; done; }
router() { # values file (always = guide default)
  ( cd "$LLMD_DIR" && export REPO_ROOT=$PWD && source guides/env.sh && helm upgrade pd ${ROUTER_STANDALONE_CHART} -f guides/recipes/router/base.values.yaml -f "$1" -n llm-d-pd --version ${ROUTER_CHART_VERSION} >/dev/null )
  kubectl -n llm-d-pd rollout status deploy/pd-epp --timeout=180s >/dev/null; sleep 5; }
arm() { # label connector shapes concurrencies nsys(0/1) [extra args]
  local label=$1 conn=$2 shapes=$3 cs=$4 ns=$5; shift 5
  t "arm $label conn=$conn $*"
  KV_CONNECTOR=$conn python3 deploy/gen-pd.py 1 2 "$@" | kubectl apply -f - >/dev/null
  wait_ready 3 || { t "$label NOT READY"; kubectl -n llm-d-pd get pods; }
  sleep 10
  scripts/scrape.sh llm-d-pd llm-d.ai/guide=pd-disaggregation $R/metrics-$label 5 & S=$!
  if [ "$ns" = 1 ]; then ( sleep 75; sudo nsys profile --gpu-metrics-devices=0,1,2 --gpu-metrics-frequency=10000 --duration=20 -o $R/nsys-$label -f true sleep 20 >/dev/null 2>&1; t "nsys captured $label" ) & fi
  sweep "$label" "$shapes" "$cs"
  kill $S 2>/dev/null
  for p in $(kubectl -n llm-d-pd get pods -l llm-d.ai/guide=pd-disaggregation -o name); do
    kubectl -n llm-d-pd logs "$p" -c modelserver > $R/logs-$label-${p#pod/}.log 2>&1; done
}
ALWAYS=$LLMD_DIR/guides/pd-disaggregation/router/pd-disaggregation.values.yaml
SELECTIVE=$REPO_DIR/deploy/router/selective-2048.values.yaml
router $ALWAYS
arm k-pull16 NixlConnector "prompt-heavy chat" "64 256" 1; teardown
arm k-push16 NixlPushConnector "prompt-heavy" "64 256" 0; teardown
arm k-pull64 NixlConnector "prompt-heavy" "64 256" 0 --block-size=64
# selective P/D on the same pods (pull, block 64): chat (ISL 1000 < 2048 -> local prefill) and prompt-heavy
router $SELECTIVE
sweep k-pull64-selective "chat prompt-heavy" "256"
router $ALWAYS
sweep k-pull64-always-chat "chat" "256"
teardown
arm k-push64 NixlPushConnector "prompt-heavy" "64 256" 0 --block-size=64; teardown
arm k-pull16-mnbt16k NixlConnector "prompt-heavy" "64 256" 0 --max-num-batched-tokens=16384; teardown
t "KNOBS_DONE"
