#!/usr/bin/env bash
# KV-capacity arms (approved): BF16 vs FP8 KV vs CPU offload vs both on agg 4x TP1, plus P/D + NIXL/Offloading
# MultiConnector. Workload: prefix shape, G=128 x 2000-token shared prefixes (256k tokens > 124k per TP1 pod),
# one warm-up pass at c256, then measured c256 and c1024.
set -uo pipefail
export KUBECONFIG="${KUBECONFIG:-/etc/rancher/k3s/k3s.yaml}"
REPO_DIR="${REPO_DIR:-$HOME/moe-disaggregation-benchmark}"   # working copy of this repo on the node
SKIP="${SKIP:-}"   # space-separated arm labels already done
cd "$REPO_DIR"; R="${BENCH_RESULTS:-/var/lib/bench-results}/bench"; OFF=107374182400   # 100 GiB CPU tier per pod
t() { echo "[$(date -u +%H:%M:%S)] $*"; }
kubectl -n llm-d cp scripts/sweep3.sh aiperf:/tmp/sweep3.sh
for d in $(kubectl -n llm-d-pd get deploy -o name | grep -E 'pd-(p|d)[0-9]'); do kubectl -n llm-d-pd delete $d --wait=true >/dev/null 2>&1; done
while kubectl -n llm-d-pd get pods | grep -qE 'pd-(p|d)[0-9]'; do sleep 3; done
sweeps() { # label url metrics
  kubectl -n llm-d exec aiperf -- env URL="$2" METRICS="$3" SHAPES=prefix CONCURRENCIES="256" bash /tmp/sweep3.sh "$1-warm" >/dev/null 2>&1
  kubectl -n llm-d exec aiperf -- env URL="$2" METRICS="$3" SHAPES=prefix CONCURRENCIES="256 1024" bash /tmp/sweep3.sh "$1"; }
agg() { # label vllm-args...
  local label=$1; shift; case " $SKIP " in *" $label "*) t "skip $label (done)"; return;; esac; t "agg arm $label $*"
  deploy/render.sh agg4 1 "--max-model-len 8192 --gpu-memory-utilization 0.90 --port 8000 --disable-access-log-for-endpoints=/health,/metrics,/v1/models $*" | sed 's/  replicas: 1/  replicas: 4/' | kubectl apply -f - >/dev/null
  for _ in $(seq 1 180); do n=$(kubectl -n llm-d get pods -l app=agg4 -o jsonpath='{range .items[*]}{.status.containerStatuses[*].ready}{"\n"}{end}' | grep -c true); [ "$n" -ge 4 ] && break; sleep 5; done
  sleep 10
  scripts/scrape.sh llm-d app=agg4 $R/metrics-$label 5 & S=$!
  sweeps "$label" http://agg4.llm-d.svc:8000 http://agg4.llm-d.svc:8000/metrics
  kill $S
  for p in $(kubectl -n llm-d get pods -l app=agg4 -o name); do kubectl -n llm-d logs "$p" > $R/logs-$label-${p#pod/}.log 2>&1; done
  kubectl -n llm-d delete deploy agg4 svc agg4 --wait=true >/dev/null 2>&1; while kubectl -n llm-d get pods | grep -q agg4; do sleep 3; done
  t "done $label"; }
agg o0-bf16
agg o1-bf16-offload "--kv-transfer-config {\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":$OFF}}"
agg q1-fp8kv "--kv-cache-dtype fp8"
t "P/D arm o2-pd-offload"
OFFLOAD_BYTES=$OFF DECODE_TP=2 python3 deploy/gen-pd.py 2 1 --block-size=64 | kubectl apply -f - >/dev/null
for _ in $(seq 1 180); do n=$(kubectl -n llm-d-pd get pods -l llm-d.ai/guide=pd-disaggregation -o jsonpath='{range .items[*]}{.status.containerStatuses[*].ready}{"\n"}{end}' | grep -cv false); [ "$n" -ge 3 ] && break; sleep 5; done
kubectl -n llm-d-pd get pods > $R/o2-pods.txt 2>&1
sleep 10
scripts/scrape.sh llm-d-pd llm-d.ai/guide=pd-disaggregation $R/metrics-o2-pd-offload 5 & S=$!
sweeps o2-pd-offload http://pd-epp.llm-d-pd.svc:80 http://pd-d0.llm-d-pd.svc:8200/metrics
kill $S
for p in $(kubectl -n llm-d-pd get pods -l llm-d.ai/guide=pd-disaggregation -o name); do kubectl -n llm-d-pd logs "$p" -c modelserver > $R/logs-o2-pd-offload-${p#pod/}.log 2>&1; done
for d in $(kubectl -n llm-d-pd get deploy -o name | grep -E 'pd-(p|d)[0-9]'); do kubectl -n llm-d-pd delete $d --wait=true >/dev/null 2>&1; done
t "done o2-pd-offload"
agg q1o1-fp8-offload "--kv-cache-dtype fp8 --kv-transfer-config {\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":$OFF}}"
t "OFFLOAD_DONE"
