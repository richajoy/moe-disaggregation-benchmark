#!/usr/bin/env bash
# Scrape vLLM /metrics from every pod matching a label selector every INTERVAL s, keep only vllm:* lines.
#   scrape.sh <namespace> <label-selector> <out-dir> [interval]   (runs on the node; stop with kill)
set -uo pipefail
NS=$1; SEL=$2; OUT=$3; IV=${4:-5}
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
mkdir -p "$OUT"
while :; do
  ts=$(date +%s)
  kubectl -n "$NS" get pods -l "$SEL" -o jsonpath='{range .items[*]}{.metadata.name} {.status.podIP} {.metadata.labels.llm-d\.ai/role}{"\n"}{end}' |
  while read -r name ip role; do
    port=8000; [ "$role" = decode ] && port=8200
    curl -s -m 3 "http://$ip:$port/metrics" | grep -E '^vllm:(nixl|num_requests|kv_cache_usage|prefix_cache|external_prefix|kv_offload|request_queue_time|time_to_first|inter_token|request_prefill|request_decode|num_preemptions|prompt_tokens)' \
      | sed "s/^/$ts $name /" >> "$OUT/$name.prom"
  done
  sleep "$IV"
done
