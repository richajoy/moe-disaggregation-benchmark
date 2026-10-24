#!/usr/bin/env python3
"""Render an nP:mD llm-d P/D topology for ONE node, KV over NVLink (cuda_ipc).
  gen-pd.py <nprefill> <ndecode> [extra vLLM args...] | kubectl apply -f -
Each pod sees all GPUs (NVIDIA_VISIBLE_DEVICES=all, no nvidia.com/gpu request) and is pinned with
CUDA_VISIBLE_DEVICES: CUDA IPC needs the reading process to see the peer GPU; with one GPU per pod UCX
silently falls back to TCP. Prefill takes GPUs 0..n-1, decode the next m.
Labels follow guides/pd-disaggregation so the llm-d router (EPP) selects pods by llm-d.ai/role.
"""
import json, os, sys
NP, ND = int(sys.argv[1]), int(sys.argv[2]); EXTRA = sys.argv[3:]
DTP = int(os.environ.get("DECODE_TP", "1"))  # decode tensor parallel size (prefill stays TP1)
NS, MODEL = "llm-d-pd", "Qwen/Qwen3-30B-A3B"
SIDECAR = "ghcr.io/llm-d/llm-d-router-disagg-sidecar:v1.0.0"
IMAGE = "docker.io/vllm/vllm-openai:v0.32.0"
CONNECTOR = os.environ.get("KV_CONNECTOR", "NixlConnector")  # NixlConnector (= NixlPullConnector) or NixlPushConnector
OFFLOAD = os.environ.get("OFFLOAD_BYTES")  # if set: MultiConnector(NIXL, OffloadingConnector to CPU)
def kvcfg(role):
    nixl = {"kv_connector": CONNECTOR, "kv_role": role, "kv_buffer_device": "cuda",
            "kv_connector_extra_config": {"backends": ["UCX"]}}
    if not OFFLOAD: return json.dumps(nixl)
    off = {"kv_connector": "OffloadingConnector", "kv_role": "kv_both",
           "kv_connector_extra_config": {"cpu_bytes_to_use": int(OFFLOAD)}}
    return json.dumps({"kv_connector": "MultiConnector", "kv_role": "kv_both",
                       "kv_connector_extra_config": {"connectors": [nixl, off]}})
def pod(name, role, gpu, port, sidecar, tp=1):
    args = [MODEL, "--disable-access-log-for-endpoints=/health,/metrics,/v1/models", f"--tensor-parallel-size={tp}",
            "--max-model-len=8192", "--gpu-memory-utilization=0.90", f"--port={port}",
            "--kv-transfer-config", kvcfg("kv_producer" if role == "prefill" else "kv_consumer"),
            "--no-disable-hybrid-kv-cache-manager", "--shutdown-timeout=45"] + EXTRA
    env = [{"name": "HF_HOME", "value": "/hf"}, {"name": "HF_HUB_OFFLINE", "value": "1"},
           {"name": "NVIDIA_VISIBLE_DEVICES", "value": "all"}, {"name": "CUDA_VISIBLE_DEVICES", "value": gpu},
           {"name": "VLLM_NIXL_SIDE_CHANNEL_HOST", "valueFrom": {"fieldRef": {"fieldPath": "status.podIP"}}},
           {"name": "VLLM_HTTP_TIMEOUT_KEEP_ALIVE", "value": "120"},
           {"name": "UCX_PROTO_INFO", "value": "y"}]
    c = [{"name": "modelserver", "image": IMAGE, "command": ["vllm", "serve"], "args": args, "env": env,
          "ports": [{"name": "modelserver", "containerPort": port}, {"name": "nixl", "containerPort": 5600}],
          "readinessProbe": {"httpGet": {"path": "/v1/models", "port": port}, "periodSeconds": 5, "failureThreshold": 600},
          "volumeMounts": [{"name": "hf", "mountPath": "/hf"}, {"name": "shm", "mountPath": "/dev/shm"}]}]
    init = []
    if sidecar:
        init = [{"name": "routing-proxy", "image": SIDECAR, "imagePullPolicy": "IfNotPresent", "restartPolicy": "Always",
                 "args": ["--port=8000", "--kv-connector=nixlv2", "--zap-log-level=1", "--secure-serving=false"],
                 "ports": [{"name": "sidecar", "containerPort": 8000}]}]
    labels = {"app": name, "llm-d.ai/inference-serving": "true", "llm-d.ai/guide": "pd-disaggregation",
              "llm-d.ai/role": role, "llm-d.ai/engine-type": "vllm"}
    return {"apiVersion": "apps/v1", "kind": "Deployment", "metadata": {"name": name, "namespace": NS},
            "spec": {"replicas": 1, "strategy": {"type": "Recreate"}, "selector": {"matchLabels": {"app": name}},
                     "template": {"metadata": {"labels": labels},
                                  "spec": {"runtimeClassName": "nvidia", "hostIPC": True, "hostPID": True,
                                           "initContainers": init, "containers": c,
                                           "volumes": [{"name": "hf", "hostPath": {"path": os.environ.get("HF_CACHE", "/var/lib/hf-cache"), "type": "Directory"}},
                                                       {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "120Gi"}}]}}}}
docs = [pod(f"pd-p{i}", "prefill", str(i), 8000, False) for i in range(NP)]
docs += [pod(f"pd-d{j}", "decode", ",".join(str(NP + j * DTP + k) for k in range(DTP)), 8200, True, DTP) for j in range(ND)]
print("\n---\n".join(json.dumps(d) for d in docs))
