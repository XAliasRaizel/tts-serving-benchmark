# TTS Serving Benchmark — Chatterbox Turbo

## Overview

This project benchmarks a Text-to-Speech (TTS) serving pipeline using **Chatterbox Turbo** under varying concurrency levels (1, 5, 10).

The focus is on:

* **Latency (TTFT — Time to First Token)**
* **Generation speed (RTF — Real-Time Factor)**
* **Scalability under concurrent workloads**

Additionally, I identified and implemented the **highest-impact optimization (dynamic batching)** to improve tail latency and system throughput.

---

## Metrics

### TTFT (Time To First Token)

Time from request arrival to the first audio chunk.
This directly reflects **user-perceived latency**.

---

### RTF (Real-Time Factor)

RTF = generation_time / audio_duration

* RTF < 1 → faster than real-time
* RTF = 1 → real-time
* RTF > 1 → slower than real-time

---

## GPU Verification

### System Check

```bash
nvidia-smi
```

### Example Output

```
+------------------------------------------------------------------------------+
| NVIDIA-SMI 535.xx       Driver Version: 535.xx       CUDA Version: 12.x      |
| GPU  Name        Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
|  0   Tesla T4              On  | 00000000:00:04.0 Off |                    0 |
+------------------------------------------------------------------------------+
```

### PyTorch Check

```python
import torch
print("GPU Available:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0))
```

---

## Setup

```bash
git clone <your-repo-link>
cd tts-serving

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

Tested on Python 3.10+

---

## Running the Server

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

---

## Running Benchmark

```bash
python3 benchmark.py
```

---

## Results

| Concurrency | TTFT p50 | TTFT p95 | RTF p50 | RTF p95 |
| ----------- | -------- | -------- | ------- | ------- |
| 1           | 0.076s   | 0.076s   | 0.09    | 0.10    |
| 5           | 0.078s   | 0.079s   | 0.16    | 0.23    |
| 10          | 0.055s   | 0.125s   | 0.23    | 0.39    |

---

## Key Improvement

* TTFT p95 reduced from **~1.39s → ~0.12s**
* RTF p95 reduced from **~1.89 → ~0.39**

This demonstrates a **significant reduction in tail latency** under high concurrency.

---

## Observations

* At low concurrency, the system achieves low latency (~76ms TTFT) and fast generation (RTF < 0.1).
* Under higher concurrency, naive request handling leads to **queueing delays and GPU underutilization**.
* Tail latency (p95) is the key bottleneck in real-time systems.
* After optimization, latency remains stable even at concurrency 10, indicating improved scalability.

---

## Optimization: Dynamic Batching

### Approach

* Batch window: **20 ms**
* Batch size: **8**
* Multiple requests processed in a single GPU forward pass

---

### Why it works

* Reduces GPU idle time
* Improves throughput
* Minimizes queueing delays

---

### Tradeoff

* Slight increase in TTFT due to batching window
* Significant reduction in p95 latency

---

## What I’d Try Next

* KV cache optimization for streaming TTS
* FP8 / INT8 quantization to increase batch size under VRAM constraints
* TensorRT-LLM for kernel-level optimization

---

## Conclusion

The system performs efficiently under low load but requires batching to scale under concurrent workloads.

Dynamic batching proved to be the **highest-impact optimization**, significantly reducing tail latency and improving real-time performance, making the system robust for production-style usage.

---
