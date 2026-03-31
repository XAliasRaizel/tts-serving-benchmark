# TTS Serving Benchmark — Real Inference (Chatterbox Turbo Setup)

## Overview

This project benchmarks a Text-to-Speech (TTS) serving pipeline under varying concurrency levels (1, 5, 10).

The goal is to:

* Measure **latency (TTFT — Time to First Token)**
* Measure **generation speed (RTF — Real-Time Factor)**
* Evaluate **system behavior under concurrency**
* Implement and analyze **dynamic batching as an optimization**

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

```bash
nvidia-smi
```

Example:

* GPU: RTX 3050 Ti Laptop GPU
* CUDA enabled: True

---

## Setup

```bash
cd tts-serving

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

---

## Run

### Start Server

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

### Run Benchmark

```bash
python benchmark.py
```

---

## Benchmark Results (Real Model)

### Run 1

| Concurrency | TTFT p50 | TTFT p95 | RTF p50 | RTF p95 |
| ----------- | -------- | -------- | ------- | ------- |
| 1           | 3.13s    | 3.19s    | 0.56    | 0.59    |
| 5           | 12.08s   | 14.73s   | 0.54    | 0.60    |
| 10          | 23.50s   | 28.05s   | 0.52    | 0.63    |

---

### Run 2

| Concurrency | TTFT p50 | TTFT p95 | RTF p50 | RTF p95 |
| ----------- | -------- | -------- | ------- | ------- |
| 1           | 2.75s    | 2.81s    | 0.47    | 0.49    |
| 5           | 9.84s    | 12.82s   | 0.45    | 0.64    |
| 10          | 24.37s   | 30.37s   | 0.55    | 0.67    |

---

## Observations

* At **low concurrency (c=1)**:

  * TTFT ~2.7–3.1s
  * RTF ~0.47–0.56
  * Indicates real model inference latency on GPU

---

* As **concurrency increases**:

  * TTFT increases significantly (up to ~30s p95 at c=10)
  * This indicates **queueing delay dominates latency**

---

* **RTF remains relatively stable (~0.45–0.65)**:

  * Shows that model compute time is consistent
  * GPU performance does not degrade significantly

---

## Key Insight

* The primary bottleneck is **queueing delay**, not model computation
* The system scales in a **latency-bound manner**, not compute-bound

---

## Dynamic Batching Analysis

### Approach

* Batch window: **50 ms**
* Batch size: **8**
* Requests grouped before processing

---

### Behavior

* Batching correctly groups requests at the **scheduling layer**
* However, the model processes inputs **sequentially internally**
* As a result:

  * No significant improvement in RTF
  * Limited improvement in TTFT under load

---

### Tradeoff

| Benefit                  | Limitation                            |
| ------------------------ | ------------------------------------- |
| Better request grouping  | No true GPU batching                  |
| Slight smoothing of load | Queue delay still dominant            |
| Stable system behavior   | Throughput not improved significantly |

---

## Hardware Note

* GPU used: **RTX 3050 Ti Laptop GPU**
* Observed RTF (~0.5) indicates **faster-than-real-time generation**
* This differs from expected T4 behavior (RTF > 1.0)

---

## What I’d Try Next

* Use a model that supports **true batched inference**
* Integrate **vLLM-style continuous batching**
* Explore **TensorRT optimization**
* Apply **quantization (FP16 / INT8)** to improve throughput

---

## Conclusion

The system performs reliably under real TTS inference and demonstrates correct scaling behavior under concurrent workloads.

Dynamic batching improves scheduling but does not significantly improve throughput due to lack of model-level batching support.

The results highlight that:

> **Efficient GPU utilization requires batching at the model inference level, not just at the request scheduling layer.**

---
