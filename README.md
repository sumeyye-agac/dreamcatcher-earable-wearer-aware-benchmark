# DreamCatcher Earable Wearer-Aware Benchmark

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Wav2Vec2](https://img.shields.io/badge/Wav2Vec2-Facebook-blue.svg)](https://huggingface.co/facebook/wav2vec2-base)

Resource-efficient wearer-aware event recognition on earables (DreamCatcher dataset).

This repository provides a reproducible benchmarking framework for earable-based
event recognition under real-world sleep conditions. The focus is on building
lightweight and resource-aware models using attention mechanisms and knowledge
distillation, while maintaining strong recognition performance.

The current implementation is audio-only, based on the publicly available
DreamCatcher release, and is designed to be easily extendable to multimodal
audio + IMU settings when additional data access is available.

---

## Problem

Wearable sleep monitoring using earables introduces unique challenges:

- Events such as snoring, breathing irregularities, coughing, or body movements
  are captured by in-ear microphones, often mixed with sounds from a co-sleeper.
- The system must distinguish wearer-generated events from external events
  (wearer-aware recognition).
- Models are expected to operate under strict resource constraints, as earables
  require low-latency and low-power inference.

This repository studies these challenges from a resource-aware applied machine
learning perspective.

---

## Installation

### Prerequisites

- Python 3.9+ (tested with Python 3.12)
- ~50GB free disk space (for dataset download and cache)
- HuggingFace account with access to the DreamCatcher dataset

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd dreamcatcher-earable-wearer-aware-benchmark
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv-earable
   source .venv-earable/bin/activate  # On Windows: .venv-earable\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Authenticate with HuggingFace:**
   The DreamCatcher dataset is gated and requires authentication.
   
   - Request access on the [HuggingFace dataset page](https://huggingface.co/datasets/THU-PI-Sensing/DreamCatcher)
   - After approval, authenticate:
     ```bash
     hf auth login
     ```
   - Enter your HuggingFace token when prompted

### Troubleshooting

**Dependency conflicts:** If you encounter issues with `datasets`, `pyarrow`, or `numpy`, ensure you're using the pinned versions in `requirements.txt`. These versions are tested and compatible.

**Disk space:** The dataset cache can grow large (~70-80GB). If you encounter "No space left on device" errors:

1. Check available disk space:
   ```bash
   df -h ~
   ```

2. Check HuggingFace cache size:
   ```bash
   du -sh ~/.cache/huggingface/datasets
   ```

3. Clear the HuggingFace cache if needed:
   ```bash
   rm -rf ~/.cache/huggingface/datasets/*
   ```
   
   **Warning:** This will delete all cached datasets. You'll need to re-download them on next use.

4. If disk space is still limited, consider using streaming mode (modify `src/data/dreamcatcher_hf.py` to set `streaming=True`). Note that streaming may be slower for training.

**Dataset loading:** The dataset uses the `sleep_event_classification` config. This is automatically handled by the code.

---

## Dataset

DreamCatcher dataset (introduced in NeurIPS 2024, Datasets & Benchmarks Track)

- Modality (public release): in-ear audio
- Setting: overnight sleep recordings with co-sleepers
- Key property: wearer-aware event annotations
- Config: `sleep_event_classification` (automatically used by the code)
- Source: [THU-PI-Sensing/DreamCatcher](https://huggingface.co/datasets/THU-PI-Sensing/DreamCatcher) on HuggingFace

Note: The public version used in this repository contains audio data only.
Multimodal pipelines are implemented in a modular way but require explicit dataset
access approval.

---

## Scope

**Relation to the original DreamCatcher work.**  
The original DreamCatcher study introduces the dataset and formulates the
wearer-aware sleep event recognition problem.  
This repository builds on that foundation by focusing on **systematic model
benchmarking under wearable constraints**, including attention mechanisms,
knowledge distillation, and explicit evaluation of accuracy–efficiency trade-offs
(parameter count and CPU latency).

This repository focuses on:

- Segment-level audio event classification on earables (event_label)
- Wearer-aware classification under real-world interference
- Lightweight model design (TinyCNN, CRNN)
- Attention mechanisms (CBAM)
- Knowledge distillation (teacher–student)
- Resource-aware evaluation (parameter count and CPU latency)

The repository does not target speech recognition or ASR.

---

## Models

**Student models (edge-oriented):**
- TinyCNN
- CRNN
- CRNN + CBAM

**Teacher model (training only):**
- Pretrained Wav2Vec2 encoder (frozen) with a lightweight classification head

*(The Wav2Vec2 model is used as a frozen teacher during training only.)*

---

## Reproducibility

All experiments are scriptable and log results to `results/leaderboard.csv`.

**Baselines and attention models:**
```bash
bash experiments/run_audio_benchmark.sh
```
**Quick smoke run (fast end-to-end sanity check):**
```bash
bash experiments/run_audio_smoke.sh
```
**Knowledge distillation:**
```bash
bash experiments/run_kd.sh
```
**KD smoke run:**
```bash
bash experiments/run_kd_smoke.sh
```
**Reporting and plots:**
```bash
bash experiments/make_plots.sh
```
---

## Key Takeaways

- Lightweight CRNN models benefit significantly from attention mechanisms in earable audio event recognition.
- Knowledge distillation narrows the performance gap between compact students and stronger pretrained teachers.
- Benchmarking accuracy together with latency and parameter count is critical for realistic wearable deployment scenarios.

---