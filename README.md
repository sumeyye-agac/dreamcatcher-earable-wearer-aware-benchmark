# DreamCatcher Earable Wearer-Aware Benchmark

Resource-efficient wearer-aware event recognition on earables (DreamCatcher dataset).

This repository provides a **reproducible benchmarking framework** for earable-based
event recognition under real-world sleep conditions. The focus is on building
**lightweight and resource-aware models** using attention mechanisms and
knowledge distillation, while maintaining strong recognition performance.

The current implementation is **audio-only**, based on the publicly available
DreamCatcher release, and is designed to be easily extendable to multimodal
audio + IMU settings when additional data access is available.

---

## Problem

Wearable sleep monitoring using earables introduces unique challenges:

- Events such as snoring, breathing irregularities, coughing, or body movements
  are captured by **in-ear microphones**, often mixed with sounds from a co-sleeper.
- The system must distinguish **wearer-generated events** from external events
  (wearer-aware recognition).
- Models are expected to operate under **strict resource constraints**, as
  earables require low-latency and low-power inference.

This repository studies these challenges from a **resource-aware applied
machine learning perspective**.

---

## Dataset

**DreamCatcher** (NeurIPS 2024, Datasets & Benchmarks Track)

- Modality (public release): **In-ear audio**
- Setting: Overnight sleep recordings with co-sleepers
- Key property: **Wearer-aware event annotations**
- Events include body movement, vocal and breathing-related sleep events

> Note: The public version used in this repository contains audio data only.
> Multimodal pipelines are implemented in a modular way but require explicit
> dataset access approval from the dataset authors.

Dataset access instructions are provided in `data/README.md`.

---

## Scope of This Repository

This repository focuses on:

- Audio-based event recognition on earables
- Wearer-aware classification under real-world interference
- Lightweight and deployable model design
- Attention mechanisms and knowledge distillation for efficiency

The repository does **not** target speech recognition or ASR.
Audio is treated as a biomedical and environmental sensing modality.

---

## Models

A compact but expressive model family is considered:

### Student Models (Edge-oriented)
- **TinyCNN**: Fully convolutional, minimal footprint
- **CRNN**: CNN-based feature extractor with lightweight temporal modeling

### Teacher Model
- **Pretrained audio embeddings (YAMNet) + MLP**
- Used only during training for knowledge distillation

---

## Methods

The following learning strategies are explored:

- Baseline training
- Attention mechanisms  
  - CBAM adapted to time–frequency audio representations
- Knowledge distillation  
  - Response-based distillation from a stronger teacher
- Combined attention and distillation for improved efficiency

The goal is to analyze **performance versus resource consumption trade-offs**
in a controlled and reproducible manner.

---

## Reproducibility

- Fixed random seeds
- Config-driven experiments
- Deterministic preprocessing
- Script-based execution

Example:
```bash
bash experiments/run_stage1_audio.sh
