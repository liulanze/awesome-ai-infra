# Microsoft Foundry & Azure AI Infrastructure

> **Source:** [Microsoft at NVIDIA GTC — New Solutions for Microsoft Foundry, Azure AI Infrastructure and Physical AI](https://blogs.microsoft.com/blog/2026/03/16/microsoft-at-nvidia-gtc-new-solutions-for-microsoft-foundry-azure-ai-infrastructure-and-physical-ai/)

## Overview

Foundry builds on Azure to bring together models, tools, data and observability into a single system designed for production agents.

New Azure AI infrastructure optimized for inference-heavy, reasoning-based workloads, including the first hyperscale cloud to power on next-generation NVIDIA Vera Rubin NVL72 systems. Vera Rubin NVL72 will be rolled out into modern, liquid-cooled Azure datacenters.

---

## NVIDIA Vera Rubin NVL72 Performance Data

> All values are preliminary and subject to change per NVIDIA. Source: [nvidia.com/en-us/data-center/vera-rubin-nvl72](https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/)

### Per-GPU Specs (Rubin GPU)

| Spec | Value |
|---|---|
| NVFP4 Inference | 50 PFLOPS |
| NVFP4 Training (dense) | 35 PFLOPS |
| FP8/FP6 Training (dense) | 17.5 PFLOPS |
| FP16/BF16 (dense) | 4 PFLOPS |
| TF32 (dense) | 2 PFLOPS |
| FP32 | 130 TFLOPS |
| FP64 | 33 TFLOPS |
| HBM4 Memory | 288 GB |
| Memory Bandwidth | 22 TB/s |
| NVLink 6 Bandwidth | 3.6 TB/s |
| Process Node | TSMC 3nm-class (reported) |

### NVL72 Rack (Full System)

| Spec | Value |
|---|---|
| GPUs | 72 Rubin GPUs |
| CPUs | 36 Vera CPUs |
| NVFP4 Inference (total) | 3,600 PFLOPS (3.6 ExaFLOPS) |
| NVFP4 Training (dense, total) | 2,520 PFLOPS |
| FP8/FP6 Training (dense, total) | 1,260 PFLOPS |
| Total GPU Memory | 20.7 TB HBM4 |
| Total GPU Memory Bandwidth | 1,580 TB/s |
| Aggregate NVLink Bandwidth | 260 TB/s |
| NVLink-C2C Bandwidth | 65 TB/s |
| Total CPU Memory | 54 TB LPDDR5X |
| ConnectX-9 SuperNIC | 1.6 Tb/s per GPU |
| Rack Design | 3rd-gen MGX NVL72, cable-free modular trays |

### Vera CPU

| Spec | Value |
|---|---|
| Architecture | Arm-compatible, NVIDIA "Olympus" cores |
| Cores per CPU | 88 |
| Total Cores per Rack | 3,168 (36 x 88) |
| CPU Memory Bandwidth | Up to 1.2 TB/s LPDDR5X per chip |
| CPU-GPU Interconnect | NVLink-C2C (coherent) |

### Vera Rubin Superchip (2 GPUs + 1 CPU)

| Spec | Value |
|---|---|
| NVFP4 Inference | 100 PFLOPS |
| HBM4 Memory | 576 GB |
| Memory Bandwidth | 44 TB/s |
| NVLink Bandwidth | 7.2 TB/s |
| NVLink-C2C Bandwidth | 1.8 TB/s |

### vs. Blackwell GB200 NVL72

| Metric | Blackwell GB200 NVL72 | Vera Rubin NVL72 | Improvement |
|---|---|---|---|
| FP4 Inference (per GPU) | ~20 PFLOPS | 50 PFLOPS | ~2.5x |
| FP4 Inference (rack) | ~720 PFLOPS (dense) | 3,600 PFLOPS | ~5x |
| GPU Memory (rack) | 13.4 TB HBM3E | 20.7 TB HBM4 | ~1.5x |
| Memory Bandwidth (rack) | 576 TB/s | 1,580 TB/s | ~2.7x |
| NVLink Bandwidth (rack) | 130 TB/s | 260 TB/s | 2x |
| CPU | Grace (Neoverse V2, 72 cores) | Vera (Olympus, 88 cores) | Custom design |
| MoE Training Efficiency | Baseline | 4x fewer GPUs | NVIDIA claim |
| Inference Cost/Token | Baseline | 1/10th cost | NVIDIA claim |
| Inference Throughput/Watt | Baseline | Up to 10x | NVIDIA claim |
| Rack Assembly | Baseline | 18x faster | NVIDIA claim |

### Availability

Expected **H2 2026** via cloud and partner systems.
