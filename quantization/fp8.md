# FP8 (8-bit Floating Point)

## Motivation

Imagine FP16 -> FP8:

- Memory: 1/2
- Throughput: 2x
- Accuracy: Drop <1%

## Floating-Point Basics

Floating-point numbers represent real numbers using three components:

- **Sign bit (S):** 1 bit, positive (0) or negative (1)
- **Exponent (E):** Bits determining the scale/range (powers of 2)
- **Mantissa/Fraction (M):** Bits determining precision (significant digits)

**Formula:** `(-1)^S × (1.M) × 2^(E - bias)`

## FP8 Data Types

| Format | Sign | Exponent | Mantissa | Dynamic Range | Precision |
|--------|------|----------|----------|---------------|-----------|
| E4M3   | 1    | 4        | 3        | Smaller (~±448) | Higher |
| E5M2   | 1    | 5        | 2        | Larger (~±57344) | Lower |

- **E4M3:** More mantissa bits = better precision, but narrower range.
- **E5M2:** More exponent bits = wider dynamic range, but coarser precision.

In practice, E4M3 is used for inference (weights and activations).
E5M2 is used for training larger gradients, where wider dynamic range matters.

## E4M3FN

The `fn` suffix in `float8_e4m3fn` stands for "finite, with NaN".
It removes infinities and uses those bit patterns for normal numbers.
Keeps one NaN encoding (for debugging/safety).
Result: max value 448 instead of 240.
Neural network weights/activations rarely need infinities, so this is a good tradeoff.

**Max value calculation:**

```
S = 0
E = 15 (stored) → actual exponent = 15 - 7 = 8
M = 110 (binary) → 1.110 = 1 + 1/2 + 1/4 + 0/8 = 1.75
(111 is reserved for NaN, so max mantissa is 110)

Max = (-1)^0 × 1.75 × 2^8 = 1 × 1.75 × 256 = 448
```

## FP8 Scaling

Scaling is needed for two reasons:

1. **Range**: Values outside ±448 would overflow/clip.
2. **Precision**: Values within range but too small (e.g. 0.001) land in a region where FP8 has very sparse representable values. E4M3FN's smallest positive value is ~0.00195, so anything smaller rounds to 0. Values like `[0.001, 0.003, 0.002]` would collapse into the same FP8 value or zero, destroying information.

In practice, precision is the bigger concern — most NN weights/activations are small (0.001–0.1), well within ±448 but exactly where FP8 is least precise. Scaling pushes them into a denser region.

Given `X` = FP16 tensor, `fmax` = 448 (E4M3 max):

```
1. scale = fmax / amax(X).clamp(min=1e-12)
   - amax(X) = max absolute value in the tensor
   - clamp prevents division by zero

2. Y = (X * scale).clamp(-fmax, fmax).to(fp8)
   - Scale up, clamp for safety, cast to FP8

3. Dequantize: X_approx = Y.to(fp16) / scale
```

**Example:**

```
X = [0.5, -1.2, 3.0, -0.8]
amax(X) = 3.0
scale = 448 / 3.0 = 149.33
X * scale = [74.67, -179.2, 448.0, -119.47]  → values now use FP8 range well
```

**Scaling granularity:**

| Granularity | Scale Factors | Accuracy | Usage |
|-------------|---------------|----------|-------|
| Per-tensor  | 1 per entire tensor | Lowest (simplest) | Most common today (NVIDIA FP8 recipe, vLLM) |
| Per-channel | 1 per output channel | Medium | Weight quantization |
| Per-token   | 1 per token/row | Higher | Activation quantization |
| Per-block   | 1 per small block (e.g. 128 elements) | Highest | OCP MX spec, DeepSeek-V3 |

**Static vs Dynamic scaling:**

| | Static | Dynamic |
|--|--------|---------|
| Scale | Fixed (precomputed) | Computed per tensor at runtime |
| Accuracy | Depends on calibration quality | Higher (always optimal) |
| Calibration data needed | Yes | No |
| Runtime overhead | None | Small (amax reduction per layer) |

- **Static:** Scale factors are computed once using calibration data (a small representative dataset, e.g., 128–512 samples run through the model to observe actual value ranges per layer). Scales are baked into the checkpoint. Zero runtime cost.
- **Dynamic:** Computes `amax(X)` for each tensor at runtime, so the scale always fits the actual data. More accurate, but adds an extra GPU kernel per layer. Introucing overheads.

In production, **static scaling dominates**. Model providers typically release pre-quantized checkpoints (e.g., `model-fp8.safetensors`) with baked-in scales. Dynamic scaling is mainly for experimentation or when no pre-quantized checkpoint is available.

## Misc

**Inter-token latency (ITL)** is the time between generating consecutive tokens during the decoding phase of LLM inference.
