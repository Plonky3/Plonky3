# Benchmark Comparison: Goldilocks vs Goldilocks-Monty

This table compares benchmark results from September 8th, 2024 between the regular Goldilocks field implementation and the new Goldilocks-Monty (Montgomery form) implementation with AVX2 support.

## Field Operations Comparison

| Operation | Goldilocks (Regular) | Goldilocks-Monty | Relative Performance | Notes |
|-----------|---------------------|------------------|---------------------|-------|
| **Basic Operations** | | | | |
| Multiplication Latency (100 ops) | 169.42 ns | 186.36 ns | **1.10x slower** | Montgomery reduction overhead |
| Multiplication Throughput (25 ops) | 138.98 ns | 268.38 ns | **1.93x slower** | Montgomery overhead more pronounced |
| Inverse | 39.44 ns | 395.66 ns | **10.03x slower** | Montgomery inverse is complex |
| Addition Latency (2000 ops) | 1.104 µs | 1.113 µs | **1.01x slower** | Minimal difference |
| Addition Throughput (200 ops) | 474.13 ns | 616.38 ns | **1.30x slower** | Some Montgomery overhead |
| Subtraction Latency (2000 ops) | 1.117 µs | 1.109 µs | **1.01x faster** | Comparable performance |
| Subtraction Throughput (200 ops) | 430.73 ns | 483.47 ns | **1.12x slower** | Minor overhead |
| 7th Root | 142.28 ns | 131.29 ns | **1.08x faster** | Monty performs better |
| **Summation & Dot Products** | | | | |
| Sum (200 elements, size 4) | 185.86 ns | 261.39 ns | **1.41x slower** | Montgomery overhead |
| Tree Sum (200 elements, size 4) | 179.39 ns | 243.74 ns | **1.36x slower** | Montgomery overhead |
| Dot Product (size 1) | 739.36 ps | 929.45 ps | **1.26x slower** | Single element overhead |
| Dot Product (size 2) | 1.009 ns | 2.018 ns | **2.00x slower** | Proportional to size |
| Dot Product (size 3) | 1.653 ns | 2.520 ns | **1.52x slower** | Montgomery overhead |
| Dot Product (size 4) | 1.998 ns | 4.924 ns | **2.46x slower** | Significant overhead |
| Dot Product (size 5) | 2.377 ns | 5.399 ns | **2.27x slower** | Overhead continues |
| Dot Product (size 6) | 2.753 ns | 6.124 ns | **2.22x slower** | Montgomery overhead |

## Packed Field Operations (AVX2)

| Operation | Goldilocks PackedAVX2 | Goldilocks-Monty PackedAVX2 | Relative Performance | Notes |
|-----------|----------------------|----------------------------|---------------------|-------|
| **Packed Operations (4 elements)** | | | | |
| Add Latency (1000 ops) | 1.158 µs | 566.02 ns | **2.04x faster** | Monty-AVX2 more efficient |
| Add Throughput (100 ops) | 499.24 ns | 307.84 ns | **1.62x faster** | Better throughput |
| Sub Latency (1000 ops) | 969.34 ns | 559.44 ns | **1.73x faster** | Monty-AVX2 advantage |
| Sub Throughput (100 ops) | 441.56 ns | 239.86 ns | **1.84x faster** | Significant improvement |
| Mul Latency (1000 ops) | 3.745 µs | 1.825 µs | **2.05x faster** | Major improvement |
| Mul Throughput (100 ops) | 1.604 µs | 1.046 µs | **1.53x faster** | Better packed multiplication |

## Custom AVX2 vs Scalar Benchmarks (Goldilocks-Monty Only)

| Operation | Scalar (1024 elements) | AVX2 (256 packed, 1024 total) | Speedup | Efficiency |
|-----------|----------------------|------------------------------|---------|------------|
| **Array Operations** | | | | |
| Addition Array | 1.041 µs | 334.39 ns | **3.11x** | 77.8% of theoretical 4x |
| Multiplication Array | 1.844 µs | 1.251 µs | **1.47x** | 36.8% of theoretical 4x |
| Sum Reduction | 585.26 ns | 250.01 ns | **2.34x** | 58.5% of theoretical 4x |
| Dot Product | 1.188 µs | 1.188 µs | **1.00x** | No improvement (dominated by reduction) |

## Extension Field Comparison

### Quadratic Extension (degree 2)

| Operation | Goldilocks EF2 | Goldilocks-Monty EF2 | Relative Performance |
|-----------|----------------|----------------------|---------------------|
| Square | 2.748 ns | 4.166 ns | **1.52x slower** |
| Inverse | 54.46 ns | 401.18 ns | **7.37x slower** |
| Multiplication | 2.362 ns | 6.037 ns | **2.56x slower** |
| Mul Throughput (50 ops) | 1.099 µs | 2.497 µs | **2.27x slower** |
| Mul Latency (500 ops) | 3.850 µs | 3.101 µs | **1.24x faster** |

### Quintic Extension (degree 5)

| Operation | Goldilocks EF5 | Goldilocks-Monty EF5 | Relative Performance |
|-----------|----------------|----------------------|---------------------|
| Square | 11.09 ns | 18.53 ns | **1.67x slower** |
| Inverse | 111.06 ns | 509.38 ns | **4.59x slower** |
| Multiplication | 18.34 ns | 32.73 ns | **1.78x slower** |
| Mul Throughput (50 ops) | 7.905 µs | 15.39 µs | **1.95x slower** |
| Mul Latency (500 ops) | 10.79 µs | 16.31 µs | **1.51x slower** |

## Summary & Analysis

### Key Findings:

1. **Montgomery Overhead**: Montgomery form introduces significant computational overhead for most operations, especially:
   - **Inverse operations**: 10x slower for base field, 4-7x slower for extensions
   - **Multiplication**: 1.1-2.5x slower across different contexts
   - **Addition/Subtraction**: 1.1-1.4x slower, relatively minor impact

2. **AVX2 Effectiveness**: 
   - **Goldilocks-Monty AVX2 outperforms regular Goldilocks AVX2** by 1.5-2x in packed operations
   - **Custom AVX2 benchmarks show 1.5-3.1x speedup** over scalar operations
   - **Addition benefits most** from vectorization (3.1x speedup)
   - **Multiplication less effective** due to Montgomery reduction complexity (1.47x speedup)

3. **Extension Fields**: Montgomery overhead compounds in extension fields, with EF2 showing 2-7x slower performance and EF5 showing 1.5-4.6x slower performance.

4. **Trade-offs**: 
   - Montgomery form provides **cryptographic benefits** (side-channel resistance, efficient modular arithmetic in some contexts)
   - **Performance cost** is significant but partially mitigated by AVX2 vectorization
   - **AVX2 implementation more effective** for Montgomery than expected, showing the vectorization can overcome some of the algorithmic overhead

### Recommendations:

1. **Use Montgomery form** when cryptographic security properties are required
2. **Enable AVX2** when using Montgomery form to maximize performance 
3. **Consider hybrid approaches** for performance-critical applications where security requirements vary
4. **Focus optimization efforts** on multiplication and inverse operations where Montgomery overhead is highest