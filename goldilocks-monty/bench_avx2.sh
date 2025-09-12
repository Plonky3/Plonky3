#!/bin/bash

# Benchmark runner for goldilocks-monty AVX2 implementation
# This script runs comprehensive benchmarks comparing scalar vs AVX2 performance

echo "=== Goldilocks Montgomery AVX2 Benchmarks ==="
echo ""

# Check if we're on x86_64
if [ $(uname -m) != "x86_64" ]; then
    echo "Warning: AVX2 benchmarks are only available on x86_64 architecture"
    echo "Running standard benchmarks only..."
    cargo bench --bench bench_field
    exit 0
fi

# Check if AVX2 is supported (this is a simple check)
echo "Running with AVX2 enabled..."
echo ""

# Run standard field benchmarks with AVX2 
echo "1. Standard field benchmarks (includes PackedField with AVX2):"
RUSTFLAGS="-C target-feature=+avx2" cargo bench --bench bench_field -- --exclude avx2

echo ""
echo "2. AVX2 vs Scalar comparison benchmarks:"

# Run AVX2 vs scalar comparison
RUSTFLAGS="-C target-feature=+avx2" cargo bench --bench bench_field -- avx2

echo ""
echo "3. Extension field benchmarks:"
RUSTFLAGS="-C target-feature=+avx2" cargo bench --bench extension

echo ""
echo "=== Benchmark Complete ==="
echo ""
echo "Key takeaways:"
echo "- avx2_* benchmarks process 4 field elements simultaneously"
echo "- Speedup = scalar_time / (avx2_time * 4) should be > 1 for good performance"
echo "- PackedField benchmarks show the performance of the field's Packing type"