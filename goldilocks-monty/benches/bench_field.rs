//! Benchmarks for Goldilocks Montgomery field implementation.
//!
//! This benchmark suite compares performance across different SIMD implementations:
//! - Scalar: Basic Montgomery arithmetic without vectorization
//! - AVX2: 4-wide vectorization (256-bit vectors)
//! - AVX512: 8-wide vectorization (512-bit vectors)
//!
//! ## Running Benchmarks
//!
//! To run all benchmarks:
//! ```bash
//! cargo bench --bench bench_field
//! ```
//!
//! To run benchmarks with AVX2 acceleration:
//! ```bash  
//! RUSTFLAGS="-C target-feature=+avx2" cargo bench --bench bench_field
//! ```
//!
//! To run benchmarks with AVX512 acceleration:
//! ```bash  
//! RUSTFLAGS="-C target-feature=+avx512f" cargo bench --bench bench_field
//! ```
//!
//! To run specific comparison benchmarks:
//! ```bash
//! RUSTFLAGS="-C target-feature=+avx2" cargo bench --bench bench_field -- avx2
//! RUSTFLAGS="-C target-feature=+avx512f" cargo bench --bench bench_field -- avx512
//! ```
//!
//! ## Performance Expectations
//!
//! Ideal performance scaling (elements/second):
//! - AVX2: ~4x scalar performance (4 elements per operation)
//! - AVX512: ~8x scalar performance (8 elements per operation)
//!
//! Real-world performance may vary due to CPU architecture, memory bandwidth,
//! and the specific operations being performed.

use core::any::type_name;
use std::hint::black_box;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use p3_field::{Field, PackedFieldPow2, PackedValue, PrimeCharacteristicRing};
use p3_field_testing::bench_func::{
    benchmark_add_latency, benchmark_add_throughput, benchmark_inv, benchmark_iter_sum,
    benchmark_sub_latency, benchmark_sub_throughput,
};
use p3_field_testing::{
    benchmark_dot_array, benchmark_mul_latency, benchmark_mul_throughput, benchmark_sum_array,
};
use p3_goldilocks_monty::Goldilocks;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512f")
))]
use p3_goldilocks_monty::PackedGoldilocksMontyAVX2;
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
use p3_goldilocks_monty::PackedGoldilocksMontyAVX512;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

type F = Goldilocks;

fn bench_field(c: &mut Criterion) {
    let name = "GoldilocksMonty";
    const REPS: usize = 200;
    benchmark_mul_latency::<F, 100>(c, name);
    benchmark_mul_throughput::<F, 25>(c, name);
    benchmark_inv::<F>(c, name);
    benchmark_iter_sum::<F, 4, REPS>(c, name);
    benchmark_sum_array::<F, 4, REPS>(c, name);

    benchmark_dot_array::<F, 1>(c, name);
    benchmark_dot_array::<F, 2>(c, name);
    benchmark_dot_array::<F, 3>(c, name);
    benchmark_dot_array::<F, 4>(c, name);
    benchmark_dot_array::<F, 5>(c, name);
    benchmark_dot_array::<F, 6>(c, name);

    // Note that each round of throughput has 10 operations
    // So we should have 10 * more repetitions for latency tests.
    const L_REPS: usize = 10 * REPS;
    benchmark_add_latency::<F, L_REPS>(c, name);
    benchmark_add_throughput::<F, REPS>(c, name);
    benchmark_sub_latency::<F, L_REPS>(c, name);
    benchmark_sub_throughput::<F, REPS>(c, name);

    let mut rng = SmallRng::seed_from_u64(1);
    c.bench_function("7th_root_monty", |b| {
        b.iter_batched(
            || rng.random::<F>(),
            |x| x.exp_u64(10540996611094048183),
            BatchSize::SmallInput,
        )
    });
}

fn bench_packedfield(c: &mut Criterion) {
    let name = type_name::<<F as Field>::Packing>().to_string();
    // Note that each round of throughput has 10 operations
    // So we should have 10 * more repetitions for latency tests.
    const REPS: usize = 100;
    const L_REPS: usize = 10 * REPS;

    benchmark_add_latency::<<F as Field>::Packing, L_REPS>(c, &name);
    benchmark_add_throughput::<<F as Field>::Packing, REPS>(c, &name);
    benchmark_sub_latency::<<F as Field>::Packing, L_REPS>(c, &name);
    benchmark_sub_throughput::<<F as Field>::Packing, REPS>(c, &name);
    benchmark_mul_latency::<<F as Field>::Packing, L_REPS>(c, &name);
    benchmark_mul_throughput::<<F as Field>::Packing, REPS>(c, &name);
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512f")
))]
fn bench_avx2_operations(c: &mut Criterion) {
    const SIZE: usize = 1024;
    let mut rng = SmallRng::seed_from_u64(42);

    // Generate test vectors
    let scalar_a: Vec<Goldilocks> = (0..SIZE).map(|_| rng.random()).collect();
    let scalar_b: Vec<Goldilocks> = (0..SIZE).map(|_| rng.random()).collect();

    // Convert to packed vectors (4 elements per AVX2 vector)
    let packed_a: Vec<PackedGoldilocksMontyAVX2> = scalar_a
        .chunks_exact(4)
        .map(|chunk| PackedGoldilocksMontyAVX2([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    let packed_b: Vec<PackedGoldilocksMontyAVX2> = scalar_b
        .chunks_exact(4)
        .map(|chunk| PackedGoldilocksMontyAVX2([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    // Benchmark scalar addition
    c.bench_function("scalar_add_array", |b| {
        b.iter(|| {
            let mut result = Vec::with_capacity(SIZE);
            for i in 0..SIZE {
                result.push(black_box(scalar_a[i] + scalar_b[i]));
            }
            result
        })
    });

    // Benchmark AVX2 addition
    c.bench_function("avx2_add_array", |b| {
        b.iter(|| {
            let mut result = Vec::with_capacity(SIZE / 4);
            for i in 0..(SIZE / 4) {
                result.push(black_box(packed_a[i] + packed_b[i]));
            }
            result
        })
    });

    // Benchmark scalar multiplication
    c.bench_function("scalar_mul_array", |b| {
        b.iter(|| {
            let mut result = Vec::with_capacity(SIZE);
            for i in 0..SIZE {
                result.push(black_box(scalar_a[i] * scalar_b[i]));
            }
            result
        })
    });

    // Benchmark AVX2 multiplication
    c.bench_function("avx2_mul_array", |b| {
        b.iter(|| {
            let mut result = Vec::with_capacity(SIZE / 4);
            for i in 0..(SIZE / 4) {
                result.push(black_box(packed_a[i] * packed_b[i]));
            }
            result
        })
    });

    // Benchmark scalar sum
    c.bench_function("scalar_sum", |b| {
        b.iter(|| {
            let mut sum = Goldilocks::ZERO;
            for &val in &scalar_a {
                sum += black_box(val);
            }
            sum
        })
    });

    // Benchmark AVX2 sum
    c.bench_function("avx2_sum", |b| {
        b.iter(|| {
            let mut sum = PackedGoldilocksMontyAVX2::ZERO;
            for &val in &packed_a {
                sum += black_box(val);
            }
            // Sum the components of the packed result
            sum.as_slice()
                .iter()
                .fold(Goldilocks::ZERO, |acc, &x| acc + x)
        })
    });

    // Benchmark dot product scalar
    c.bench_function("scalar_dot_product", |b| {
        b.iter(|| {
            let mut result = Goldilocks::ZERO;
            for i in 0..SIZE {
                result += black_box(scalar_a[i] * scalar_b[i]);
            }
            result
        })
    });

    // Benchmark dot product AVX2
    c.bench_function("avx2_dot_product", |b| {
        b.iter(|| {
            let mut sum = PackedGoldilocksMontyAVX2::ZERO;
            for i in 0..(SIZE / 4) {
                sum += black_box(packed_a[i] * packed_b[i]);
            }
            // Sum the components of the packed result
            sum.as_slice()
                .iter()
                .fold(Goldilocks::ZERO, |acc, &x| acc + x)
        })
    });
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
fn bench_avx512_operations(c: &mut Criterion) {
    const SIZE: usize = 1024;
    let mut rng = SmallRng::seed_from_u64(42);

    // Generate test vectors
    let scalar_a: Vec<Goldilocks> = (0..SIZE).map(|_| rng.random()).collect();
    let scalar_b: Vec<Goldilocks> = (0..SIZE).map(|_| rng.random()).collect();

    // Convert to packed vectors (8 elements per AVX512 vector)
    let packed_a: Vec<PackedGoldilocksMontyAVX512> = scalar_a
        .chunks_exact(8)
        .map(|chunk| {
            PackedGoldilocksMontyAVX512([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ])
        })
        .collect();
    let packed_b: Vec<PackedGoldilocksMontyAVX512> = scalar_b
        .chunks_exact(8)
        .map(|chunk| {
            PackedGoldilocksMontyAVX512([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ])
        })
        .collect();

    // Benchmark scalar addition
    c.bench_function("scalar_add_array", |b| {
        b.iter(|| {
            let mut result = Vec::with_capacity(SIZE);
            for i in 0..SIZE {
                result.push(black_box(scalar_a[i] + scalar_b[i]));
            }
            result
        })
    });

    // Benchmark AVX512 addition
    c.bench_function("avx512_add_array", |b| {
        b.iter(|| {
            let mut result = Vec::with_capacity(SIZE / 8);
            for i in 0..(SIZE / 8) {
                result.push(black_box(packed_a[i] + packed_b[i]));
            }
            result
        })
    });

    // Benchmark scalar multiplication
    c.bench_function("scalar_mul_array", |b| {
        b.iter(|| {
            let mut result = Vec::with_capacity(SIZE);
            for i in 0..SIZE {
                result.push(black_box(scalar_a[i] * scalar_b[i]));
            }
            result
        })
    });

    // Benchmark AVX512 multiplication
    c.bench_function("avx512_mul_array", |b| {
        b.iter(|| {
            let mut result = Vec::with_capacity(SIZE / 8);
            for i in 0..(SIZE / 8) {
                result.push(black_box(packed_a[i] * packed_b[i]));
            }
            result
        })
    });

    // Benchmark scalar sum
    c.bench_function("scalar_sum", |b| {
        b.iter(|| {
            let mut sum = Goldilocks::ZERO;
            for &val in &scalar_a {
                sum += black_box(val);
            }
            sum
        })
    });

    // Benchmark AVX512 sum
    c.bench_function("avx512_sum", |b| {
        b.iter(|| {
            let mut sum = PackedGoldilocksMontyAVX512::ZERO;
            for &val in &packed_a {
                sum += black_box(val);
            }
            // Sum the components of the packed result
            sum.as_slice()
                .iter()
                .fold(Goldilocks::ZERO, |acc, &x| acc + x)
        })
    });

    // Benchmark dot product scalar
    c.bench_function("scalar_dot_product", |b| {
        b.iter(|| {
            let mut result = Goldilocks::ZERO;
            for i in 0..SIZE {
                result += black_box(scalar_a[i] * scalar_b[i]);
            }
            result
        })
    });

    // Benchmark dot product AVX512
    c.bench_function("avx512_dot_product", |b| {
        b.iter(|| {
            let mut sum = PackedGoldilocksMontyAVX512::ZERO;
            for i in 0..(SIZE / 8) {
                sum += black_box(packed_a[i] * packed_b[i]);
            }
            // Sum the components of the packed result
            sum.as_slice()
                .iter()
                .fold(Goldilocks::ZERO, |acc, &x| acc + x)
        })
    });

    // AVX512-specific benchmarks
    c.bench_function("avx512_interleave_test", |b| {
        let a = PackedGoldilocksMontyAVX512::from(Goldilocks::from_canonical_u64(123));
        let b = PackedGoldilocksMontyAVX512::from(Goldilocks::from_canonical_u64(456));

        b.iter(|| {
            let (r1, r2) = black_box(a.interleave(b, 1));
            let (r3, r4) = black_box(a.interleave(b, 2));
            let (r5, r6) = black_box(a.interleave(b, 4));
            let (r7, r8) = black_box(a.interleave(b, 8));
            (r1, r2, r3, r4, r5, r6, r7, r8)
        })
    });

    // Matrix operations benchmark - useful for cryptographic operations
    c.bench_function("avx512_matrix_mul_4x4", |b| {
        let matrix_a: Vec<Vec<Goldilocks>> = (0..4)
            .map(|_| (0..4).map(|_| rng.random()).collect())
            .collect();
        let matrix_b: Vec<Vec<Goldilocks>> = (0..4)
            .map(|_| (0..4).map(|_| rng.random()).collect())
            .collect();

        b.iter(|| {
            let mut result = vec![vec![Goldilocks::ZERO; 4]; 4];
            for i in 0..4 {
                for j in 0..4 {
                    let mut sum = Goldilocks::ZERO;
                    for k in 0..4 {
                        sum += black_box(matrix_a[i][k] * matrix_b[k][j]);
                    }
                    result[i][j] = sum;
                }
            }
            result
        })
    });

    // Large array operations - typical in zkSNARK computations
    const LARGE_SIZE: usize = 8192;
    let large_a: Vec<Goldilocks> = (0..LARGE_SIZE).map(|_| rng.random()).collect();
    let large_b: Vec<Goldilocks> = (0..LARGE_SIZE).map(|_| rng.random()).collect();

    let large_packed_a: Vec<PackedGoldilocksMontyAVX512> = large_a
        .chunks_exact(8)
        .map(|chunk| {
            PackedGoldilocksMontyAVX512([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ])
        })
        .collect();
    let large_packed_b: Vec<PackedGoldilocksMontyAVX512> = large_b
        .chunks_exact(8)
        .map(|chunk| {
            PackedGoldilocksMontyAVX512([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ])
        })
        .collect();

    c.bench_function("avx512_large_array_ops", |b| {
        b.iter(|| {
            let mut result = Vec::with_capacity(LARGE_SIZE / 8);
            for i in 0..(LARGE_SIZE / 8) {
                let sum = black_box(large_packed_a[i] + large_packed_b[i]);
                let prod = black_box(large_packed_a[i] * large_packed_b[i]);
                result.push(black_box(sum * prod));
            }
            result
        })
    });
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512f")
))]
criterion_group!(
    goldilocks_monty_arithmetic,
    bench_field,
    bench_packedfield,
    bench_avx2_operations
);

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
criterion_group!(
    goldilocks_monty_arithmetic,
    bench_field,
    bench_packedfield,
    bench_avx512_operations
);

#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512f")
)))]
criterion_group!(goldilocks_monty_arithmetic, bench_field, bench_packedfield);

criterion_main!(goldilocks_monty_arithmetic);
