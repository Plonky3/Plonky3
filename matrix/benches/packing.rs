//! Benchmarks for matrix packing operations.
//!
//! These benchmarks measure the performance of:
//! - `vertically_packed_row`: Pack WIDTH consecutive rows into packed field elements
//! - `vertically_packed_row_pair`: Pack two sets of WIDTH rows (current + next)
//! - `horizontally_packed_row`: Pack a single row into packed field elements
//! - `padded_horizontally_packed_row`: Same as above, with zero-padding for remainder
//!
//! ## Running benchmarks
//!
//! ```bash
//! # Without WIDTH=1 optimization (baseline)
//! RUSTFLAGS="-Ctarget-cpu=native" cargo bench --bench packing -- --save-baseline original
//!
//! # With WIDTH=1 optimization
//! RUSTFLAGS="-Ctarget-cpu=native" cargo bench --bench packing --features width1_opt -- --baseline original
//! ```
//!
//! ## Expected results
//!
//! - Goldilocks (WIDTH=1 on ARM): 30-40% improvement with width1_opt
//! - BabyBear (WIDTH=4 on ARM NEON): No significant change (±5%)

use criterion::{
    BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main,
};
use p3_baby_bear::BabyBear;
use p3_field::{Field, PackedValue};
use p3_goldilocks::Goldilocks;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use rand::rngs::SmallRng;
use rand::SeedableRng;

/// Matrix height for all benchmarks (2^16 = 65536 rows)
const LOG_HEIGHT: usize = 16;
const HEIGHT: usize = 1 << LOG_HEIGHT;

/// Matrix widths to test:
/// - 128, 1024: Powers of 2, aligned to all packing widths
/// - 129, 1025: Off-by-one, tests padding/remainder handling
const WIDTHS: &[usize] = &[128, 129, 1024, 1025];

/// Step size for vertically_packed_row_pair (simulates constraint evaluation stride)
/// Using height/2 is realistic for "next row" access patterns
const STEP: usize = HEIGHT / 2;

// ============================================================================
// Vertical Packing Benchmarks
// ============================================================================

/// Benchmark `vertically_packed_row` for Goldilocks field.
///
/// On ARM without AVX, Goldilocks::Packing::WIDTH = 1 (scalar).
/// This is where the WIDTH=1 optimization should show significant gains.
fn bench_vertically_packed_row_goldilocks(c: &mut Criterion) {
    let mut group = c.benchmark_group("vertically_packed_row/goldilocks");
    group.sample_size(50);

    type F = Goldilocks;
    type P = <F as Field>::Packing;
    let packing_width = P::WIDTH;

    for &width in WIDTHS {
        let mut rng = SmallRng::seed_from_u64(42);
        let matrix = RowMajorMatrix::<F>::rand(&mut rng, HEIGHT, width);

        // Total packed elements produced = (height / packing_width) * width
        let num_calls = HEIGHT / packing_width;
        let elements_per_call = width;
        let total_elements = num_calls * elements_per_call;
        group.throughput(Throughput::Elements(total_elements as u64));

        group.bench_with_input(
            BenchmarkId::new(format!("w{packing_width}"), format!("{width}x{HEIGHT}")),
            &matrix,
            |b, matrix| {
                b.iter(|| {
                    for r in (0..HEIGHT).step_by(packing_width) {
                        // Consume the iterator to force evaluation
                        for packed in matrix.vertically_packed_row::<P>(r) {
                            let _ = black_box(packed);
                        }
                    }
                })
            },
        );
    }
    group.finish();
}

/// Benchmark `vertically_packed_row` for BabyBear field.
///
/// On ARM with NEON, BabyBear::Packing::WIDTH = 4.
/// The WIDTH=1 optimization should not affect this path.
fn bench_vertically_packed_row_babybear(c: &mut Criterion) {
    let mut group = c.benchmark_group("vertically_packed_row/babybear");
    group.sample_size(50);

    type F = BabyBear;
    type P = <F as Field>::Packing;
    let packing_width = P::WIDTH;

    for &width in WIDTHS {
        let mut rng = SmallRng::seed_from_u64(42);
        let matrix = RowMajorMatrix::<F>::rand(&mut rng, HEIGHT, width);

        let num_calls = HEIGHT / packing_width;
        let elements_per_call = width;
        let total_elements = num_calls * elements_per_call;
        group.throughput(Throughput::Elements(total_elements as u64));

        group.bench_with_input(
            BenchmarkId::new(format!("w{packing_width}"), format!("{width}x{HEIGHT}")),
            &matrix,
            |b, matrix| {
                b.iter(|| {
                    for r in (0..HEIGHT).step_by(packing_width) {
                        for packed in matrix.vertically_packed_row::<P>(r) {
                            let _ = black_box(packed);
                        }
                    }
                })
            },
        );
    }
    group.finish();
}

/// Benchmark `vertically_packed_row_pair` for Goldilocks field.
///
/// This function returns a `Vec<P>` containing two packed rows.
/// The WIDTH=1 optimization should show significant gains here.
fn bench_vertically_packed_row_pair_goldilocks(c: &mut Criterion) {
    let mut group = c.benchmark_group("vertically_packed_row_pair/goldilocks");
    group.sample_size(50);

    type F = Goldilocks;
    type P = <F as Field>::Packing;
    let packing_width = P::WIDTH;

    for &width in WIDTHS {
        let mut rng = SmallRng::seed_from_u64(42);
        let matrix = RowMajorMatrix::<F>::rand(&mut rng, HEIGHT, width);

        // Returns 2*width packed elements per call
        let num_calls = HEIGHT / packing_width;
        let elements_per_call = width * 2;
        let total_elements = num_calls * elements_per_call;
        group.throughput(Throughput::Elements(total_elements as u64));

        group.bench_with_input(
            BenchmarkId::new(format!("w{packing_width}"), format!("{width}x{HEIGHT}")),
            &matrix,
            |b, matrix| {
                b.iter(|| {
                    for r in (0..HEIGHT).step_by(packing_width) {
                        let packed = matrix.vertically_packed_row_pair::<P>(r, STEP);
                        let _ = black_box(packed);
                    }
                })
            },
        );
    }
    group.finish();
}

/// Benchmark `vertically_packed_row_pair` for BabyBear field.
fn bench_vertically_packed_row_pair_babybear(c: &mut Criterion) {
    let mut group = c.benchmark_group("vertically_packed_row_pair/babybear");
    group.sample_size(50);

    type F = BabyBear;
    type P = <F as Field>::Packing;
    let packing_width = P::WIDTH;

    for &width in WIDTHS {
        let mut rng = SmallRng::seed_from_u64(42);
        let matrix = RowMajorMatrix::<F>::rand(&mut rng, HEIGHT, width);

        let num_calls = HEIGHT / packing_width;
        let elements_per_call = width * 2;
        let total_elements = num_calls * elements_per_call;
        group.throughput(Throughput::Elements(total_elements as u64));

        group.bench_with_input(
            BenchmarkId::new(format!("w{packing_width}"), format!("{width}x{HEIGHT}")),
            &matrix,
            |b, matrix| {
                b.iter(|| {
                    for r in (0..HEIGHT).step_by(packing_width) {
                        let packed = matrix.vertically_packed_row_pair::<P>(r, STEP);
                        let _ = black_box(packed);
                    }
                })
            },
        );
    }
    group.finish();
}

// ============================================================================
// Horizontal Packing Benchmarks
// ============================================================================

/// Benchmark `horizontally_packed_row` for Goldilocks field.
///
/// This function packs a single row horizontally, returning packed elements
/// plus any remainder elements that don't fit a full pack.
fn bench_horizontally_packed_row_goldilocks(c: &mut Criterion) {
    let mut group = c.benchmark_group("horizontally_packed_row/goldilocks");
    group.sample_size(50);

    type F = Goldilocks;
    type P = <F as Field>::Packing;
    let packing_width = P::WIDTH;

    for &width in WIDTHS {
        let mut rng = SmallRng::seed_from_u64(42);
        let matrix = RowMajorMatrix::<F>::rand(&mut rng, HEIGHT, width);

        // Process all rows, each row produces width elements
        let total_elements = HEIGHT * width;
        group.throughput(Throughput::Elements(total_elements as u64));

        group.bench_with_input(
            BenchmarkId::new(format!("w{packing_width}"), format!("{width}x{HEIGHT}")),
            &matrix,
            |b, matrix| {
                b.iter(|| {
                    for r in 0..HEIGHT {
                        let (packed, suffix) = matrix.horizontally_packed_row::<P>(r);
                        for p in packed {
                            let _ = black_box(p);
                        }
                        for s in suffix {
                            let _ = black_box(s);
                        }
                    }
                })
            },
        );
    }
    group.finish();
}

/// Benchmark `horizontally_packed_row` for BabyBear field.
fn bench_horizontally_packed_row_babybear(c: &mut Criterion) {
    let mut group = c.benchmark_group("horizontally_packed_row/babybear");
    group.sample_size(50);

    type F = BabyBear;
    type P = <F as Field>::Packing;
    let packing_width = P::WIDTH;

    for &width in WIDTHS {
        let mut rng = SmallRng::seed_from_u64(42);
        let matrix = RowMajorMatrix::<F>::rand(&mut rng, HEIGHT, width);

        let total_elements = HEIGHT * width;
        group.throughput(Throughput::Elements(total_elements as u64));

        group.bench_with_input(
            BenchmarkId::new(format!("w{packing_width}"), format!("{width}x{HEIGHT}")),
            &matrix,
            |b, matrix| {
                b.iter(|| {
                    for r in 0..HEIGHT {
                        let (packed, suffix) = matrix.horizontally_packed_row::<P>(r);
                        for p in packed {
                            let _ = black_box(p);
                        }
                        for s in suffix {
                            let _ = black_box(s);
                        }
                    }
                })
            },
        );
    }
    group.finish();
}

/// Benchmark `padded_horizontally_packed_row` for Goldilocks field.
///
/// Similar to horizontally_packed_row but zero-pads the last element
/// instead of returning a separate suffix iterator.
fn bench_padded_horizontally_packed_row_goldilocks(c: &mut Criterion) {
    let mut group = c.benchmark_group("padded_horizontally_packed_row/goldilocks");
    group.sample_size(50);

    type F = Goldilocks;
    type P = <F as Field>::Packing;
    let packing_width = P::WIDTH;

    for &width in WIDTHS {
        let mut rng = SmallRng::seed_from_u64(42);
        let matrix = RowMajorMatrix::<F>::rand(&mut rng, HEIGHT, width);

        // Produces ceil(width / packing_width) packed elements per row
        let packed_per_row = width.div_ceil(packing_width);
        let total_packed = HEIGHT * packed_per_row;
        group.throughput(Throughput::Elements(total_packed as u64));

        group.bench_with_input(
            BenchmarkId::new(format!("w{packing_width}"), format!("{width}x{HEIGHT}")),
            &matrix,
            |b, matrix| {
                b.iter(|| {
                    for r in 0..HEIGHT {
                        for packed in matrix.padded_horizontally_packed_row::<P>(r) {
                            let _ = black_box(packed);
                        }
                    }
                })
            },
        );
    }
    group.finish();
}

/// Benchmark `padded_horizontally_packed_row` for BabyBear field.
fn bench_padded_horizontally_packed_row_babybear(c: &mut Criterion) {
    let mut group = c.benchmark_group("padded_horizontally_packed_row/babybear");
    group.sample_size(50);

    type F = BabyBear;
    type P = <F as Field>::Packing;
    let packing_width = P::WIDTH;

    for &width in WIDTHS {
        let mut rng = SmallRng::seed_from_u64(42);
        let matrix = RowMajorMatrix::<F>::rand(&mut rng, HEIGHT, width);

        let packed_per_row = width.div_ceil(packing_width);
        let total_packed = HEIGHT * packed_per_row;
        group.throughput(Throughput::Elements(total_packed as u64));

        group.bench_with_input(
            BenchmarkId::new(format!("w{packing_width}"), format!("{width}x{HEIGHT}")),
            &matrix,
            |b, matrix| {
                b.iter(|| {
                    for r in 0..HEIGHT {
                        for packed in matrix.padded_horizontally_packed_row::<P>(r) {
                            let _ = black_box(packed);
                        }
                    }
                })
            },
        );
    }
    group.finish();
}

// ============================================================================
// Single-Call Overhead Benchmarks
// ============================================================================

/// Benchmark single-call overhead for `vertically_packed_row`.
///
/// Uses a small width to minimize loop time and expose setup overhead.
/// This is where allocation elimination should be most visible.
fn bench_single_call_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_call_overhead");
    group.sample_size(100);

    // Small width to minimize loop time, exposing setup overhead
    let width = 16;
    let height = 1024;

    // Goldilocks
    {
        type F = Goldilocks;
        type P = <F as Field>::Packing;
        let packing_width = P::WIDTH;

        let mut rng = SmallRng::seed_from_u64(42);
        let matrix = RowMajorMatrix::<F>::rand(&mut rng, height, width);

        group.throughput(Throughput::Elements(width as u64));
        group.bench_function(
            BenchmarkId::new("vertically_packed_row", format!("goldilocks_w{packing_width}")),
            |b| {
                let mut r = 0usize;
                b.iter(|| {
                    for packed in matrix.vertically_packed_row::<P>(r) {
                        let _ = black_box(packed);
                    }
                    r = (r + packing_width) % height;
                })
            },
        );

        group.throughput(Throughput::Elements((width * 2) as u64));
        group.bench_function(
            BenchmarkId::new(
                "vertically_packed_row_pair",
                format!("goldilocks_w{packing_width}"),
            ),
            |b| {
                let mut r = 0usize;
                b.iter(|| {
                    let packed = matrix.vertically_packed_row_pair::<P>(r, height / 2);
                    let _ = black_box(packed);
                    r = (r + packing_width) % height;
                })
            },
        );
    }

    // BabyBear
    {
        type F = BabyBear;
        type P = <F as Field>::Packing;
        let packing_width = P::WIDTH;

        let mut rng = SmallRng::seed_from_u64(42);
        let matrix = RowMajorMatrix::<F>::rand(&mut rng, height, width);

        group.throughput(Throughput::Elements(width as u64));
        group.bench_function(
            BenchmarkId::new("vertically_packed_row", format!("babybear_w{packing_width}")),
            |b| {
                let mut r = 0usize;
                b.iter(|| {
                    for packed in matrix.vertically_packed_row::<P>(r) {
                        let _ = black_box(packed);
                    }
                    r = (r + packing_width) % height;
                })
            },
        );

        group.throughput(Throughput::Elements((width * 2) as u64));
        group.bench_function(
            BenchmarkId::new(
                "vertically_packed_row_pair",
                format!("babybear_w{packing_width}"),
            ),
            |b| {
                let mut r = 0usize;
                b.iter(|| {
                    let packed = matrix.vertically_packed_row_pair::<P>(r, height / 2);
                    let _ = black_box(packed);
                    r = (r + packing_width) % height;
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    // Vertical packing (main optimization target)
    bench_vertically_packed_row_goldilocks,
    bench_vertically_packed_row_babybear,
    bench_vertically_packed_row_pair_goldilocks,
    bench_vertically_packed_row_pair_babybear,
    // Horizontal packing (should be unchanged)
    bench_horizontally_packed_row_goldilocks,
    bench_horizontally_packed_row_babybear,
    bench_padded_horizontally_packed_row_goldilocks,
    bench_padded_horizontally_packed_row_babybear,
    // Single-call overhead (exposes setup cost)
    bench_single_call_overhead,
);

criterion_main!(benches);
