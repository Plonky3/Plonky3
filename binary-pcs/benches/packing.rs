//! Packing a narrow-alphabet trace into the elements a commitment holds.
//!
//! Two arms per group, over the same data:
//!
//! ```text
//!     packed     the layout change this crate ships
//!     per_cell   the same result, one cell or one bit at a time
//! ```
//!
//! The per-cell arm is the reference the tests pin correctness against.
//! The ratio therefore says what the layout change buys over the obvious loop.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_binary_field::{BinaryField32, BinaryField128, Gf2, PackedGf2, PackedGf2x64, Underlier};
use p3_binary_pcs::{Coordinates, PackedStack, pack};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

type EF = BinaryField128;

/// Trace heights under test, in log rows.
const LOG_ROWS: [usize; 3] = [16, 18, 20];

/// Columns a bit-sliced fixture keeps, out of the lanes a block holds.
const NUM_BIT_COLUMNS: usize = 40;

/// Pack a run of words through the field's own basis accessor, one element at a time.
///
/// This is the obvious loop: assemble each element from the coefficients it absorbs.
fn pack_words_per_cell(cells: &[BinaryField32]) -> Vec<EF> {
    cells
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| EF::from_basis_coefficients_fn(|index| chunk[index]))
        .collect()
}

/// Exchange rows and columns of a bit-sliced block one lane at a time.
///
/// `d^2` bit reads per block, against the `log2(d)` word passes the square transpose takes.
fn transpose_per_bit<U: Underlier>(
    rows: &[PackedGf2<U>],
    num_columns: usize,
) -> Vec<Vec<PackedGf2<U>>> {
    let width = PackedGf2::<U>::WIDTH;
    (0..num_columns)
        .map(|column| {
            rows.chunks_exact(width)
                .map(|block| PackedGf2::<U>::from_fn(|lane| block[lane].get(column)))
                .collect()
        })
        .collect()
}

/// One 32-bit column packed into the elements holding it.
fn bench_pack_words(c: &mut Criterion) {
    let mut group = c.benchmark_group("pack_words");
    group.sample_size(20);
    let mut rng = SmallRng::seed_from_u64(0x32C0);
    for &log_rows in &LOG_ROWS {
        let cells: Vec<BinaryField32> = (0..1 << log_rows).map(|_| rng.random()).collect();

        group.bench_with_input(BenchmarkId::new("packed", log_rows), &cells, |b, cells| {
            b.iter(|| pack::<BinaryField32, EF>(cells));
        });
        group.bench_with_input(
            BenchmarkId::new("per_cell", log_rows),
            &cells,
            |b, cells| {
                b.iter(|| pack_words_per_cell(cells));
            },
        );
    }
    group.finish();
}

/// A row-major bit-sliced trace turned column-major and packed.
fn bench_pack_bit_rows(c: &mut Criterion) {
    let mut group = c.benchmark_group("pack_bit_rows");
    group.sample_size(20);
    let mut rng = SmallRng::seed_from_u64(0xB175);
    for &log_rows in &LOG_ROWS {
        let rows: Vec<PackedGf2x64> = (0..1 << log_rows)
            .map(|_| PackedGf2x64::new(rng.random::<u64>()))
            .collect();

        group.bench_with_input(BenchmarkId::new("packed", log_rows), &rows, |b, rows| {
            b.iter(|| PackedStack::<PackedGf2x64, EF>::from_bit_rows(rows, NUM_BIT_COLUMNS));
        });
        group.bench_with_input(BenchmarkId::new("per_bit", log_rows), &rows, |b, rows| {
            b.iter(|| {
                let columns = transpose_per_bit(rows, NUM_BIT_COLUMNS);
                let views: Vec<&[PackedGf2x64]> = columns.iter().map(Vec::as_slice).collect();
                PackedStack::<PackedGf2x64, EF>::from_columns(&views)
            });
        });
    }
    group.finish();
}

/// The same bit trace committed one element per bit, which is what no packing costs.
///
/// Both arms cover the same columns, so the times and the element counts compare directly.
fn bench_embedding_cost(c: &mut Criterion) {
    let mut group = c.benchmark_group("embed_bits");
    group.sample_size(20);
    let mut rng = SmallRng::seed_from_u64(0xE3B1);
    for &log_rows in &LOG_ROWS {
        let rows: Vec<PackedGf2x64> = (0..1 << log_rows)
            .map(|_| PackedGf2x64::new(rng.random::<u64>()))
            .collect();

        // Bits of the kept columns, against the elements each layout writes for them.
        let bits = (1usize << log_rows) * NUM_BIT_COLUMNS;
        eprintln!(
            "packing/elements/{log_rows}: packed {} vs embedded {bits}",
            bits / EF::COORDINATES,
        );

        group.bench_with_input(
            BenchmarkId::new("one_per_bit", log_rows),
            &rows,
            |b, rows| {
                b.iter(|| {
                    rows.iter()
                        .flat_map(|block| {
                            (0..NUM_BIT_COLUMNS).map(move |lane| {
                                if block.get(lane) == Gf2::ONE {
                                    EF::ONE
                                } else {
                                    EF::ZERO
                                }
                            })
                        })
                        .collect::<Vec<EF>>()
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_pack_words,
    bench_pack_bit_rows,
    bench_embedding_cost
);
criterion_main!(benches);
