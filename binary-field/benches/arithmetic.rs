//! Tower arithmetic, comparing the recursive reference routines against whatever `Mul` and
//! `square` dispatch to on the host.
//!
//! The two are the same routine below `GF(2^64)`; at 64 and 128 bits both operators take the
//! carryless-multiply fast path where the target has the instruction for it. Rerunning with
//! `RUSTFLAGS="-C target-feature=-aes"` (AArch64) or without `+pclmulqdq` (x86-64) measures the
//! same code with the fast path turned off.

use std::hint::black_box;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use p3_binary_field::{
    BinaryField8, BinaryField16, BinaryField32, BinaryField64, BinaryField128, Ghash128,
};
use p3_field::{BasedVectorSpace, Field, PackedValue, PrimeCharacteristicRing};
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

/// Multiplications performed by a single iteration of every benchmark here.
const REPS: usize = 1000;

/// Independent chains a throughput benchmark interleaves.
const LANES: usize = 10;

/// Latency and throughput of one level, for both multiplication routines.
///
/// The latency benchmark folds a dependent chain, as `p3_field_testing::bench_func` does, so
/// each product waits on the previous one. The throughput benchmark runs [`LANES`] such chains
/// side by side over the same number of products, leaving the CPU free to overlap them.
macro_rules! bench_level {
    ($c:expr, $t:ty, $bits:literal) => {{
        let mut rng = SmallRng::seed_from_u64(1);
        let operands: Vec<$t> = (0..REPS).map(|_| rng.random::<$t>()).collect();

        let mut group = $c.benchmark_group(concat!("mul/", $bits));

        group.bench_function("reference/latency", |b| {
            b.iter(|| {
                black_box(&operands)
                    .iter()
                    .fold(<$t>::ONE, |acc, &y| acc.reference_mul(y))
            });
        });
        group.bench_function("dispatched/latency", |b| {
            b.iter(|| {
                black_box(&operands)
                    .iter()
                    .fold(<$t>::ONE, |acc, &y| acc * y)
            });
        });

        group.bench_function("reference/throughput", |b| {
            b.iter(|| {
                let mut acc = [<$t>::ONE; LANES];
                for chunk in black_box(&operands).chunks_exact(LANES) {
                    for (lane, &y) in acc.iter_mut().zip(chunk) {
                        *lane = lane.reference_mul(y);
                    }
                }
                acc
            });
        });
        group.bench_function("dispatched/throughput", |b| {
            b.iter(|| {
                let mut acc = [<$t>::ONE; LANES];
                for chunk in black_box(&operands).chunks_exact(LANES) {
                    for (lane, &y) in acc.iter_mut().zip(chunk) {
                        *lane *= y;
                    }
                }
                acc
            });
        });

        group.finish();
    }};
}

/// Latency and throughput of squaring one level, against the multiplication that computes the
/// same value.
///
/// Each step mixes the next operand into the accumulator before squaring, so the chain cannot
/// be folded away at compile time and both benchmarks pay the same `XOR`.
macro_rules! bench_square_level {
    ($c:expr, $t:ty, $bits:literal) => {{
        let mut rng = SmallRng::seed_from_u64(1);
        let operands: Vec<$t> = (0..REPS).map(|_| rng.random::<$t>()).collect();

        let mut group = $c.benchmark_group(concat!("square/", $bits));

        group.bench_function("square/latency", |b| {
            b.iter(|| {
                black_box(&operands)
                    .iter()
                    .fold(<$t>::ONE, |acc, &y| (acc + y).square())
            });
        });
        group.bench_function("mul/latency", |b| {
            b.iter(|| {
                black_box(&operands).iter().fold(<$t>::ONE, |acc, &y| {
                    let x = acc + y;
                    x * x
                })
            });
        });

        group.bench_function("square/throughput", |b| {
            b.iter(|| {
                let mut acc = [<$t>::ONE; LANES];
                for chunk in black_box(&operands).chunks_exact(LANES) {
                    for (lane, &y) in acc.iter_mut().zip(chunk) {
                        *lane = (*lane + y).square();
                    }
                }
                acc
            });
        });
        group.bench_function("mul/throughput", |b| {
            b.iter(|| {
                let mut acc = [<$t>::ONE; LANES];
                for chunk in black_box(&operands).chunks_exact(LANES) {
                    for (lane, &y) in acc.iter_mut().zip(chunk) {
                        let x = *lane + y;
                        *lane = x * x;
                    }
                }
                acc
            });
        });

        group.finish();
    }};
}

fn bench_mul(c: &mut Criterion) {
    bench_level!(c, BinaryField8, 8);
    bench_level!(c, BinaryField16, 16);
    bench_level!(c, BinaryField32, 32);
    bench_level!(c, BinaryField64, 64);
    bench_level!(c, BinaryField128, 128);
}

fn bench_square(c: &mut Criterion) {
    bench_square_level!(c, BinaryField8, 8);
    bench_square_level!(c, BinaryField16, 16);
    bench_square_level!(c, BinaryField32, 32);
    bench_square_level!(c, BinaryField64, 64);
    bench_square_level!(c, BinaryField128, 128);
}

fn bench_inverse(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(1);
    // Inversion recurses through the level below's `Mul`, so it picks up the fast path too.
    let operands: Vec<BinaryField128> = (0..REPS)
        .map(|_| {
            let x = rng.random::<BinaryField128>();
            if x.is_zero() { BinaryField128::ONE } else { x }
        })
        .collect();

    let mut group = c.benchmark_group("inverse/128");
    group.bench_function("dispatched", |b| {
        b.iter(|| {
            black_box(&operands)
                .iter()
                // Addition is `XOR`, so the fold barely adds to the inversions it accumulates.
                .fold(BinaryField128::ZERO, |acc, &y| acc + y.inverse())
        });
    });
    group.finish();
}

fn bench_mul_alpha(c: &mut Criterion) {
    // `α = X_5`, the generator of `GF(2^64)` over `GF(2^32)`, sits at bit 32 of the tower
    // representation. `TowerLevel::mul_alpha` is crate-internal, so this measures the product
    // by that element through the public operator instead.
    let mut alpha_bytes = [0u8; 16];
    alpha_bytes[4] = 1;
    let alpha = BinaryField128::from_le_bytes(alpha_bytes);

    let mut rng = SmallRng::seed_from_u64(1);
    let operands: Vec<BinaryField128> = (0..REPS).map(|_| rng.random()).collect();

    let mut group = c.benchmark_group("mul_alpha/128");
    group.bench_function("reference", |b| {
        b.iter(|| {
            black_box(&operands)
                .iter()
                .fold(BinaryField128::ZERO, |acc, &y| acc + y.reference_mul(alpha))
        });
    });
    group.bench_function("dispatched", |b| {
        b.iter(|| {
            black_box(&operands)
                .iter()
                .fold(BinaryField128::ZERO, |acc, &y| acc + y * alpha)
        });
    });
    group.finish();
}

/// Elements flattened by one iteration of [`bench_flatten_to_base`].
const FLATTEN_LEN: usize = 1 << 20;

/// The tower's `flatten_to_base` against `p3-field`'s default, which the tower overrides.
///
/// The default is reproduced here rather than called, since the override shadows it: it takes
/// `as_basis_coefficients_slice().to_vec()` per element, so it heap-allocates once per field
/// element on a buffer the override allocates once for.
fn bench_flatten_to_base(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(1);
    let elems: Vec<BinaryField128> = (0..FLATTEN_LEN).map(|_| rng.random()).collect();

    let mut group = c.benchmark_group("flatten_to_base/BinaryField128");

    group.bench_function("p3-field default", |b| {
        b.iter_batched(
            || elems.clone(),
            |v| {
                let out: Vec<BinaryField8> = v
                    .into_iter()
                    .flat_map(|x| {
                        BasedVectorSpace::<BinaryField8>::as_basis_coefficients_slice(&x).to_vec()
                    })
                    .collect();
                black_box(out)
            },
            BatchSize::PerIteration,
        );
    });

    group.bench_function("tower override", |b| {
        b.iter_batched(
            || elems.clone(),
            |v| black_box(<BinaryField128 as BasedVectorSpace<BinaryField8>>::flatten_to_base(v)),
            BatchSize::PerIteration,
        );
    });

    group.finish();
}

/// Random operands in both representations of `GF(2^128)`, lane for lane.
fn representation_operands() -> (Vec<BinaryField128>, Vec<Ghash128>) {
    let mut rng = SmallRng::seed_from_u64(1);
    let tower: Vec<BinaryField128> = (0..REPS).map(|_| rng.random()).collect();
    let ghash = tower.iter().map(|&x| Ghash128::from(x)).collect();
    (tower, ghash)
}

/// Multiplication in the two representations, on a dependent chain.
///
/// Both are the same field.
///
/// A tower product converts both operands and the result, sixteen lookups apiece.
/// A polynomial-basis product is already in the basis the instruction wants.
fn bench_representation_mul_latency(c: &mut Criterion) {
    let (tower, ghash) = representation_operands();

    let mut group = c.benchmark_group("representations/mul/latency");
    group.bench_function("tower", |b| {
        b.iter(|| {
            black_box(&tower)
                .iter()
                .fold(BinaryField128::ONE, |acc, &y| acc * y)
        });
    });
    group.bench_function("ghash", |b| {
        b.iter(|| {
            black_box(&ghash)
                .iter()
                .fold(Ghash128::ONE, |acc, &y| acc * y)
        });
    });
    group.finish();
}

/// Multiplication in the two representations, on independent chains the core can overlap.
fn bench_representation_mul_throughput(c: &mut Criterion) {
    let (tower, ghash) = representation_operands();

    let mut group = c.benchmark_group("representations/mul/throughput");
    group.bench_function("tower", |b| {
        b.iter(|| {
            let mut acc = [BinaryField128::ONE; LANES];
            let (chunks, _) = black_box(&tower).as_chunks::<LANES>();
            for chunk in chunks {
                for (lane, &y) in acc.iter_mut().zip(chunk) {
                    *lane *= y;
                }
            }
            acc
        });
    });
    group.bench_function("ghash", |b| {
        b.iter(|| {
            let mut acc = [Ghash128::ONE; LANES];
            let (chunks, _) = black_box(&ghash).as_chunks::<LANES>();
            for chunk in chunks {
                for (lane, &y) in acc.iter_mut().zip(chunk) {
                    *lane *= y;
                }
            }
            acc
        });
    });
    group.finish();
}

/// Squaring in the two representations.
///
/// Each step mixes in the next operand, so the chain cannot fold away at compile time.
fn bench_representation_square(c: &mut Criterion) {
    let (tower, ghash) = representation_operands();

    let mut group = c.benchmark_group("representations/square");
    group.bench_function("tower", |b| {
        b.iter(|| {
            black_box(&tower)
                .iter()
                .fold(BinaryField128::ONE, |acc, &y| (acc + y).square())
        });
    });
    group.bench_function("ghash", |b| {
        b.iter(|| {
            black_box(&ghash)
                .iter()
                .fold(Ghash128::ONE, |acc, &y| (acc + y).square())
        });
    });
    // The dedicated squaring drops the two cross products a general multiply pays for.
    group.bench_function("ghash, through multiplication", |b| {
        b.iter(|| {
            black_box(&ghash).iter().fold(Ghash128::ONE, |acc, &y| {
                let x = acc + y;
                x * x
            })
        });
    });
    group.finish();
}

/// Inversion in the polynomial basis, against converting to the tower and inverting there.
///
/// Both arms see the same operands, none of them zero, so neither can panic.
fn bench_representation_inverse(c: &mut Criterion) {
    let (_, ghash) = representation_operands();
    let nonzero: Vec<Ghash128> = ghash
        .iter()
        .map(|&x| if x.is_zero() { Ghash128::ONE } else { x })
        .collect();

    let mut group = c.benchmark_group("representations/inverse");
    group.bench_function("through tower", |b| {
        // Include both basis conversions in the comparison.
        b.iter(|| {
            black_box(&nonzero).iter().fold(Ghash128::ZERO, |acc, &y| {
                acc + Ghash128::from(BinaryField128::from(y).inverse())
            })
        });
    });

    group.bench_function("ghash", |b| {
        b.iter(|| {
            black_box(&nonzero)
                .iter()
                .fold(Ghash128::ZERO, |acc, &y| acc + y.inverse())
        });
    });
    group.finish();
}

/// What the change of basis costs on its own, in both directions.
fn bench_representation_convert(c: &mut Criterion) {
    let (tower, ghash) = representation_operands();

    let mut group = c.benchmark_group("representations/convert");
    group.bench_function("tower to ghash", |b| {
        b.iter(|| {
            black_box(&tower)
                .iter()
                .fold(Ghash128::ZERO, |acc, &y| acc + Ghash128::from(y))
        });
    });
    group.bench_function("ghash to tower", |b| {
        b.iter(|| {
            black_box(&ghash)
                .iter()
                .fold(BinaryField128::ZERO, |acc, &y| {
                    acc + BinaryField128::from(y)
                })
        });
    });
    group.finish();
}

/// The packing of the polynomial-basis field against its scalar.
///
/// One element fills one 128-bit lane.
/// A packed product is the scalar kernel applied to every lane at once.
/// Both arms multiply the same number of field elements.
fn bench_packing(c: &mut Criterion) {
    type Packing = <Ghash128 as Field>::Packing;

    let width = Packing::WIDTH;
    let mut rng = SmallRng::seed_from_u64(1);

    // A whole number of packed vectors, so neither arm has a remainder to handle.
    let scalars: Vec<Ghash128> = (0..REPS.next_multiple_of(width))
        .map(|_| rng.random())
        .collect();
    let packed: Vec<Packing> = Packing::pack_slice(&scalars).to_vec();

    let mut group = c.benchmark_group("packing/mul/throughput");
    group.throughput(criterion::Throughput::Elements(scalars.len() as u64));

    group.bench_function("scalar", |b| {
        b.iter(|| {
            let mut acc = [Ghash128::ONE; LANES];
            let (chunks, _) = black_box(&scalars).as_chunks::<LANES>();
            for chunk in chunks {
                for (lane, &y) in acc.iter_mut().zip(chunk) {
                    *lane *= y;
                }
            }
            acc
        });
    });
    group.bench_function("packed", |b| {
        b.iter(|| {
            let mut acc = [Packing::ONE; LANES];
            let (chunks, _) = black_box(&packed).as_chunks::<LANES>();
            for chunk in chunks {
                for (lane, &y) in acc.iter_mut().zip(chunk) {
                    *lane *= y;
                }
            }
            acc
        });
    });
    group.finish();
}

/// Compare direct square roots with the repeated-squaring definition.
fn bench_ghash_sqrt(c: &mut Criterion) {
    let (_, operands) = representation_operands();
    let mut group = c.benchmark_group("ghash/sqrt");
    for direct in [false, true] {
        group.bench_function(if direct { "direct" } else { "127 squares" }, |b| {
            // Mix fresh data into the chain to keep every square root observable.
            b.iter(|| {
                black_box(&operands).iter().fold(Ghash128::ONE, |acc, &x| {
                    let value = acc + x;
                    if direct {
                        value.try_sqrt().unwrap()
                    } else {
                        value.exp_power_of_2(127)
                    }
                })
            });
        });
    }
    group.finish();
}

/// Compare deferred reduction with a sum of separately reduced products.
fn bench_dot_width<P: PackedValue + PrimeCharacteristicRing + Copy, const N: usize>(
    c: &mut Criterion,
    label: &str,
) where
    StandardUniform: Distribution<P>,
{
    let mut rng = SmallRng::seed_from_u64(1);
    // Both algorithms consume identical full-width random inputs.
    let a: [P; N] = core::array::from_fn(|_| rng.random());
    let b: [P; N] = core::array::from_fn(|_| rng.random());
    let mut group = c.benchmark_group(format!("ghash/dot/{label}/{N}"));
    group.bench_function("separate", |bencher| {
        bencher.iter(|| {
            black_box(&a)
                .iter()
                .zip(black_box(&b))
                .map(|(&x, &y)| x * y)
                .sum::<P>()
        });
    });
    group.bench_function("deferred", |bencher| {
        bencher.iter(|| P::dot_product(black_box(&a), black_box(&b)));
    });
    group.finish();
}

/// Exercise small and long dot products at scalar and native packed widths.
fn bench_ghash_dot(c: &mut Criterion) {
    bench_dot_width::<Ghash128, 4>(c, "scalar");
    bench_dot_width::<Ghash128, 16>(c, "scalar");
    bench_dot_width::<Ghash128, 64>(c, "scalar");
    bench_dot_width::<<Ghash128 as Field>::Packing, 4>(c, "packed");
    bench_dot_width::<<Ghash128 as Field>::Packing, 16>(c, "packed");
    bench_dot_width::<<Ghash128 as Field>::Packing, 64>(c, "packed");
}

criterion_group!(
    benches,
    bench_mul,
    bench_square,
    bench_inverse,
    bench_mul_alpha,
    bench_flatten_to_base,
    bench_representation_mul_latency,
    bench_representation_mul_throughput,
    bench_representation_square,
    bench_representation_inverse,
    bench_representation_convert,
    bench_packing,
    bench_ghash_sqrt,
    bench_ghash_dot
);
criterion_main!(benches);
