use std::sync::Arc;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_air::symbolic::{AirLayout, SymbolicAirBuilder, SymbolicExpression};
use p3_air::{AirBuilder, WindowAccess};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_fri::TwoAdicFriPcs;
use p3_lookup::LookupProtocol;
use p3_lookup::logup::LogUpGadget;
use p3_lookup::traits::{Kind, Lookup};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark::StarkConfig;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;
type Perm = Poseidon2BabyBear<16>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type ValMmcs =
    MerkleTreeMmcs<<F as Field>::Packing, <F as Field>::Packing, MyHash, MyCompress, 2, 8>;
type ChallengeMmcs = ExtensionMmcs<F, EF, ValMmcs>;
type Challenger = DuplexChallenger<F, Perm, 16, 8>;
type Dft = Radix2DitParallel<F>;
type MyPcs = TwoAdicFriPcs<F, Dft, ValMmcs, ChallengeMmcs>;
type SC = StarkConfig<MyPcs, EF, Challenger>;

/// Per-lookup column stride: `tuple_size` read keys, `tuple_size` provide keys, one selector.
const fn cols_per_lookup(tuple_size: usize) -> usize {
    2 * tuple_size + 1
}

/// Build a main trace with random keys and a sparsity-controlled selector per lookup.
///
/// - Every column is random by default.
/// - Each lookup's selector column is `1` once every `active_period` rows, else `0`.
/// - `active_period == 1` makes every row active (a dense workload).
fn build_trace(config: &BenchConfig, rng: &mut SmallRng) -> RowMajorMatrix<F> {
    let height = 1 << config.log_height;
    let width = config.trace_width;
    let stride = cols_per_lookup(config.tuple_size);

    let mut values = F::zero_vec(height * width);
    for row in 0..height {
        let base = row * width;
        // Random keys fill the whole row first.
        for v in &mut values[base..base + width] {
            *v = F::from_u32(rng.random::<u32>() % (1 << 27));
        }
        // A row is active once every `active_period` rows.
        let active = row % config.active_period == 0;
        // Overwrite each lookup's selector column with the sparse flag.
        for lookup_idx in 0..config.num_lookups {
            let selector_col = lookup_idx * stride + 2 * config.tuple_size;
            values[base + selector_col] = if active { F::ONE } else { F::ZERO };
        }
    }
    RowMajorMatrix::new(values, width)
}

/// Build symbolic lookup contexts, both sides gated by one shared selector column.
///
/// A zero selector zeroes both multiplicities, so the row contributes nothing.
/// This mirrors a selector-gated lookup (active only on some rows).
fn build_lookups(num_lookups: usize, tuple_size: usize, trace_width: usize) -> Vec<Lookup<F>> {
    let symbolic_builder = SymbolicAirBuilder::<F>::new(AirLayout {
        main_width: trace_width,
        ..Default::default()
    });
    let symbolic_main = symbolic_builder.main();
    let symbolic_main_local = symbolic_main.current_slice();

    let stride = cols_per_lookup(tuple_size);

    (0..num_lookups)
        .map(|lookup_idx| {
            let base_col = lookup_idx * stride;

            let read_elements: Vec<SymbolicExpression<F>> = (0..tuple_size)
                .map(|j| symbolic_main_local[base_col + j].into())
                .collect();

            let provide_elements: Vec<SymbolicExpression<F>> = (0..tuple_size)
                .map(|j| symbolic_main_local[base_col + tuple_size + j].into())
                .collect();

            // One selector column gates both sides: read sends `+sel`, provide sends `-sel`.
            let selector: SymbolicExpression<F> =
                symbolic_main_local[base_col + 2 * tuple_size].into();

            let elements = vec![read_elements, provide_elements];
            let multiplicities = vec![
                selector.clone(),
                SymbolicExpression::Neg {
                    x: Arc::new(selector),
                    degree_multiple: 1,
                },
            ];

            Lookup {
                kind: Kind::Local,
                elements,
                multiplicities,
                count_weight: 0,
                column: lookup_idx,
                flags: None,
            }
        })
        .collect()
}

/// Generate random challenges for the given number of lookups.
fn random_challenges(num_lookups: usize, rng: &mut SmallRng) -> Vec<EF> {
    (0..num_lookups * 2)
        .map(|_| {
            EF::new([
                F::from_u32(rng.random::<u32>() % (1 << 27)),
                F::from_u32(rng.random::<u32>() % (1 << 27)),
                F::from_u32(rng.random::<u32>() % (1 << 27)),
                F::from_u32(rng.random::<u32>() % (1 << 27)),
            ])
        })
        .collect()
}

struct BenchConfig {
    name: &'static str,
    log_height: usize,
    num_lookups: usize,
    tuple_size: usize,
    trace_width: usize,
    /// One active row every `active_period` rows; `1` is fully dense.
    active_period: usize,
}

/// Terse constructor; `trace_width` must equal `num_lookups * cols_per_lookup(tuple_size)`.
const fn cfg(
    name: &'static str,
    log_height: usize,
    num_lookups: usize,
    tuple_size: usize,
    active_period: usize,
) -> BenchConfig {
    BenchConfig {
        name,
        log_height,
        num_lookups,
        tuple_size,
        trace_width: num_lookups * cols_per_lookup(tuple_size),
        active_period,
    }
}

const CONFIGS: &[BenchConfig] = &[
    // Dense baselines: every row active, so flag-zero skip never fires.
    cfg("dense_small", 10, 1, 1, 1),
    cfg("dense_medium", 16, 2, 2, 1),
    cfg("dense_large", 20, 4, 1, 1),
    // Wide selector-gated payload, swept from dense to very sparse.
    // The flag-zero skip should make the sparse rows cheaper.
    cfg("gated_tw4_p1", 18, 2, 4, 1),
    cfg("gated_tw4_p8", 18, 2, 4, 8),
    cfg("gated_tw4_p64", 18, 2, 4, 64),
];

fn bench_generate_permutation(c: &mut Criterion) {
    let mut group = c.benchmark_group("logup_generate_permutation");
    group.sample_size(10);

    let gadget = LogUpGadget::new();

    for config in CONFIGS {
        let mut rng = SmallRng::seed_from_u64(42);

        let main_trace = build_trace(config, &mut rng);
        let lookups = build_lookups(config.num_lookups, config.tuple_size, config.trace_width);
        let challenges = random_challenges(config.num_lookups, &mut rng);

        group.bench_function(
            BenchmarkId::new(
                config.name,
                format!(
                    "h=2^{}_lookups={}_tuple={}_period={}",
                    config.log_height, config.num_lookups, config.tuple_size, config.active_period
                ),
            ),
            |b| {
                b.iter(|| {
                    gadget.generate_permutation::<SC>(
                        &main_trace,
                        &None,
                        &[],
                        &lookups,
                        &challenges,
                    );
                });
            },
        );
    }

    group.finish();
}

/// Trace where exactly one of `num_branches` flags is `1` per row.
///
/// Layout per row is `[flag_0 .. flag_{M-1}, key_0 .. key_{M-1}]`, width `2M`.
/// The active branch rotates round-robin, so the workload is fully dense.
fn build_exclusive_trace(
    num_branches: usize,
    log_height: usize,
    rng: &mut SmallRng,
) -> RowMajorMatrix<F> {
    let height = 1 << log_height;
    let width = 2 * num_branches;

    let mut values = F::zero_vec(height * width);
    for row in 0..height {
        let base = row * width;
        // Random keys fill the key half of the row.
        for k in 0..num_branches {
            values[base + num_branches + k] = F::from_u32(rng.random::<u32>() % (1 << 27));
        }
        // Exactly one flag fires, rotating across rows.
        values[base + row % num_branches] = F::ONE;
    }
    RowMajorMatrix::new(values, width)
}

/// One exclusive column carrying `num_branches` mutually-exclusive width-1 queries.
fn build_exclusive_lookup(num_branches: usize) -> Lookup<F> {
    let width = 2 * num_branches;
    let sb = SymbolicAirBuilder::<F>::new(AirLayout {
        main_width: width,
        ..Default::default()
    });
    let main = sb.main();
    let local = main.current_slice();

    // Branch k reads flag column k and key column M + k.
    let flags = (0..num_branches).map(|k| local[k].into()).collect();
    let elements = (0..num_branches)
        .map(|k| vec![local[num_branches + k].into()])
        .collect();
    let multiplicities = (0..num_branches)
        .map(|_| SymbolicExpression::from(F::ONE))
        .collect();

    Lookup {
        kind: Kind::Local,
        elements,
        multiplicities,
        count_weight: 1,
        column: 0,
        flags: Some(flags),
    }
}

/// The no-exclusivity baseline: one gated single-query column per branch.
///
/// Each column is a unit query whose multiplicity is its own selector flag.
/// This is how the same queries must be expressed without exclusivity support.
fn build_additive_baseline(num_branches: usize) -> Vec<Lookup<F>> {
    let width = 2 * num_branches;
    let sb = SymbolicAirBuilder::<F>::new(AirLayout {
        main_width: width,
        ..Default::default()
    });
    let main = sb.main();
    let local = main.current_slice();

    (0..num_branches)
        .map(|k| {
            // Multiplicity is the flag, so an inactive row contributes nothing.
            let flag: SymbolicExpression<F> = local[k].into();
            let key: SymbolicExpression<F> = local[num_branches + k].into();
            Lookup {
                kind: Kind::Local,
                elements: vec![vec![key]],
                multiplicities: vec![flag],
                count_weight: 1,
                column: k,
                flags: None,
            }
        })
        .collect()
}

/// Compare one exclusive column against the equivalent stack of additive columns.
///
/// Both express the same `M` mutually-exclusive queries over one trace.
/// The exclusive form needs one auxiliary column; the baseline needs `M`.
fn bench_exclusive_vs_additive(c: &mut Criterion) {
    let mut group = c.benchmark_group("logup_exclusive_vs_additive");
    group.sample_size(10);

    let gadget = LogUpGadget::new();
    let log_height = 16;

    for &branches in &[8usize, 16, 32] {
        let mut rng = SmallRng::seed_from_u64(7);
        let trace = build_exclusive_trace(branches, log_height, &mut rng);

        // Exclusive: a single column, so a single challenge pair.
        let exclusive = vec![build_exclusive_lookup(branches)];
        let exclusive_challenges = random_challenges(1, &mut rng);

        // Baseline: M columns, so M challenge pairs.
        let additive = build_additive_baseline(branches);
        let additive_challenges = random_challenges(branches, &mut rng);

        group.bench_function(
            BenchmarkId::new(
                "exclusive_1col",
                format!("branches={branches}_h=2^{log_height}"),
            ),
            |b| {
                b.iter(|| {
                    gadget.generate_permutation::<SC>(
                        &trace,
                        &None,
                        &[],
                        &exclusive,
                        &exclusive_challenges,
                    );
                });
            },
        );
        group.bench_function(
            BenchmarkId::new(
                "additive_Ncol",
                format!("branches={branches}_h=2^{log_height}"),
            ),
            |b| {
                b.iter(|| {
                    gadget.generate_permutation::<SC>(
                        &trace,
                        &None,
                        &[],
                        &additive,
                        &additive_challenges,
                    );
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_generate_permutation,
    bench_exclusive_vs_additive
);
criterion_main!(benches);
