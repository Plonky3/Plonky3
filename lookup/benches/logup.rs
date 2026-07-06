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

criterion_group!(benches, bench_generate_permutation);
criterion_main!(benches);
