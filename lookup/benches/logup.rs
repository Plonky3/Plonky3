use std::sync::Arc;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_air::AirBuilder;
use p3_air::symbolic::{SymbolicAirBuilder, SymbolicExpression};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_fri::TwoAdicFriPcs;
use p3_lookup::logup::LogUpGadget;
use p3_lookup::lookup_traits::{Direction, Kind, Lookup, LookupData, LookupGadget};
use p3_matrix::Matrix;
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
type ValMmcs = MerkleTreeMmcs<<F as Field>::Packing, <F as Field>::Packing, MyHash, MyCompress, 8>;
type ChallengeMmcs = ExtensionMmcs<F, EF, ValMmcs>;
type Challenger = DuplexChallenger<F, Perm, 16, 8>;
type Dft = Radix2DitParallel<F>;
type MyPcs = TwoAdicFriPcs<F, Dft, ValMmcs, ChallengeMmcs>;
type SC = StarkConfig<MyPcs, EF, Challenger>;

/// Build a random main trace of given height and width.
fn random_main_trace(height: usize, width: usize, rng: &mut SmallRng) -> RowMajorMatrix<F> {
    let values: Vec<F> = (0..height * width)
        .map(|_| F::from_u32(rng.random::<u32>() % (1 << 27)))
        .collect();
    RowMajorMatrix::new(values, width)
}

/// Build symbolic lookup contexts similar to RangeCheckAir pattern.
fn build_lookups(num_lookups: usize, tuple_size: usize, trace_width: usize) -> Vec<Lookup<F>> {
    let symbolic_builder = SymbolicAirBuilder::<F>::new(0, trace_width, 0, 0, 0);
    let symbolic_main = symbolic_builder.main();
    let symbolic_main_local = symbolic_main.row_slice(0).unwrap();

    let cols_per_lookup = 2 * tuple_size + 1;

    (0..num_lookups)
        .map(|lookup_idx| {
            let base_col = lookup_idx * cols_per_lookup;

            let read_elements: Vec<SymbolicExpression<F>> = (0..tuple_size)
                .map(|j| symbolic_main_local[base_col + j].into())
                .collect();
            let read_mult = SymbolicExpression::Constant(F::ONE);

            let provide_elements: Vec<SymbolicExpression<F>> = (0..tuple_size)
                .map(|j| symbolic_main_local[base_col + tuple_size + j].into())
                .collect();
            let provide_mult: SymbolicExpression<F> =
                symbolic_main_local[base_col + 2 * tuple_size].into();

            let lookup_inputs = [
                (read_elements, read_mult, Direction::Receive),
                (provide_elements, provide_mult, Direction::Send),
            ];

            let element_exprs: Vec<Vec<SymbolicExpression<F>>> = lookup_inputs
                .iter()
                .map(|(elts, _, _): &(Vec<_>, _, _)| elts.clone())
                .collect();
            let multiplicities_exprs: Vec<SymbolicExpression<F>> = lookup_inputs
                .iter()
                .map(|(_, mult, dir): &(_, SymbolicExpression<F>, Direction)| {
                    let m = mult.clone();
                    match dir {
                        Direction::Send => SymbolicExpression::Neg {
                            x: Arc::new(m),
                            degree_multiple: 1,
                        },
                        Direction::Receive => m,
                    }
                })
                .collect();

            Lookup {
                kind: Kind::Local,
                element_exprs,
                multiplicities_exprs,
                columns: vec![lookup_idx],
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
}

const CONFIGS: &[BenchConfig] = &[
    BenchConfig {
        name: "small",
        log_height: 10,
        num_lookups: 1,
        tuple_size: 1,
        trace_width: 3,
    },
    BenchConfig {
        name: "medium",
        log_height: 16,
        num_lookups: 2,
        tuple_size: 2,
        trace_width: 10,
    },
    BenchConfig {
        name: "large",
        log_height: 20,
        num_lookups: 4,
        tuple_size: 1,
        trace_width: 12,
    },
];

fn bench_generate_permutation(c: &mut Criterion) {
    let mut group = c.benchmark_group("logup_generate_permutation");
    group.sample_size(10);

    let gadget = LogUpGadget::new();

    for config in CONFIGS {
        let mut rng = SmallRng::seed_from_u64(42);
        let height = 1 << config.log_height;

        let main_trace = random_main_trace(height, config.trace_width, &mut rng);
        let lookups = build_lookups(config.num_lookups, config.tuple_size, config.trace_width);
        let challenges = random_challenges(config.num_lookups, &mut rng);

        group.bench_function(
            BenchmarkId::new(
                config.name,
                format!(
                    "h=2^{}_lookups={}_tuple={}",
                    config.log_height, config.num_lookups, config.tuple_size
                ),
            ),
            |b| {
                b.iter(|| {
                    let mut lookup_data: Vec<LookupData<EF>> = Vec::new();
                    gadget.generate_permutation::<SC>(
                        &main_trace,
                        &None,
                        &[],
                        &lookups,
                        &mut lookup_data,
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
