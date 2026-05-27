use core::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_air::WindowAccess;
use p3_air::symbolic::AirLayout;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark::{StarkConfig, get_log_num_quotient_chunks, prove};
use rand::SeedableRng;
use rand::rngs::SmallRng;

type Val = BabyBear;
type Challenge = BinomialExtensionField<Val, 4>;
type Perm = Poseidon2BabyBear<16>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type ValMmcs =
    MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 2, 8>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
type Dft = Radix2DitParallel<Val>;
type MyPcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
type MyConfig = StarkConfig<MyPcs, Challenge, Challenger>;

const TRACE_LOG_DEGREES: [usize; 2] = [14, 16];
const PERIODIC_CASES: [(&str, &[usize]); 3] = [
    ("period2_x5", &[2, 2, 2, 2, 2]),
    ("period4_x6", &[4, 4, 4, 4, 4, 4]),
    ("period8_x9", &[8, 8, 8, 8, 8, 8, 8, 8, 8]),
];

#[derive(Clone, Debug)]
struct PeriodicProductAir {
    periods: Vec<usize>,
}

impl<F> BaseAir<F> for PeriodicProductAir
where
    F: PrimeCharacteristicRing,
{
    fn width(&self) -> usize {
        1
    }

    fn num_periodic_columns(&self) -> usize {
        self.periods.len()
    }

    fn periodic_columns(&self) -> Vec<Vec<F>> {
        self.periods
            .iter()
            .enumerate()
            .map(|(column_index, period)| {
                (0..*period)
                    .map(|row| F::from_usize(column_index + row + 1))
                    .collect()
            })
            .collect()
    }
}

impl<AB: AirBuilder> Air<AB> for PeriodicProductAir
where
    AB::F: Field,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice()[0];
        let product = builder
            .periodic_values()
            .iter()
            .copied()
            .map(Into::<AB::Expr>::into)
            .reduce(|acc, value| acc * value)
            .expect("benchmark AIR must define at least one periodic column");

        builder.assert_zero(local - product);
    }
}

fn air_layout(air: &PeriodicProductAir) -> AirLayout {
    AirLayout {
        main_width: 1,
        num_periodic_columns: air.periods.len(),
        ..Default::default()
    }
}

fn make_trace(log_degree: usize) -> RowMajorMatrix<Val> {
    let degree = 1 << log_degree;
    RowMajorMatrix::new(
        (0..degree)
            .map(|i| Val::from_u64((i & 0xff) as u64))
            .collect(),
        1,
    )
}

fn make_config() -> MyConfig {
    let mut rng = SmallRng::seed_from_u64(1337);
    let perm = Perm::new_from_rng_128(&mut rng);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress, 0);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft::default();
    let fri_params = FriParameters::new_testing(challenge_mmcs, 2);
    let pcs = MyPcs::new(dft, val_mmcs, fri_params);
    let challenger = Challenger::new(perm);
    StarkConfig::new(pcs, challenger)
}

fn bench_infer_chunks(c: &mut Criterion) {
    let mut group = c.benchmark_group("periodic_degree_infer_chunks");

    for log_degree in TRACE_LOG_DEGREES {
        let trace_degree = 1 << log_degree;
        for (label, periods) in PERIODIC_CASES {
            let air = PeriodicProductAir {
                periods: periods.to_vec(),
            };
            let layout = air_layout(&air);
            group.bench_function(BenchmarkId::new(label, format!("2^{log_degree}")), |b| {
                b.iter(|| {
                    black_box(get_log_num_quotient_chunks::<Val, _>(
                        black_box(&air),
                        black_box(layout),
                        black_box(trace_degree),
                        black_box(0),
                    ))
                });
            });
        }
    }

    group.finish();
}

fn bench_prove_periodic(c: &mut Criterion) {
    let config = make_config();
    let mut group = c.benchmark_group("periodic_degree_prove");
    group.sample_size(10);

    for log_degree in TRACE_LOG_DEGREES {
        let trace = make_trace(log_degree);
        for (label, periods) in PERIODIC_CASES {
            let air = PeriodicProductAir {
                periods: periods.to_vec(),
            };
            group.bench_function(BenchmarkId::new(label, format!("2^{log_degree}")), |b| {
                b.iter(|| {
                    black_box(prove(
                        black_box(&config),
                        black_box(&air),
                        black_box(trace.clone()),
                        black_box(&[]),
                    ))
                });
            });
        }
    }

    group.finish();
}

criterion_group!(benches, bench_infer_chunks, bench_prove_periodic);
criterion_main!(benches);
