use criterion::{Criterion, criterion_group, criterion_main};
use p3_air::symbolic::SymbolicAirBuilder;
use p3_air::{Air, BaseAir};
use p3_baby_bear::BabyBear;
use p3_blake3_air::Blake3Air;
use p3_challenger::DuplexChallenger;
use p3_commit::testing::TrivialPcs;
use p3_dft::NaiveDft;
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::extension::BinomialExtensionField;
use p3_field::{ExtensionField, PrimeField64, TwoAdicField};
use p3_goldilocks::Goldilocks;
use p3_keccak_air::KeccakAir;
use p3_matrix::dense::RowMajorMatrix;
use p3_symmetric::{CryptographicPermutation, Permutation};
use p3_uni_stark::{ProverConstraintFolder, StarkConfig, quotient_values};
use rand::SeedableRng;
use rand::distr::StandardUniform;
use rand::prelude::Distribution;
use rand::rngs::SmallRng;

/// Identity permutation — never actually called; only used to satisfy type bounds.
#[derive(Clone)]
struct IdPerm;
impl<T: Clone> Permutation<T> for IdPerm {
    fn permute_mut(&self, _input: &mut T) {}
}
impl<T: Clone> CryptographicPermutation<T> for IdPerm {}

type Challenger<F> = DuplexChallenger<F, IdPerm, 16, 8>;
type SC<F, EF> = StarkConfig<TrivialPcs<F, NaiveDft>, EF, Challenger<F>>;

fn bench_quotient<F, EF, A>(c: &mut Criterion, air: &A, label: &str, log_quotient_chunks: usize)
where
    F: PrimeField64 + TwoAdicField,
    EF: ExtensionField<F>,
    StandardUniform: Distribution<F>,
    A: BaseAir<F> + Air<SymbolicAirBuilder<F>> + for<'a> Air<ProverConstraintFolder<'a, SC<F, EF>>>,
{
    let width = air.width();

    for log_trace_length in [14, 16] {
        let trace_domain = TwoAdicMultiplicativeCoset::new(F::ONE, log_trace_length).unwrap();
        let quotient_domain =
            TwoAdicMultiplicativeCoset::new(F::GENERATOR, log_trace_length + log_quotient_chunks)
                .unwrap();

        let layout = p3_air::symbolic::AirLayout {
            preprocessed_width: 0,
            main_width: width,
            num_public_values: 0,
            ..Default::default()
        };

        let mut rng = SmallRng::seed_from_u64(1);
        let trace_on_quotient_domain =
            RowMajorMatrix::<F>::rand(&mut rng, quotient_domain.size(), width);

        let alpha = EF::from_basis_coefficients_fn(|i| F::from_u32(100 + i as u32));

        c.bench_function(
            &format!("quotient_values<{label}>/log_n={log_trace_length}"),
            |b| {
                b.iter(|| {
                    quotient_values::<SC<F, EF>, _, _>(
                        air,
                        &[],
                        layout,
                        trace_domain,
                        quotient_domain,
                        &trace_on_quotient_domain,
                        None,
                        alpha,
                    )
                });
            },
        );
    }
}

type BabyBearEF = BinomialExtensionField<BabyBear, 4>;
type GoldilocksEF = BinomialExtensionField<Goldilocks, 2>;

fn bench_quotient_values(c: &mut Criterion) {
    let keccak = KeccakAir {};
    let blake3 = Blake3Air {};

    // Both AIRs have max constraint degree 3 → log_quotient_chunks = log2_ceil(3-1) = 1
    bench_quotient::<BabyBear, BabyBearEF, _>(c, &keccak, "KeccakAir,BabyBear", 1);
    bench_quotient::<Goldilocks, GoldilocksEF, _>(c, &keccak, "KeccakAir,Goldilocks", 1);
    bench_quotient::<BabyBear, BabyBearEF, _>(c, &blake3, "Blake3Air,BabyBear", 1);
    bench_quotient::<Goldilocks, GoldilocksEF, _>(c, &blake3, "Blake3Air,Goldilocks", 1);
}

criterion_group!(benches, bench_quotient_values);
criterion_main!(benches);
