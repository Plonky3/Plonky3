use criterion::{Criterion, criterion_group, criterion_main};
use p3_air::BaseAir;
use p3_baby_bear::BabyBear;
use p3_challenger::DuplexChallenger;
use p3_commit::testing::TrivialPcs;
use p3_dft::NaiveDft;
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::extension::BinomialExtensionField;
use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing};
use p3_keccak_air::KeccakAir;
use p3_matrix::dense::RowMajorMatrix;
use p3_symmetric::{CryptographicPermutation, Permutation};
use p3_uni_stark::{AirLayout, StarkConfig, quotient_values};
use rand::SeedableRng;
use rand::rngs::SmallRng;

// KeccakAir has max constraint degree 3.
const LOG_QUOTIENT_CHUNKS: usize = 1; // log2_ceil(3 - 1)

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;

/// Identity permutation — never actually called; only used to satisfy type bounds.
#[derive(Clone)]
struct IdPerm;
impl<T: Clone> Permutation<T> for IdPerm {
    fn permute_mut(&self, _input: &mut T) {}
}
impl<T: Clone> CryptographicPermutation<T> for IdPerm {}

type Challenger = DuplexChallenger<F, IdPerm, 16, 8>;
type SC = StarkConfig<TrivialPcs<F, NaiveDft>, EF, Challenger>;

fn bench_quotient_values(c: &mut Criterion) {
    let air = KeccakAir {};
    let width = <KeccakAir as BaseAir<F>>::width(&air);

    for log_trace_length in [14, 16] {
        // Build domains directly.
        let trace_domain = TwoAdicMultiplicativeCoset::new(F::ONE, log_trace_length).unwrap();
        let quotient_domain =
            TwoAdicMultiplicativeCoset::new(F::GENERATOR, log_trace_length + LOG_QUOTIENT_CHUNKS)
                .unwrap();

        let layout = AirLayout {
            preprocessed_width: 0,
            main_width: width,
            num_public_values: 0,
            ..Default::default()
        };

        // Random matrix standing in for trace-on-quotient-domain.
        let mut rng = SmallRng::seed_from_u64(1);
        let trace_on_quotient_domain =
            RowMajorMatrix::<F>::rand(&mut rng, quotient_domain.size(), width);

        let alpha = EF::from_basis_coefficients_fn(|i| F::from_u32(100 + i as u32));

        c.bench_function(
            &format!("quotient_values<KeccakAir>/log_n={log_trace_length}"),
            |b| {
                b.iter(|| {
                    quotient_values::<SC, _, _>(
                        &air,
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

criterion_group!(benches, bench_quotient_values);
criterion_main!(benches);
