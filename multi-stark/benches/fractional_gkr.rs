//! Fractional GKR prover over two tables of `2^n` fractions, proven back to back.
//!
//! Set `RAYON_NUM_THREADS` to pick the thread count.

use std::hint::black_box;

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_binary_field::{BinaryChallenger, Poly64, Poly192};
use p3_challenger::fs::TranscriptField;
use p3_challenger::{DuplexChallenger, FieldChallenger, HashChallenger};
use p3_field::extension::BinomialExtensionField;
use p3_field::{ExtensionField, Field, batch_multiplicative_inverse};
use p3_keccak::Keccak256Hash;
use p3_multi_stark::fractional_gkr::{Fraction, LeafNumerator, prove_fractional_gkr};
use p3_multilinear_util::poly::{Poly, PolyMaybePacked};
use rand::SeedableRng;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;

type Bb = BabyBear;
type Bb4 = BinomialExtensionField<Bb, 4>;
type Perm = Poseidon2BabyBear<16>;

/// Tables proven per measured run.
const TABLES: usize = 2;

/// Table sizes, as variable counts.
const LOG_SIZES: [usize; 2] = [20, 22];

/// Two packed fraction tables over `F` and `EF`, each summing to zero.
struct Tables<F: Field, EF: ExtensionField<F>> {
    /// Base-field numerators.
    numerators: Vec<Poly<F>>,
    /// Extension-field numerators, the same values lifted and packed.
    lifted: Vec<PolyMaybePacked<F, EF>>,
    /// Packed denominators.
    denominators: Vec<PolyMaybePacked<F, EF>>,
}

impl<F: Field, EF: ExtensionField<F>> Tables<F, EF>
where
    StandardUniform: Distribution<F> + Distribution<EF>,
{
    fn new(log_size: usize) -> Self {
        let mut rng = SmallRng::seed_from_u64(log_size as u64);
        let (numerators, (lifted, denominators)) = (0..TABLES)
            .map(|_| {
                // Random fractions, the last denominator solved for so the table sums to zero.
                let numer = Poly::<F>::rand(&mut rng, log_size);
                let mut denom = Poly::<EF>::rand(&mut rng, log_size);
                let last = denom.num_evals() - 1;
                let inverses = batch_multiplicative_inverse(&denom.as_slice()[..last]);
                let partial: EF = inverses
                    .iter()
                    .zip(numer.as_slice())
                    .map(|(&inverse, &n)| inverse * n)
                    .sum();
                denom.as_mut_slice()[last] =
                    -(partial * numer.as_slice()[last].inverse()).inverse();

                let lifted =
                    Poly::<EF>::new(numer.as_slice().iter().map(|&n| EF::from(n)).collect());
                let lifted = PolyMaybePacked::Packed(lifted.pack::<F, EF>());
                let denom = PolyMaybePacked::Packed(denom.pack::<F, EF>());
                (numer, (lifted, denom))
            })
            .unzip();
        Self {
            numerators,
            lifted,
            denominators,
        }
    }

    /// Proves every table on one challenger.
    fn prove<C: FieldChallenger<F>>(&self, challenger: &mut C, lifted: bool)
    where
        F: TranscriptField,
    {
        for table in 0..TABLES {
            let n = if lifted {
                LeafNumerator::Ext(&self.lifted[table])
            } else {
                LeafNumerator::Base(&self.numerators[table])
            };
            black_box(prove_fractional_gkr(
                Fraction {
                    n,
                    d: &self.denominators[table],
                },
                challenger,
            ));
        }
    }
}

fn bench_field<F, EF, C>(c: &mut Criterion, name: &str, challenger: impl Fn() -> C)
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: FieldChallenger<F>,
    StandardUniform: Distribution<F> + Distribution<EF>,
{
    let mut group = c.benchmark_group(format!("fractional_gkr/{name}"));
    group.sample_size(10);
    for log_size in LOG_SIZES {
        let tables = Tables::<F, EF>::new(log_size);
        for (numerator, lifted) in [("base", false), ("ext", true)] {
            group.bench_function(BenchmarkId::new(numerator, log_size), |b| {
                b.iter_batched_ref(
                    &challenger,
                    |challenger| tables.prove(challenger, lifted),
                    BatchSize::SmallInput,
                );
            });
        }
    }
    group.finish();
}

fn bench(c: &mut Criterion) {
    let perm = Perm::new_from_rng_128(&mut SmallRng::seed_from_u64(1));
    bench_field::<Bb, Bb4, _>(c, "babybear4", || {
        DuplexChallenger::<Bb, Perm, 16, 8>::new(perm.clone())
    });
    bench_field::<Poly64, Poly192, _>(c, "poly192", || {
        BinaryChallenger::<Poly64, HashChallenger<u8, Keccak256Hash, 32>>::from_hasher(
            b"fractional-gkr-bench".to_vec(),
            Keccak256Hash,
        )
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
