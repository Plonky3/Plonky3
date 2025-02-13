use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use itertools::Itertools;
use p3_baby_bear::BabyBear;
use p3_coset::TwoAdicCoset;
use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
use p3_field::extension::BinomialExtensionField;
use p3_field::TwoAdicField;
use p3_goldilocks::Goldilocks;
use p3_poly::Polynomial;
use rand::distributions::{Distribution, Standard};
use rand::{thread_rng, Rng};

type BB = BabyBear;
type BBExt = BinomialExtensionField<BB, 5>;
type GL = Goldilocks;
type GLExt = BinomialExtensionField<GL, 2>;

// In the special case where one wants to evaluate a polynomial on a coset whose
// shift is equal to the generator of its subgroup, two different methods arise
// naturally (cf. method 1 and method 2 below). This benchmark compares the
// performance of the two and shows that TwoAdicCoset::evaluate_polynomial costs
// essentially the same as the more efficient one, which is 10-20% faster than
// the alternative.
fn bench_field<F: NamedField + TwoAdicField>(c: &mut Criterion, log_sizes: &[usize])
where
    Standard: Distribution<F>,
{
    let mut group = c.benchmark_group(format!("coset_evaluation_{}", F::name()));
    group.sample_size(10);

    let dft = Radix2Dit::default();

    for &log_size in log_sizes {
        let generator = F::two_adic_generator(log_size);

        // Coset w * <w>, which contains the same elements as <w> but has a
        // different ordering
        let mut coset = TwoAdicCoset::new(generator, log_size);
        coset.initialise_fft();

        let rng = thread_rng();
        let poly_coeffs = rng.sample_iter(&Standard).take(1 << log_size).collect_vec();
        let poly = Polynomial::from_coeffs(poly_coeffs.clone());

        // Sanity check
        let res_1 = dft.coset_dft(poly_coeffs.clone(), generator);
        let res_2 = {
            let mut res = dft.dft(poly_coeffs.clone());
            res.rotate_left(1);
            res
        };
        let res_3 = coset.evaluate_polynomial(&poly);
        assert_eq!(res_1, res_2);
        assert_eq!(res_1, res_3);

        // Method 1: call coset_dft, which transforms the polynomial by
        // multiplying each coefficient by a suitable power of the shift
        group.bench_function(BenchmarkId::new("coset_dft", log_size), |b| {
            b.iter(
                // We clone here to mimic the internls of
                // TwoAdicCoset::evaluate_polynomial
                || dft.coset_dft(poly_coeffs.clone(), generator),
            )
        });

        // Method 2: call dft, then rotate the result
        group.bench_function(BenchmarkId::new("dft + rotate", log_size), |b| {
            b.iter(|| {
                // We clone here to mimic the internls of
                // TwoAdicCoset::evaluate_polynomial
                let mut evals = dft.dft(poly_coeffs.clone());
                evals.rotate_left(1);
            })
        });

        // TwoAdicCoset does the more efficient of the two
        group.bench_function(BenchmarkId::new("TwoAdicCoset", log_size), |b| {
            b.iter(|| coset.evaluate_polynomial(&poly))
        });
    }
}

fn bench(c: &mut Criterion) {
    let log_sizes = (16..=22).step_by(2).collect_vec();

    bench_field::<BB>(c, &log_sizes);
    bench_field::<BBExt>(c, &log_sizes);
    bench_field::<GL>(c, &log_sizes);
    bench_field::<GLExt>(c, &log_sizes);
}

criterion_group!(benches, bench);
criterion_main!(benches);

// Pretty field names

trait NamedField {
    fn name() -> &'static str;
}

impl NamedField for BB {
    fn name() -> &'static str {
        "BabyBear"
    }
}

impl NamedField for BBExt {
    fn name() -> &'static str {
        "BabyBearExt"
    }
}

impl NamedField for GL {
    fn name() -> &'static str {
        "Goldilocks"
    }
}

impl NamedField for GLExt {
    fn name() -> &'static str {
        "GoldilocksExt"
    }
}
