//! CCS reduction prover and verifier benchmarks.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_ccs::{
    BlockDiagonal, Ccs, Constraint, Operator, RowSelector, SparseMatrix, Term, TraceColumn, prove,
    verify,
};
use p3_challenger::DuplexChallenger;
use p3_field::PrimeCharacteristicRing;
use p3_field::extension::BinomialExtensionField;
use rand::SeedableRng;
use rand::rngs::SmallRng;

/// Base field.
type F = BabyBear;
/// Challenge field.
type EF = BinomialExtensionField<F, 4>;
/// Fiat-Shamir challenger.
type Challenger = DuplexChallenger<F, Poseidon2BabyBear<16>, 16, 8>;

fn challenger() -> Challenger {
    // Fixed permutation: every run draws the same challenges.
    let mut rng = SmallRng::seed_from_u64(1);
    Challenger::new(Poseidon2BabyBear::new_from_rng_128(&mut rng))
}

/// Fibonacci AIR and its trace.
fn fibonacci(log_height: usize) -> (Ccs<F>, Vec<F>) {
    let cell = |column, offset| Operator::from(TraceColumn::new(log_height, 1, column, offset));
    let ccs = Ccs::new(
        vec![cell(0, 0), cell(1, 0), cell(0, 1), cell(1, 1)],
        vec![
            Constraint::new(vec![Term::new(F::ONE, [2]), Term::new(F::NEG_ONE, [1])]),
            Constraint::new(vec![
                Term::new(F::ONE, [3]),
                Term::new(F::NEG_ONE, [0]),
                Term::new(F::NEG_ONE, [1]),
            ]),
        ]
        .into_iter()
        .map(|k| k.on(RowSelector::Transition))
        .collect(),
    )
    .unwrap();

    // Column-major trace.
    let height = 1 << log_height;
    let (mut a, mut b) = (vec![F::ONE], vec![F::ONE]);
    for row in 1..height {
        a.push(b[row - 1]);
        b.push(a[row - 1] + b[row - 1]);
    }
    a.extend(b);
    (ccs, a)
}

/// Copies of the R1CS x * y = 12, x + y = 7, each over (1, 3, 4, 0).
fn copied_r1cs(log_copies: usize) -> (Ccs<F>, Vec<F>) {
    let block = |entries| BlockDiagonal::new(log_copies, SparseMatrix::new(1, 2, entries)).into();
    let ccs = Ccs::new(
        vec![
            block(vec![(0, 1, F::ONE), (1, 1, F::ONE), (1, 2, F::ONE)]),
            block(vec![(0, 2, F::ONE), (1, 0, F::ONE)]),
            block(vec![(0, 0, F::from_u32(12)), (1, 0, F::from_u32(7))]),
        ],
        vec![Constraint::new(vec![
            Term::new(F::ONE, [0, 1]),
            Term::new(F::NEG_ONE, [2]),
        ])],
    )
    .unwrap();
    let z = (0..1 << log_copies)
        .flat_map(|_| [1, 3, 4, 0])
        .map(F::from_u32)
        .collect();
    (ccs, z)
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("ccs");
    group.sample_size(10);

    for (name, log_size, (ccs, z)) in [
        ("fibonacci", 20, fibonacci(20)),
        ("copied_r1cs", 18, copied_r1cs(18)),
    ] {
        // Full prover.
        group.bench_with_input(
            BenchmarkId::new(format!("{name}/prove"), log_size),
            &z,
            |b, z| {
                b.iter(|| prove::<F, EF, _, _>(&ccs, z, 0, &mut challenger()));
            },
        );

        // Full verifier.
        let (proof, _) = prove::<F, EF, _, _>(&ccs, &z, 0, &mut challenger());
        group.bench_with_input(
            BenchmarkId::new(format!("{name}/verify"), log_size),
            &proof,
            |b, proof| {
                b.iter(|| verify(&ccs, proof, 0, &mut challenger()).unwrap());
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
