use core::borrow::Borrow;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_batch_stark::{ProverData, StarkInstance, prove_batch, verify_batch};
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeField64};
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark::StarkConfig;
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

// ---------------------------------------------------------------------------
// Fibonacci AIR (5 constraints, width 2, 1 preprocessed column)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct FibonacciAir {
    log_height: usize,
}

impl<F: Field> BaseAir<F> for FibonacciAir {
    fn width(&self) -> usize {
        2
    }
    fn num_public_values(&self) -> usize {
        3
    }
    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let n = 1 << self.log_height;
        let mut m = RowMajorMatrix::new(F::zero_vec(n), 1);
        for (i, v) in m.values.iter_mut().enumerate().take(n) {
            *v = F::from_u64(i as u64);
        }
        Some(m)
    }
    fn preprocessed_width(&self) -> usize {
        1
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
struct FibRow<F> {
    left: F,
    right: F,
}

impl<F> Borrow<FibRow<F>> for [F] {
    fn borrow(&self) -> &FibRow<F> {
        debug_assert_eq!(self.len(), 2);
        unsafe { &*(self.as_ptr() as *const FibRow<F>) }
    }
}

impl<AB: AirBuilder> Air<AB> for FibonacciAir
where
    AB::F: Field,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let pis = builder.public_values();
        let (a0, b0, x) = (pis[0], pis[1], pis[2]);
        let local: &FibRow<AB::Var> = main.current_slice().borrow();
        let next: &FibRow<AB::Var> = main.next_slice().borrow();

        builder.when_first_row().assert_eq(local.left, a0);
        builder.when_first_row().assert_eq(local.right, b0);
        builder.when_transition().assert_eq(local.right, next.left);
        builder
            .when_transition()
            .assert_eq(local.left + local.right, next.right);
        builder.when_last_row().assert_eq(local.right, x);
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn fib_trace<F: PrimeField64>(n: usize) -> (RowMajorMatrix<F>, Vec<F>) {
    let mut vals = Vec::with_capacity(n * 2);
    let (mut a, mut b) = (F::ZERO, F::ONE);
    for _ in 0..n {
        vals.push(a);
        vals.push(b);
        let c = a + b;
        a = b;
        b = c;
    }
    let last = vals[2 * (n - 1) + 1];
    (RowMajorMatrix::new(vals, 2), vec![F::ZERO, F::ONE, last])
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

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_prove_batch(c: &mut Criterion) {
    let config = make_config();
    let log_height = 14; // 16 384 rows per instance

    let mut group = c.benchmark_group("prove_batch");

    for n_instances in [1, 4, 8, 16] {
        // Build instance data once; reused across iterations.
        let instances_data: Vec<_> = (0..n_instances)
            .map(|_| {
                let air = FibonacciAir { log_height };
                let (trace, pis) = fib_trace::<Val>(1 << log_height);
                (air, trace, pis)
            })
            .collect();

        let instances: Vec<_> = instances_data
            .iter()
            .map(|(air, trace, pis)| StarkInstance {
                air,
                trace,
                public_values: pis.clone(),
            })
            .collect();

        let prover_data = ProverData::from_instances(&config, &instances);

        group.bench_function(BenchmarkId::new("fib", n_instances), |b| {
            b.iter(|| {
                let proof = prove_batch(&config, &instances, &prover_data);

                // Lightweight sanity check — only on the first iteration would be
                // enough, but verify is cheap relative to prove so we keep it.
                let airs: Vec<_> = instances_data.iter().map(|(a, _, _)| *a).collect();
                let pvs: Vec<_> = instances_data.iter().map(|(_, _, p)| p.clone()).collect();
                verify_batch(&config, &airs, &proof, &pvs, &prover_data.common).unwrap();
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_prove_batch);
criterion_main!(benches);
