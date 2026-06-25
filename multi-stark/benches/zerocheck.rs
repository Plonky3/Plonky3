use core::borrow::Borrow;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_field::PrimeCharacteristicRing;
use p3_field::extension::BinomialExtensionField;
use p3_matrix::dense::RowMajorMatrix;
use p3_multi_stark::zerocheck::AirZerocheck;
use rand::SeedableRng;
use rand::rngs::SmallRng;

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;
type Perm = Poseidon2BabyBear<16>;
type Ch = DuplexChallenger<F, Perm, 16, 8>;

const NUM_COLS: usize = 2;

struct FibAir;
struct FibRow<T> {
    left: T,
    right: T,
}

impl<T> Borrow<FibRow<T>> for [T] {
    fn borrow(&self) -> &FibRow<T> {
        let ptr = self.as_ptr() as *const FibRow<T>;
        unsafe { &*ptr }
    }
}

impl<X> BaseAir<X> for FibAir {
    fn width(&self) -> usize {
        NUM_COLS
    }
    fn num_public_values(&self) -> usize {
        3
    }
}

impl<AB: AirBuilder> Air<AB> for FibAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let pis = builder.public_values();
        let (a, b, x) = (pis[0], pis[1], pis[2]);
        let local: &FibRow<AB::Var> = main.current_slice().borrow();
        let next: &FibRow<AB::Var> = main.next_slice().borrow();
        let mut first = builder.when_first_row();
        first.assert_eq(local.left, a);
        first.assert_eq(local.right, b);
        let mut trans = builder.when_transition();
        trans.assert_eq(local.right, next.left);
        trans.assert_eq(local.left + local.right, next.right);
        builder.when_last_row().assert_eq(local.right, x);
    }
}

fn fib_trace(n: usize) -> RowMajorMatrix<F> {
    let mut left = F::ZERO;
    let mut right = F::ONE;
    let mut values = Vec::with_capacity(NUM_COLS * n);
    for _ in 0..n {
        values.push(left);
        values.push(right);
        let next_right = left + right;
        left = right;
        right = next_right;
    }
    RowMajorMatrix::new(values, NUM_COLS)
}

fn fresh_challenger() -> Ch {
    let mut rng = SmallRng::seed_from_u64(0xC0FFEE);
    let perm = Perm::new_from_rng_128(&mut rng);
    Ch::new(perm)
}

fn bench_zerocheck(c: &mut Criterion) {
    let mut group = c.benchmark_group("zerocheck_prove");
    for log_height in [16, 18, 20] {
        let n = 1 << log_height;
        let trace = fib_trace(n);
        let last = trace.values[(n - 1) * NUM_COLS + 1];
        let pis = [F::ZERO, F::ONE, last];
        let zerocheck = AirZerocheck::new(&FibAir, 0);
        group.bench_with_input(BenchmarkId::from_parameter(log_height), &n, |b, _| {
            b.iter(|| {
                let mut ch = fresh_challenger();
                zerocheck.prove::<F, EF, _>(&trace, &pis, &mut ch)
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_zerocheck);
criterion_main!(benches);
