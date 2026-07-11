use core::borrow::Borrow;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_air::{Air, AirBuilder, BaseAir, BoundaryEnd, BoundaryPublic, WindowAccess};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_field::PrimeCharacteristicRing;
use p3_field::extension::BinomialExtensionField;
use p3_matrix::dense::RowMajorMatrix;
use p3_multi_stark::zerocheck::AirZerocheck;
use p3_sumcheck::layout::Table;
use rand::SeedableRng;
use rand::rngs::SmallRng;

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;
type Perm = Poseidon2BabyBear<16>;
type Ch = DuplexChallenger<F, Perm, 16, 8>;

const NUM_COLS: usize = 2;

/// Width of the wide trace whose columns are mostly current-row-only filler.
const WIDE_COLS: usize = 64;

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

/// Wide AIR carrying a Fibonacci recurrence in columns 0 and 1, with the
/// remaining `WIDE_COLS - 2` columns committed but read only on the current row.
///
/// `all_next` selects the declared next-row footprint: `true` declares every
/// column (the conservative default for an AIR that does not override
/// [`BaseAir::main_next_row_columns`]), `false` declares just the two columns
/// the constraints read on the next row. Both share one [`Air::eval`], so the
/// prover-time gap between them reflects the successor-table cost of the
/// undeclared columns.
struct WideFibAir {
    all_next: bool,
}

impl<X> BaseAir<X> for WideFibAir {
    fn width(&self) -> usize {
        WIDE_COLS
    }
    fn main_next_row_columns(&self) -> Vec<usize> {
        if self.all_next {
            (0..WIDE_COLS).collect()
        } else {
            vec![0, 1]
        }
    }
}

impl<AB: AirBuilder> Air<AB> for WideFibAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();
        let next = main.next_slice();
        let mut trans = builder.when_transition();
        trans.assert_eq(local[1], next[0]);
        trans.assert_eq(local[0] + local[1], next[1]);
    }
}

fn wide_fib_trace(n: usize) -> RowMajorMatrix<F> {
    let mut left = F::ZERO;
    let mut right = F::ONE;
    let mut values = Vec::with_capacity(WIDE_COLS * n);
    for i in 0..n {
        values.push(left);
        values.push(right);
        // Filler columns: committed and opened at the current row, never read ahead.
        for c in 2..WIDE_COLS {
            values.push(F::from_u64((i + c) as u64));
        }
        let next_right = left + right;
        left = right;
        right = next_right;
    }
    RowMajorMatrix::new(values, WIDE_COLS)
}

/// Fibonacci AIR binding its public inputs by boundary IO instead of constraints.
///
/// The transition recurrence is asserted here.
/// The two seeds and the output are declared as public boundary cells.
/// The prover asserts corner-zero pins in place of first/last-row constraints.
struct FibIoAir;

impl<X> BaseAir<X> for FibIoAir {
    fn width(&self) -> usize {
        NUM_COLS
    }
    fn num_public_values(&self) -> usize {
        3
    }
    fn public_boundary_io(&self) -> Vec<BoundaryPublic> {
        vec![
            BoundaryPublic::new(0, BoundaryEnd::First, 0),
            BoundaryPublic::new(1, BoundaryEnd::First, 1),
            BoundaryPublic::new(1, BoundaryEnd::Last, 2),
        ]
    }
}

impl<AB: AirBuilder> Air<AB> for FibIoAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &FibRow<AB::Var> = main.current_slice().borrow();
        let next: &FibRow<AB::Var> = main.next_slice().borrow();
        let mut trans = builder.when_transition();
        trans.assert_eq(local.right, next.left);
        trans.assert_eq(local.left + local.right, next.right);
    }
}

fn fresh_challenger() -> Ch {
    let mut rng = SmallRng::seed_from_u64(0xC0FFEE);
    let perm = Perm::new_from_rng_128(&mut rng);
    Ch::new(perm)
}

fn bench_zerocheck(c: &mut Criterion) {
    let mut group = c.benchmark_group("zerocheck_prove");
    group.sample_size(10);
    for log_height in [16, 18, 20] {
        let n = 1 << log_height;
        let trace = fib_trace(n);
        let last = trace.values[(n - 1) * NUM_COLS + 1];
        let pis = [F::ZERO, F::ONE, last];
        let table = Table::new(trace.transpose());
        let air = FibAir;
        for num_airs in [1, 5, 10] {
            let airs = (0..num_airs).map(|_| &air).collect::<Vec<_>>();
            let preprocessed = vec![None; num_airs];
            let tables = vec![&table; num_airs];
            let public_values = vec![&pis[..]; num_airs];
            let zerocheck = AirZerocheck::new(&airs, 0);
            group.bench_with_input(
                BenchmarkId::new(format!("{num_airs}_airs"), log_height),
                &n,
                |b, _| {
                    b.iter(|| {
                        let mut ch = fresh_challenger();
                        zerocheck.prove::<F, EF, _>(&preprocessed, &tables, &public_values, &mut ch)
                    });
                },
            );
        }
    }
    group.finish();
}

fn bench_wide_zerocheck(c: &mut Criterion) {
    let mut group = c.benchmark_group("zerocheck_prove_wide");
    group.sample_size(10);
    for log_height in [16, 18, 20] {
        let n = 1 << log_height;
        let table = Table::new(wide_fib_trace(n).transpose());
        let empty_public_values: &[F] = &[];
        for (label, all_next) in [("all_next", true), ("sparse_next", false)] {
            let air = WideFibAir { all_next };
            for num_airs in [1, 5, 10] {
                let airs = (0..num_airs).map(|_| &air).collect::<Vec<_>>();
                let preprocessed = vec![None; num_airs];
                let tables = vec![&table; num_airs];
                let public_values = vec![empty_public_values; num_airs];
                let zerocheck = AirZerocheck::new(&airs, 0);
                group.bench_with_input(
                    BenchmarkId::new(format!("{label}_{num_airs}_airs"), log_height),
                    &n,
                    |b, _| {
                        b.iter(|| {
                            let mut ch = fresh_challenger();
                            zerocheck.prove::<F, EF, _>(
                                &preprocessed,
                                &tables,
                                &public_values,
                                &mut ch,
                            )
                        });
                    },
                );
            }
        }
    }
    group.finish();
}

/// Compare the two ways to bind a Fibonacci AIR's public inputs, at equal size.
///
/// - `constraint`: first/last-row boundary constraints on the committed trace.
/// - `boundary_io`: corner-zero pins on the same trace, folded as in the real prover.
///
/// Both assert the same number of degree-2 constraints, so this isolates the
/// per-round fold cost of the two bindings.
/// The one extra cost of boundary IO, a single trace clone before commitment,
/// lives in the committed prover and is not exercised here.
fn bench_public_input_binding(c: &mut Criterion) {
    let mut group = c.benchmark_group("public_input_binding");
    group.sample_size(10);
    for log_height in [18, 20] {
        // A shared Fibonacci trace and its public seeds and output.
        let n = 1 << log_height;
        let trace = fib_trace(n);
        let last = trace.values[(n - 1) * NUM_COLS + 1];
        let pis = [F::ZERO, F::ONE, last];
        let table = Table::new(trace.transpose());
        let public_values = vec![&pis[..]];
        let preprocessed = vec![None; 1];
        let tables = vec![&table; 1];

        // Constraint binding: the AIR asserts the boundary equalities directly.
        let constraint_air = FibAir;
        let constraint_airs = vec![&constraint_air];
        let constraint = AirZerocheck::new(&constraint_airs, 0);
        group.bench_with_input(BenchmarkId::new("constraint", log_height), &n, |b, _| {
            b.iter(|| {
                let mut ch = fresh_challenger();
                constraint.prove::<F, EF, _>(&preprocessed, &tables, &public_values, &mut ch)
            });
        });

        // Boundary-IO binding: the folder asserts corner-zero pins instead.
        let io_air = FibIoAir;
        let io_airs = vec![&io_air];
        let io = AirZerocheck::new(&io_airs, 0);
        group.bench_with_input(BenchmarkId::new("boundary_io", log_height), &n, |b, _| {
            b.iter(|| {
                let mut ch = fresh_challenger();
                io.prove::<F, EF, _>(&preprocessed, &tables, &public_values, &mut ch)
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_zerocheck,
    bench_wide_zerocheck,
    bench_public_input_binding
);
criterion_main!(benches);
