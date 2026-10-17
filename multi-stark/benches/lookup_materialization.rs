use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_baby_bear::BabyBear;
use p3_field::PrimeCharacteristicRing;
use p3_field::extension::BinomialExtensionField;
use p3_lookup::{Count, InteractionBuilder};
use p3_matrix::dense::RowMajorMatrix;
use p3_multi_stark::lookup::LookupPlan;
use p3_sumcheck::layout::Table;

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;
struct LookupAir {
    width: usize,
    tuples: usize,
}
impl BaseAir<F> for LookupAir {
    fn width(&self) -> usize {
        self.width
    }
}
impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for LookupAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();
        let next = main.next_slice();
        for _ in 0..self.tuples {
            builder.push_local_interaction([
                (
                    vec![local[0].into(), next[1].into()],
                    Count::bounded(AB::Expr::ONE, 1),
                ),
                (
                    vec![local[0].into(), next[1].into()],
                    Count::provided(-AB::Expr::ONE),
                ),
            ]);
        }
    }
}
fn table(log: usize, width: usize) -> Table<F> {
    let n = 1 << log;
    Table::new(RowMajorMatrix::new(
        (0..width * n)
            .map(|i| F::from_u64((i % 251) as u64))
            .collect(),
        n,
    ))
}
fn bench_materialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("lookup_materialization");
    group.sample_size(10);
    for width in [8, 64, 512] {
        let air = LookupAir { width, tuples: 1 };
        let tall = table(16, width);
        let tiny = table(0, width);
        for mixed in [false, true] {
            let airs = if mixed { vec![&air, &air] } else { vec![&air] };
            let tables = if mixed {
                vec![&tall, &tiny]
            } else {
                vec![&tall]
            };
            let logs = if mixed { vec![16, 0] } else { vec![16] };
            let plan = LookupPlan::<F>::build::<EF, _>(&airs, &logs)
                .unwrap()
                .unwrap();
            let fixed = vec![None; airs.len()];
            let publics = vec![&[][..]; airs.len()];
            group.bench_function(
                BenchmarkId::new(if mixed { "mixed_16_0" } else { "tall_16" }, width),
                |b| {
                    b.iter(|| {
                        plan.materialize_fraction(
                            &tables,
                            &fixed,
                            &publics,
                            EF::from_u64(98765),
                            EF::from_u64(13),
                        )
                    })
                },
            );
        }
    }
    group.finish();
}
criterion_group!(benches, bench_materialization);
criterion_main!(benches);
