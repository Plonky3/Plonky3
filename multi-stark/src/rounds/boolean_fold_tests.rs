//! Packed first folds agree with independent dense MLEs, including successor tails.

use alloc::vec;

use p3_air::{AirBuilder, WindowAccess};
use p3_baby_bear::BabyBear;
use p3_binary_field::{BinaryField128, Ghash128};
use p3_field::extension::BinomialExtensionField;
use p3_matrix::dense::RowMajorMatrix;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use super::*;

#[derive(Clone)]
struct FoldAir<F> {
    width: usize,
    prep: usize,
    periodic: Vec<Vec<F>>,
    next: Vec<usize>,
}
impl<F: Field> BaseAir<F> for FoldAir<F> {
    fn width(&self) -> usize {
        self.width
    }
    fn preprocessed_width(&self) -> usize {
        self.prep
    }
    fn main_next_row_columns(&self) -> Vec<usize> {
        self.next.clone()
    }
    fn preprocessed_next_row_columns(&self) -> Vec<usize> {
        (0..self.prep).collect()
    }
    fn num_periodic_columns(&self) -> usize {
        self.periodic.len()
    }
    fn periodic_columns(&self) -> alloc::borrow::Cow<'_, [Vec<F>]> {
        alloc::borrow::Cow::Borrowed(&self.periodic)
    }
}
impl<AB: AirBuilder> Air<AB> for FoldAir<AB::F>
where
    AB::F: Field,
{
    fn eval(&self, b: &mut AB) {
        b.assert_bool(b.main().current_slice()[0]);
    }
}
fn bits<F: Field>(width: usize, n: usize, seed: u64) -> Table<F> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let h = 1usize << n;
    let mask = if h < 64 { (1u64 << h) - 1 } else { u64::MAX };
    Table::from_packed_bits(
        RowMajorMatrix::new(
            (0..width * h.div_ceil(64))
                .map(|_| rng.random::<u64>() & mask)
                .collect(),
            width,
        ),
        n,
    )
}
fn dense<F: Field>(table: &Table<F>) -> Table<F> {
    Table::new(RowMajorMatrix::new(
        table.columns().flat_map(|c| c.values()).collect(),
        1 << table.num_variables(),
    ))
}
fn base<'a, F: Field, EF: ExtensionField<F>>(
    air: &'a FoldAir<F>,
    table: &'a Table<F>,
    prep: Option<&'a Table<F>>,
    tau: &[EF],
) -> RoundStateBase<'a, 'a, FoldAir<F>, F, EF> {
    let stage = Stage::new(
        vec![air],
        vec![&[]],
        vec![0],
        vec![prep],
        vec![table],
        vec![AirProfile {
            degrees: AirDegrees {
                constraints: 2,
                interactions: 0,
            },
            num_constraints: 1,
            declares_lookups: false,
            declares_indexed: false,
        }],
        StageCoupling::new(BTreeMap::new(), BTreeMap::new(), vec![]),
    );
    RoundStateBase::new(
        stage,
        EF::ONE,
        EF::ONE,
        vec![EF::ONE],
        Point::new(tau.to_vec()),
        0,
    )
}
fn cells<F: Field, EF: ExtensionField<F>>(
    state: &RoundStateExt<'_, '_, FoldAir<F>, F, EF>,
) -> Vec<Vec<EF>> {
    match &state.columns {
        ExtColumns::Scalar(cols) => cols.iter().map(|c| c.as_slice().to_vec()).collect(),
        ExtColumns::Packed(cols) => cols
            .iter()
            .map(|c| {
                c.as_slice()
                    .iter()
                    .flat_map(|p| (0..F::Packing::WIDTH).map(|lane| p.extract(lane)))
                    .collect()
            })
            .collect(),
        _ => unreachable!(),
    }
}
fn check<F: Field, EF: ExtensionField<F>>(points: &[EF])
where
    EF::ExtensionPacking: From<EF> + From<F::Packing>,
{
    for n in 1..=8 {
        for packed in [false, true] {
            for prep_packed in [false, true] {
                for first in [EF::ZERO, EF::ONE, points[0]] {
                    let h = 1 << n;
                    let main_bits = bits::<F>(5, n, 17);
                    let table = if packed { main_bits } else { dense(&main_bits) };
                    let prep_bits = bits::<F>(2, n, 71);
                    let prep = if prep_packed {
                        prep_bits
                    } else {
                        dense(&prep_bits)
                    };
                    let air = FoldAir {
                        width: 5,
                        prep: 2,
                        periodic: vec![vec![F::ZERO, F::ONE]],
                        next: vec![0, 2, 4],
                    };
                    let tau = vec![points[1]; n];
                    let mut a = base(&air, &table, Some(&prep), &tau).fold(first);
                    let dense_main = dense(&table);
                    let dense_prep = dense(&prep);
                    let mut b = base(&air, &dense_main, Some(&dense_prep), &tau).fold(first);
                    let mut all_columns = table
                        .columns()
                        .chain(prep.columns())
                        .map(|c| c.values().map(EF::from).collect::<Vec<_>>())
                        .collect::<Vec<_>>();
                    all_columns.push((0..h).map(|i| EF::from(air.periodic[0][i % 2])).collect());
                    let mut shifted = all_columns
                        .iter()
                        .map(|v| {
                            v.iter()
                                .skip(1)
                                .copied()
                                .chain(v.last().copied())
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();
                    for round in 0..n {
                        let r = if round == 0 {
                            first
                        } else {
                            points[(round + 1) % points.len()]
                        };
                        if round > 0 {
                            a.fold(r);
                            b.fold(r);
                        }
                        for cols in [&mut all_columns, &mut shifted] {
                            for col in cols.iter_mut() {
                                let half = col.len() / 2;
                                for row in 0..half {
                                    col[row] = col[row] + r * (col[row + half] - col[row]);
                                }
                                col.truncate(half);
                            }
                        }
                        assert_eq!(
                            cells(&a),
                            all_columns,
                            "mixed-layout fold n={n} round={round}"
                        );
                        assert_eq!(cells(&b), all_columns, "dense fold n={n} round={round}");
                        assert_eq!(a.next_tail, b.next_tail, "repeat-last tails");
                        for col in [0, 2, 4, 5, 6] {
                            assert_eq!(
                                a.next_tail[col],
                                *shifted[col].last().unwrap(),
                                "independent shifted MLE tail"
                            );
                        }
                    }
                }
            }
        }
    }
}
#[test]
fn packed_folds_match_dense_tables_and_repeat_last_successors() {
    let mut rng = SmallRng::seed_from_u64(0xB052);
    check::<BinaryField128, BinaryField128>(&[(); 10].map(|_| rng.random()));
    check::<Ghash128, Ghash128>(&[(); 10].map(|_| rng.random()));
    check::<BabyBear, BinomialExtensionField<BabyBear, 4>>(&[(); 10].map(|_| rng.random()));
}
