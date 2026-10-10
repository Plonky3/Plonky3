//! Boolean tensor rounds must agree with the generic cubic-extension kernels.

use alloc::borrow::Cow;
use alloc::collections::BTreeMap;

use p3_air::{AirBuilder, WindowAccess};
use p3_binary_field::{BinaryChallenger, Poly64, Poly192, TowerLevel};
use p3_challenger::CanSample;
use p3_keccak::Keccak256Hash;
use p3_matrix::dense::RowMajorMatrix;
use p3_multilinear_util::point::Point;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use super::*;
use crate::backend::{BooleanTensorBackend, GenericBackend, ZerocheckBackend};
use crate::lookup::LookupRuntime;
use crate::rounds::{Stage, StageCoupling};
use crate::zerocheck::{AirZerocheck, get_air_profile};

type F = Poly64;
type EF = Poly192;

#[derive(Clone, Copy)]
struct BooleanAir {
    constant: F,
    cubic: bool,
    successor: bool,
    periodic: bool,
    preprocessed: bool,
}

impl Default for BooleanAir {
    fn default() -> Self {
        Self {
            constant: F::ONE,
            cubic: false,
            successor: false,
            periodic: false,
            preprocessed: false,
        }
    }
}

impl BaseAir<F> for BooleanAir {
    fn width(&self) -> usize {
        3
    }
    fn num_public_values(&self) -> usize {
        1
    }
    fn preprocessed_width(&self) -> usize {
        usize::from(self.preprocessed)
    }
    fn preprocessed_next_row_columns(&self) -> Vec<usize> {
        vec![]
    }
    fn main_next_row_columns(&self) -> Vec<usize> {
        if self.successor { vec![0] } else { vec![] }
    }
    fn num_periodic_columns(&self) -> usize {
        usize::from(self.periodic)
    }
    fn periodic_columns(&self) -> Cow<'_, [Vec<F>]> {
        Cow::Owned(if self.periodic {
            vec![vec![F::ZERO, F::ONE]]
        } else {
            vec![]
        })
    }
}

impl<AB: AirBuilder<F = F>> Air<AB> for BooleanAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.current_slice();
        let mut product = row[0] * row[1];
        if self.cubic {
            product *= row[0];
        }
        builder.assert_eq(row[2], product * self.constant);
        let public = builder.public_values()[0];
        builder.when_first_row().assert_eq(row[0], public);
        if self.successor {
            builder.assert_bool(main.next_slice()[0]);
        }
        if self.periodic {
            builder.assert_bool(builder.periodic_values()[0]);
        }
        if self.preprocessed {
            builder.assert_eq(builder.preprocessed().current_slice()[0], row[0]);
        }
    }
}

fn table(num_vars: usize, valid: bool) -> Table<F> {
    let mut rng = SmallRng::seed_from_u64(37);
    let words = (0..1 << (num_vars - LANE_VARIABLES))
        .flat_map(|_| {
            let a: u64 = rng.random();
            let b: u64 = rng.random();
            [a, b, if valid { a & b } else { rng.random() }]
        })
        .collect();
    Table::from_packed_bits(RowMajorMatrix::new(words, 3), num_vars)
}

fn with_state<T>(
    air: &BooleanAir,
    table: &Table<F>,
    public: F,
    rounds: usize,
    body: impl FnOnce(RoundStateBase<'_, '_, BooleanAir, F, EF>) -> T,
) -> T {
    let publics = [public];
    with_tables(&[air], &[table], &[None], &[&publics], rounds, body)
}

fn with_tables<T>(
    airs: &[&BooleanAir],
    tables: &[&Table<F>],
    preprocessed: &[Option<&Table<F>>],
    publics: &[&[F]],
    rounds: usize,
    body: impl FnOnce(RoundStateBase<'_, '_, BooleanAir, F, EF>) -> T,
) -> T {
    let stage = Stage::new(
        airs.to_vec(),
        publics.to_vec(),
        (0..airs.len()).collect(),
        preprocessed.to_vec(),
        tables.to_vec(),
        airs.iter()
            .map(|air| get_air_profile::<F, EF, _>(*air))
            .collect(),
        StageCoupling::new(BTreeMap::new(), BTreeMap::new(), vec![]),
    );
    let mut rng = SmallRng::seed_from_u64(83);
    let tau = Point::rand(&mut rng, stage.num_vars);
    body(RoundStateBase::new(
        stage,
        rng.random(),
        rng.random(),
        (0..airs.len()).map(|_| rng.random()).collect(),
        tau,
        rounds,
    ))
}

fn rounds_and_openings(
    mut state: RoundStateBase<'_, '_, BooleanAir, F, EF>,
    challenges: &[EF],
    tensor: bool,
) -> (Vec<Vec<EF>>, Vec<EF>) {
    let bound_columns: Vec<_> = state
        .slots
        .iter()
        .flat_map(|slot| {
            core::iter::once(state.tables[slot.stage_index])
                .chain(state.preprocessed[slot.stage_index])
                .flat_map(|table| table.columns())
                .map(|column| {
                    let mut poly = Poly::new(
                        (0..state.num_evals())
                            .map(|row| EF::from(column.value(row)))
                            .collect(),
                    );
                    for &r in &challenges[..4] {
                        poly.fix_prefix_var_mut(r);
                    }
                    poly
                })
        })
        .collect();
    let eq = Poly::new_from_point(&state.tau.as_slice()[1..], EF::ONE);
    let first = if tensor {
        let first = state
            .round_poly_boolean_tensor()
            .expect("Boolean cubic stage is eligible");
        assert!(state.has_sliced_tensor());
        assert_eq!(state.sliced.as_ref().unwrap().trace.successors.len(), 0);
        first
    } else {
        state.round_poly(&eq)
    };
    let mut polys = vec![first];
    let mut state = if tensor {
        state.fold_sliced::<EF>(challenges[0])
    } else {
        state.fold(challenges[0])
    };
    for (round, &challenge) in challenges.iter().enumerate().skip(1) {
        let eq = Poly::new_from_point(&state.tau.as_slice()[round + 1..], EF::ONE);
        let poly = if tensor && round < 4 {
            assert!(state.has_sliced_tensor());
            state
                .round_poly_tensor()
                .expect("tensor retains four rounds")
        } else {
            state.round_poly(&eq)
        };
        polys.push(poly);
        if tensor && round < 3 {
            assert!(state.fold_sliced(challenge));
        } else if tensor && round == 3 {
            assert!(state.fold_boolean_boundary(challenge));
            assert!(!state.has_sliced_tensor());
            assert_eq!(state.round, 4);
            assert_eq!(state.num_evals(), 1 << (challenges.len() - 4));
            assert!(matches!(state.columns, ExtColumns::Packed(_)));
            let actual: Vec<_> = state
                .columns
                .as_packed()
                .iter()
                .map(|column| column.unpack::<F, EF>())
                .collect();
            assert_eq!(
                actual, bound_columns,
                "all four challenges must reach every materialized column"
            );
        } else {
            state.fold(challenge);
        }
    }
    let openings = state
        .into_openings()
        .into_iter()
        .enumerate()
        .flat_map(|(expected, (index, opening))| {
            assert_eq!(index, expected);
            assert!(opening.next.is_empty());
            assert!(opening.preprocessed_next.is_empty());
            opening.local.into_iter().chain(opening.preprocessed_local)
        })
        .collect();
    (polys, openings)
}

#[test]
fn boolean_tensor_merges_main_and_preprocessed_tables() {
    let air = BooleanAir {
        preprocessed: true,
        ..BooleanAir::default()
    };
    let main = table(10, true);
    let preprocessed = Table::from_packed_bits(
        RowMajorMatrix::new(
            main.packed_bits()
                .unwrap()
                .values
                .as_chunks::<3>()
                .0
                .iter()
                .map(|row| row[0])
                .collect(),
            1,
        ),
        10,
    );
    let publics = [main.column(0).value(0)];
    let mut rng = SmallRng::seed_from_u64(15);
    let challenges: Vec<EF> = (0..10).map(|_| rng.random()).collect();
    for count in [1, 2] {
        let run = |tensor| {
            with_tables(
                &vec![&air; count],
                &vec![&main; count],
                &vec![Some(&preprocessed); count],
                &vec![publics.as_slice(); count],
                3,
                |state| rounds_and_openings(state, &challenges, tensor),
            )
        };
        assert_eq!(run(true), run(false));
    }
}

#[test]
fn boolean_tensor_cubic_rounds_and_packed_handoff_match_generic() {
    let air = BooleanAir::default();
    for num_vars in [10, 11] {
        for valid in [true, false] {
            let table = table(num_vars, valid);
            let public = table.column(0).value(0);
            let mut rng = SmallRng::seed_from_u64(91);
            let mut challenges: Vec<EF> = (0..num_vars).map(|_| rng.random()).collect();
            let node = EF::from(F::from_repr(2));
            for prefix in [
                [EF::ZERO, EF::ONE, node, EF::ZERO],
                [EF::ONE, node, EF::ZERO, EF::ONE],
                [node, EF::ZERO, EF::ONE, node],
                [rng.random(), rng.random(), rng.random(), rng.random()],
            ] {
                challenges[..4].copy_from_slice(&prefix);
                let generic = with_state(&air, &table, public, 3, |state| {
                    rounds_and_openings(state, &challenges, false)
                });
                let tensor = with_state(&air, &table, public, 3, |state| {
                    rounds_and_openings(state, &challenges, true)
                });
                assert_eq!(tensor, generic, "{num_vars} variables, valid={valid}");
            }
        }
    }
}

#[test]
fn boolean_tensor_fallback_is_transactional() {
    let air = BooleanAir::default();
    let packed = table(10, true);
    let dense = Table::new(RowMajorMatrix::new(
        (0..3)
            .flat_map(|col| {
                let packed = &packed;
                (0..1 << 10).map(move |row| packed.column(col).value(row))
            })
            .collect(),
        1 << 10,
    ));
    let short = table(9, true);
    for (air, table, public, rounds) in [
        (air, &short, F::ONE, 3),
        (air, &dense, F::ONE, 3),
        (air, &packed, F::from_repr(2), 3),
        (air, &packed, F::ONE, 0),
        (air, &packed, F::ONE, 2),
        (air, &packed, F::ONE, 4),
        (
            BooleanAir {
                constant: F::from_repr(2),
                ..air
            },
            &packed,
            F::ONE,
            3,
        ),
        (BooleanAir { cubic: true, ..air }, &packed, F::ONE, 3),
        (
            BooleanAir {
                successor: true,
                ..air
            },
            &packed,
            F::ONE,
            3,
        ),
        (
            BooleanAir {
                periodic: true,
                ..air
            },
            &packed,
            F::ONE,
            3,
        ),
    ] {
        let generic = with_state(&air, table, public, rounds, |mut state| {
            let eq = Poly::new_from_point(&state.tau.as_slice()[1..], EF::ONE);
            state.round_poly(&eq)
        });
        let fallback = with_state(&air, table, public, rounds, |mut state| {
            assert!(state.round_poly_boolean_tensor().is_none());
            assert!(!state.is_sliced());
            let eq = Poly::new_from_point(&state.tau.as_slice()[1..], EF::ONE);
            state.round_poly(&eq)
        });
        assert_eq!(fallback, generic);
    }
}

fn transcript<B: ZerocheckBackend<F, EF, BooleanAir>>(
    air: &BooleanAir,
    table: &Table<F>,
) -> (Vec<u8>, F) {
    let airs = [air];
    let public = [table.column(0).value(0)];
    let mut challenger = BinaryChallenger::<F, _>::from_hasher(Vec::new(), Keccak256Hash);
    let (proof, point) = AirZerocheck::new(&airs, 0).prove_with_lookup::<F, EF, B, _>(
        &[None],
        &[table],
        &[&public],
        LookupRuntime::Inactive,
        3,
        &mut challenger,
    );
    let bytes = postcard::to_allocvec(&(
        &proof.sumcheck,
        &proof.local,
        &proof.next,
        &proof.preprocessed_local,
        &proof.preprocessed_next,
        point.as_slice(),
    ))
    .unwrap();
    (bytes, CanSample::<F>::sample(&mut challenger))
}

#[test]
fn boolean_tensor_cubic_transcript_matches_generic() {
    let table = table(10, true);
    for air in [
        BooleanAir::default(),
        BooleanAir {
            constant: F::from_repr(2),
            ..BooleanAir::default()
        },
    ] {
        assert_eq!(
            transcript::<BooleanTensorBackend>(&air, &table),
            transcript::<GenericBackend>(&air, &table)
        );
    }
}

#[test]
fn boolean_tensor_rejects_prime_fields_and_lookup_stages() {
    use p3_baby_bear::BabyBear;

    use crate::rounds::subfield::tests::{link_coupling, with_state as with_lookup_state};
    use crate::zerocheck::backend_tests::{FixtureAir, Instance};

    struct PrimeAir;
    impl BaseAir<BabyBear> for PrimeAir {
        fn width(&self) -> usize {
            1
        }
        fn main_next_row_columns(&self) -> Vec<usize> {
            vec![]
        }
    }
    impl<AB: AirBuilder<F = BabyBear>> Air<AB> for PrimeAir {
        fn eval(&self, builder: &mut AB) {
            builder.assert_bool(builder.main().current_slice()[0]);
        }
    }
    let table = Table::from_packed_bits(RowMajorMatrix::new(vec![0; 16], 1), 10);
    let stage = Stage::new(
        vec![&PrimeAir],
        vec![&[]],
        vec![0],
        vec![None],
        vec![&table],
        vec![get_air_profile::<BabyBear, BabyBear, _>(&PrimeAir)],
        StageCoupling::new(BTreeMap::new(), BTreeMap::new(), vec![]),
    );
    let mut state = RoundStateBase::new(
        stage,
        BabyBear::ONE,
        BabyBear::ONE,
        vec![BabyBear::ONE],
        Point::new(vec![BabyBear::ONE; 10]),
        3,
    );
    assert!(state.round_poly_boolean_tensor().is_none());
    assert!(!state.is_sliced());

    with_lookup_state(
        &[Instance::honest(FixtureAir::Link, 1024, 49)],
        link_coupling(),
        |mut state, _| {
            assert!(state.round_poly_boolean_tensor().is_none());
            assert!(!state.is_sliced());
        },
    );
}
