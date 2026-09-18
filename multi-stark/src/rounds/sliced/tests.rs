use alloc::vec;
use alloc::vec::Vec;

use p3_binary_field::{Ghash128, TowerLevel};
use p3_field::{Field, PrimeCharacteristicRing};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use super::*;
use crate::rounds::StageCoupling;
use crate::rounds::subfield::tests::{
    first_challenge, later_rounds, link_coupling, no_lookups, with_state,
};
use crate::zerocheck::backend_tests::{FixtureAir, Gf4, Instance, Tower, gf4, outside};

/// The smallest height whose residual half fills a word.
const SHORTEST: usize = 2 * SLICED_LANES;

/// The sliced first round polynomial, accumulated in the tower and in the polynomial basis,
/// beside the generic kernel's.
fn first_rounds(
    instances: &[Instance],
    coupling: impl Fn() -> StageCoupling<Tower>,
) -> [Option<Vec<Tower>>; 3] {
    let tower = with_state(instances, coupling(), |mut state, eq_suffix| {
        state.round_poly_sliced::<Gf4, Tower>(eq_suffix)
    });
    with_state(instances, coupling(), |mut state, eq_suffix| {
        let poly_basis = state.round_poly_sliced::<Gf4, Ghash128>(eq_suffix);
        [tower, poly_basis, Some(state.round_poly(eq_suffix))]
    })
}

#[test]
fn a_fitting_stage_is_sliced_and_matches_the_generic_kernel() {
    for height in [SHORTEST, 4 * SHORTEST, 32 * SHORTEST] {
        for (name, instances) in [
            (
                "gate",
                vec![Instance::honest(
                    FixtureAir::Gate { scale: gf4(2) },
                    height,
                    1,
                )],
            ),
            ("pair", vec![Instance::honest(FixtureAir::Pair, height, 2)]),
            (
                "gate and pair",
                vec![
                    Instance::honest(FixtureAir::Gate { scale: Tower::ONE }, height, 3),
                    Instance::honest(FixtureAir::Pair, height, 4),
                ],
            ),
            (
                "periodic",
                vec![Instance::honest(
                    FixtureAir::Periodic {
                        period: [gf4(2), gf4(3)],
                    },
                    height,
                    5,
                )],
            ),
        ] {
            let [tower, poly_basis, generic] = first_rounds(&instances, no_lookups);
            assert!(tower.is_some(), "{name} at {height} rows");
            assert_eq!(tower, generic, "{name} at {height} rows, tower sums");
            assert_eq!(
                poly_basis, generic,
                "{name} at {height} rows, polynomial-basis sums"
            );
        }
    }
}

#[test]
fn a_stage_too_short_to_fill_a_word_is_not_sliced() {
    let instances = [Instance::honest(
        FixtureAir::Gate { scale: Tower::ONE },
        SHORTEST / 2,
        6,
    )];
    let [tower, ..] = first_rounds(&instances, no_lookups);
    assert_eq!(tower, None);
}

#[test]
fn each_misfit_is_not_sliced() {
    let gate = |seed| Instance::honest(FixtureAir::Gate { scale: Tower::ONE }, SHORTEST, seed);
    let mut cell = gate(7);
    // Column e of the last row, which only the last word of its column reads.
    cell.main.values[4 * (SHORTEST - 1) + 3] = outside();
    let mut preprocessed = gate(8);
    preprocessed
        .preprocessed
        .as_mut()
        .expect("the gate AIR declares a preprocessed column")
        .values[SHORTEST - 1] = outside();
    let mut public = gate(9);
    public.public_values[1] = outside();
    let periodic = Instance::honest(
        FixtureAir::Periodic {
            period: [gf4(2), outside()],
        },
        SHORTEST,
        10,
    );
    let constant = Instance::honest(
        FixtureAir::Gate {
            scale: Tower::from_repr(5),
        },
        SHORTEST,
        11,
    );
    let quartic = Instance::honest(FixtureAir::Quartic, SHORTEST, 12);

    for (name, instance) in [
        ("cell", cell),
        ("preprocessed cell", preprocessed),
        ("public value", public),
        ("periodic value", periodic),
        ("constant", constant),
        ("interpolation node", quartic),
    ] {
        let [tower, poly_basis, _] = first_rounds(&[instance], no_lookups);
        assert_eq!(tower, None, "{name}");
        assert_eq!(poly_basis, None, "{name}");
    }
}

#[test]
fn a_lookup_stage_is_not_sliced() {
    let instances = [Instance::honest(FixtureAir::Link, SHORTEST, 13)];
    let [tower, ..] = first_rounds(&instances, link_coupling);
    assert_eq!(tower, None);
}

#[test]
fn the_planes_hold_every_cell_and_its_repeat_last_successor() {
    let height = 4 * SHORTEST;
    let mut rng = SmallRng::seed_from_u64(23);
    let mut instance = Instance::honest(FixtureAir::Gate { scale: Tower::ONE }, height, 23);
    // Every element of GF(4), so both planes of both kinds of word carry data.
    for value in &mut instance.main.values {
        *value = gf4(rng.random_range(0..4));
    }
    with_state(&[instance], no_lookups(), |state, _| {
        let trace = state
            .sliced_trace::<Gf4>()
            .expect("every cell lies in GF(4)");
        let columns = state.tables[0]
            .iter_polys()
            .chain(
                state.preprocessed[0]
                    .into_iter()
                    .flat_map(Table::iter_polys),
            )
            .chain(state.periodic[0].iter().flat_map(Table::iter_polys))
            .collect::<Vec<_>>();
        assert_eq!(trace.width, columns.len());
        let successor_columns = next_row_runs(&state.slots)
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let cell = |planes: &[[u64; 2]], column: usize, row: usize| {
            let [low, high] = planes[(row / SLICED_LANES) * trace.width + column];
            let bit = |plane: u64| (plane >> (row % SLICED_LANES)) & 1 == 1;
            Tower::from(Gf4::from_bool(bit(low)) + Gf4::from_bool(bit(high)) * Gf4::GENERATOR)
        };
        for (index, column) in columns.iter().enumerate() {
            for row in 0..height {
                assert_eq!(
                    cell(&trace.cells, index, row),
                    column[row],
                    "column {index}"
                );
                let successor = if successor_columns.contains(&index) {
                    column[(row + 1).min(height - 1)]
                } else {
                    Tower::ZERO
                };
                assert_eq!(
                    cell(&trace.successors, index, row),
                    successor,
                    "column {index}"
                );
            }
        }
    });
}

#[test]
fn a_sliced_first_round_records_the_fit_for_the_fold() {
    let instances = [Instance::honest(
        FixtureAir::Gate { scale: Tower::ONE },
        SHORTEST,
        24,
    )];
    with_state(&instances, no_lookups(), |mut state, eq_suffix| {
        assert!(state.round_poly_sliced::<Gf4, Tower>(eq_suffix).is_some());
        assert!(state.fits_subfield());
        assert!(state.is_sliced());
    });
}

/// Every round polynomial of a stage, then its openings.
type Rounds = (Vec<Vec<Tower>>, Vec<[Vec<Tower>; 4]>);

/// The challenge bound at `round`, the one [`later_rounds`] binds.
fn challenge(round: usize) -> Tower {
    if round == 0 {
        first_challenge()
    } else {
        first_challenge() + Tower::from_repr(round as u128)
    }
}

/// Every round polynomial and the openings, with the sliced rounds accumulated in `R`.
///
/// Also returns how many rounds ran on the planes.
fn sliced_rounds<R>(instances: &[Instance]) -> (Rounds, usize)
where
    R: Field + From<Tower> + p3_field::Algebra<Tower>,
    Tower: From<R>,
    FixtureAir: for<'b> Air<SlicedFolder<'b, Tower, Gf4, R>>
        + for<'b> Air<crate::folder::MultilinearFolder<'b, Tower, R, R>>
        + for<'b> Air<crate::folder::InteractionMultilinearFolder<'b, Tower, R, R>>,
{
    with_state(instances, no_lookups(), |mut state, eq_suffix| {
        let first = state
            .round_poly_sliced::<Gf4, R>(eq_suffix)
            .expect("the stage is sliced");
        let mut state = state.fold_sliced::<R>(challenge(0));
        let tau = state.tau.as_slice().to_vec();
        let mut on_planes = 1;
        let mut round_polys = vec![first];
        for round in 1..tau.len() {
            let eq_suffix = Poly::new_from_point(&tau[round + 1..], Tower::ONE);
            let round_poly = state.round_poly_sliced::<Gf4>(&eq_suffix).map_or_else(
                || {
                    state.unslice::<Gf4>();
                    state.round_poly_repr(&eq_suffix)
                },
                |round_poly| {
                    on_planes += 1;
                    round_poly
                },
            );
            round_polys.push(round_poly);
            if !state.fold_sliced(challenge(round)) {
                state.fold_repr(challenge(round));
            }
        }
        let openings = state
            .into_openings()
            .into_iter()
            .map(|(_, opening)| {
                [
                    opening.local,
                    opening.next,
                    opening.preprocessed_local,
                    opening.preprocessed_next,
                ]
            })
            .collect();
        ((round_polys, openings), on_planes)
    })
}

/// Every round polynomial and the openings through the generic kernels.
fn generic_rounds(instances: &[Instance]) -> Rounds {
    with_state(instances, no_lookups(), |mut state, eq_suffix| {
        let first = state.round_poly(eq_suffix);
        let (mut round_polys, openings) = later_rounds(state.fold(challenge(0)));
        round_polys.insert(0, first);
        (round_polys, openings)
    })
}

#[test]
fn every_round_on_and_off_the_planes_matches_the_generic_kernel() {
    for (height, expected_on_planes) in [(SHORTEST, 1), (4 * SHORTEST, 3), (64 * SHORTEST, 3)] {
        let instances = [
            Instance::honest(FixtureAir::Gate { scale: gf4(3) }, height, 20),
            Instance::honest(FixtureAir::Pair, height, 21),
        ];
        let generic = generic_rounds(&instances);
        let (tower, tower_on_planes) = sliced_rounds::<Tower>(&instances);
        assert_eq!(tower_on_planes, expected_on_planes, "{height} rows");
        assert_eq!(tower, generic, "{height} rows, tower");
        let (poly_basis, on_planes) = sliced_rounds::<Ghash128>(&instances);
        assert_eq!(on_planes, expected_on_planes, "{height} rows");
        assert_eq!(poly_basis, generic, "{height} rows, polynomial basis");
    }
}

#[test]
fn lane_masks_transpose_the_corner_words() {
    let mut rng = SmallRng::seed_from_u64(22);
    for corners in 1..=8 {
        let words = (0..corners).map(|_| rng.random()).collect::<Vec<u64>>();
        let masks = lane_masks(&words);
        for (lane, &mask) in masks.iter().enumerate() {
            let expected = words.iter().enumerate().fold(0_u8, |mask, (i, &word)| {
                mask | ((((word >> lane) & 1) as u8) << i)
            });
            assert_eq!(mask, expected, "lane {lane} of {corners} corners");
        }
    }
}
