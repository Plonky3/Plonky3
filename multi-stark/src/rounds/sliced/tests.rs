use alloc::vec;
use alloc::vec::Vec;

use p3_binary_field::{Ghash128, TowerLevel};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use super::*;
use crate::packed_ext::PackedRepr;
use crate::rounds::StageCoupling;
use crate::rounds::subfield::tests::{
    first_challenge, later_rounds, link_coupling, no_lookups, with_stage_state, with_state,
};
use crate::zerocheck::backend_tests::{FixtureAir, Gf4, Instance, Tower, gf4, outside};

/// The smallest height whose residual half fills a word.
const SHORTEST: usize = 2 * SLICED_LANES;

fn packed_boolean_table(table: &Table<Tower>) -> Table<Tower> {
    let height = 1usize << table.num_variables();
    let words = (0..height / SLICED_LANES)
        .flat_map(|block| {
            (0..table.num_polys()).map(move |column| {
                (0..SLICED_LANES).fold(0u64, |word, lane| {
                    let row = block * SLICED_LANES + lane;
                    word | u64::from(table.column(column).value(row) == Tower::ONE) << lane
                })
            })
        })
        .collect();
    Table::from_packed_bits(
        RowMajorMatrix::new(words, table.num_polys()),
        table.num_variables(),
    )
}

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
fn packed_boolean_columns_match_dense_sliced_first_round() {
    let instance = Instance::honest(FixtureAir::Pair, SHORTEST, 0xBEEF);
    let dense = instance.main_table();
    let packed = packed_boolean_table(&dense);
    let airs = vec![&instance.air];
    let publics = vec![instance.public_values.as_slice()];

    let dense_rounds = with_stage_state(
        &airs,
        &publics,
        &[None],
        &[&dense],
        no_lookups(),
        |mut state, eq_suffix| {
            let sliced = state.round_poly_sliced::<Gf4, Tower>(eq_suffix);
            (sliced, state.round_poly(eq_suffix))
        },
    );
    let packed_rounds = with_stage_state(
        &airs,
        &publics,
        &[None],
        &[&packed],
        no_lookups(),
        |mut state, eq_suffix| {
            let sliced = state.round_poly_sliced::<Gf4, Tower>(eq_suffix);
            (sliced, state.round_poly(eq_suffix))
        },
    );

    assert_eq!(packed_rounds, dense_rounds);
}

#[test]
fn packed_tables_ingest_directly_in_merged_order() {
    let mut rng = SmallRng::seed_from_u64(0xD1CE);
    for height in [128, 256, 512] {
        let words = height / SLICED_LANES;
        let pair_words = (0..words * 3)
            .map(|index| match index % 3 {
                0 => 0,
                1 => u64::MAX,
                _ => rng.random(),
            })
            .collect();
        let linear_words = (0..words * 2)
            .map(|index| match index % 3 {
                0 => 0,
                1 => u64::MAX,
                _ => rng.random(),
            })
            .collect();
        let pair = Table::from_packed_bits(
            RowMajorMatrix::new(pair_words, 3),
            height.trailing_zeros() as usize,
        );
        let linear = Table::from_packed_bits(
            RowMajorMatrix::new(linear_words, 2),
            height.trailing_zeros() as usize,
        );
        let tables = [&pair, &linear];

        let expected = (0..words)
            .flat_map(|word| {
                tables.iter().flat_map(move |table| {
                    table.packed_bits().unwrap().values
                        [word * table.num_polys()..(word + 1) * table.num_polys()]
                        .iter()
                        .map(|&value| [value, 0])
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(
            direct_packed_cells(&tables, words),
            Some(expected.clone()),
            "packed sources should be eligible for direct ingestion at height {height}"
        );

        with_stage_state(
            &[&FixtureAir::Pair, &FixtureAir::Linear { scale: Tower::ONE }],
            &[&[], &[]],
            &[None, None],
            &[&pair, &linear],
            no_lookups(),
            |state, _| {
                let trace = state
                    .sliced_trace::<Gf4>()
                    .expect("packed Boolean tables should fit the sliced path");
                assert_eq!(trace.width, 5);
                assert_eq!(trace.cells, expected, "height {height}");
                assert!(trace.successors.iter().all(|planes| *planes == [0; 2]));
            },
        );
    }
}

#[test]
fn packed_direct_ingestion_matches_dense_rounds_for_multiple_tables() {
    for height in [128, 256, 512] {
        let instances = [
            Instance::honest(FixtureAir::Pair, height, 0xA11CE),
            Instance::honest(FixtureAir::Linear { scale: Tower::ONE }, height, 0xB0B),
        ];
        let dense = instances
            .iter()
            .map(Instance::main_table)
            .collect::<Vec<_>>();
        let packed = dense.iter().map(packed_boolean_table).collect::<Vec<_>>();
        let airs = instances
            .iter()
            .map(|instance| &instance.air)
            .collect::<Vec<_>>();
        let publics = instances
            .iter()
            .map(|instance| instance.public_values.as_slice())
            .collect::<Vec<_>>();

        let run = |main: &[Table<Tower>]| {
            with_stage_state(
                &airs,
                &publics,
                &[None, None],
                &main.iter().collect::<Vec<_>>(),
                no_lookups(),
                |mut state, eq_suffix| {
                    (
                        state
                            .round_poly_sliced::<Gf4, Tower>(eq_suffix)
                            .expect("stage should fit the sliced path"),
                        state.round_poly(eq_suffix),
                    )
                },
            )
        };

        assert_eq!(run(&packed), run(&dense), "height {height}");
    }
}

#[test]
fn packed_direct_ingestion_rejects_a_word_count_mismatch() {
    let table = packed_boolean_table(&Instance::honest(FixtureAir::Pair, 128, 0xBAD).main_table());
    assert!(direct_packed_cells(&[&table], 3).is_none());
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

/// Which kernels a stage runs once its sliced rounds are spent.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Boundary {
    /// Every residual column is written out, then read by the round and the fold.
    Unsliced,
    /// The round and the fold each read the planes.
    OnPlanes,
}

/// Every round polynomial and the openings, with the sliced rounds accumulated in `R`.
///
/// Also returns how many rounds ran on the planes, how many residual rows the stage left them
/// with, and whether the round that spent them read the planes.
fn sliced_rounds<R>(
    instances: &[Instance],
    boundary: Boundary,
) -> (Rounds, usize, Option<usize>, bool)
where
    R: Field + From<Tower> + p3_field::Algebra<Tower>,
    Tower: From<R>,
    R::Packing: p3_field::Algebra<Tower>,
    FixtureAir: for<'b> Air<SlicedFolder<'b, Tower, Gf4, R>>
        + for<'b> Air<crate::folder::MultilinearFolder<'b, Tower, R, R>>
        + for<'b> Air<crate::folder::InteractionMultilinearFolder<'b, Tower, R, R>>
        + for<'b> Air<
            crate::folder::MultilinearFolder<'b, Tower, PackedRepr<Tower, R>, PackedRepr<Tower, R>>,
        > + for<'b> Air<
            crate::folder::InteractionMultilinearFolder<
                'b,
                Tower,
                PackedRepr<Tower, R>,
                PackedRepr<Tower, R>,
            >,
        >,
{
    with_state(instances, no_lookups(), |mut state, eq_suffix| {
        let first = state
            .round_poly_sliced::<Gf4, R>(eq_suffix)
            .expect("the stage is sliced");
        let mut state = state.fold_sliced::<R>(challenge(0));
        let tau = state.tau.as_slice().to_vec();
        let mut on_planes = 1;
        let mut residual_rows = None;
        let mut read_planes = false;
        let mut round_polys = vec![first];
        for round in 1..tau.len() {
            let eq_suffix = Poly::new_from_point(&tau[round + 1..], Tower::ONE);
            let round_poly = if let Some(round_poly) = state.round_poly_sliced::<Gf4>(&eq_suffix) {
                on_planes += 1;
                round_poly
            } else {
                let lazy = (boundary == Boundary::OnPlanes)
                    .then(|| state.round_poly_boundary::<Gf4>(&eq_suffix))
                    .flatten();
                if let Some(round_poly) = lazy {
                    read_planes = true;
                    round_poly
                } else {
                    state.unslice::<Gf4>();
                    if residual_rows.is_none() {
                        residual_rows = match &state.columns {
                            ExtColumns::Scalar(columns) => Some(columns[0].as_slice().len()),
                            _ => None,
                        };
                    }
                    state.round_poly_repr(&eq_suffix)
                }
            };
            round_polys.push(round_poly);
            let on_planes_fold =
                boundary == Boundary::OnPlanes && state.fold_boundary::<Gf4>(challenge(round));
            if on_planes_fold {
                if residual_rows.is_none() {
                    residual_rows = match &state.columns {
                        ExtColumns::Scalar(columns) => Some(columns[0].as_slice().len() * 2),
                        _ => None,
                    };
                }
            } else if !state.fold_sliced(challenge(round)) {
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
        (
            (round_polys, openings),
            on_planes,
            residual_rows,
            read_planes,
        )
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
    for (height, expected_on_planes) in [
        (SHORTEST, 1),
        (2 * SHORTEST, 2),
        (4 * SHORTEST, 3),
        (64 * SHORTEST, 3),
    ] {
        let instances = [
            Instance::honest(FixtureAir::Gate { scale: gf4(3) }, height, 20),
            Instance::honest(FixtureAir::Pair, height, 21),
        ];
        let generic = generic_rounds(&instances);
        let (tower, tower_on_planes, tower_residual_rows, _) =
            sliced_rounds::<Tower>(&instances, Boundary::Unsliced);
        assert_eq!(tower_on_planes, expected_on_planes, "{height} rows");
        assert_eq!(
            tower_residual_rows,
            Some(height >> expected_on_planes),
            "{height} rows"
        );
        assert_eq!(tower, generic, "{height} rows, tower");
        let (poly_basis, on_planes, poly_residual_rows, _) =
            sliced_rounds::<Ghash128>(&instances, Boundary::Unsliced);
        assert_eq!(on_planes, expected_on_planes, "{height} rows");
        assert_eq!(
            poly_residual_rows,
            Some(height >> expected_on_planes),
            "{height} rows"
        );
        assert_eq!(poly_basis, generic, "{height} rows, polynomial basis");
    }
}

#[test]
fn a_boundary_round_and_fold_on_the_planes_match_the_unsliced_kernels() {
    // A word pair of residual rows is the shortest the planes can serve a boundary round, and
    // several pairs reuse one tile, join the accumulators, and put the high half's successor
    // word short of the last.
    for (height, on_planes) in [
        (32 * SHORTEST, true),
        (16 * SHORTEST, true),
        (8 * SHORTEST, true),
        (4 * SHORTEST, false),
    ] {
        let instances = [
            Instance::honest(FixtureAir::Gate { scale: gf4(3) }, height, 25),
            Instance::honest(FixtureAir::Pair, height, 26),
        ];
        for name in ["tower", "polynomial basis"] {
            let (unsliced, planes) = if name == "tower" {
                (
                    sliced_rounds::<Tower>(&instances, Boundary::Unsliced),
                    sliced_rounds::<Tower>(&instances, Boundary::OnPlanes),
                )
            } else {
                (
                    sliced_rounds::<Ghash128>(&instances, Boundary::Unsliced),
                    sliced_rounds::<Ghash128>(&instances, Boundary::OnPlanes),
                )
            };
            assert_eq!(planes.3, on_planes, "{height} rows, {name}");
            assert_eq!(planes.0, unsliced.0, "{height} rows, {name}");
            assert_eq!(planes.2, unsliced.2, "{height} rows, {name}");
        }
    }
}

#[test]
fn the_top_lane_mask_is_the_transpose_of_the_top_lane() {
    let mut rng = SmallRng::seed_from_u64(27);
    for corners in 1..=8 {
        let words = (0..corners).map(|_| rng.random()).collect::<Vec<u64>>();
        assert_eq!(
            top_lane_mask(&words),
            lane_masks(&words)[SLICED_LANES - 1],
            "{corners} corners"
        );
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
