use alloc::vec;
use alloc::vec::Vec;

use p3_air::BaseAir;
use p3_binary_field::{Ghash128, TowerLevel};
use p3_field::{Field, HasSubfield, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;
use p3_multilinear_util::point::Point;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use super::*;
use crate::packed_ext::PackedRepr;
use crate::rounds::subfield::tests::{
    first_challenge, later_rounds, link_coupling, no_lookups, with_stage_state, with_state,
};
use crate::rounds::{AirSlot, StageCoupling};
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
fn plane_fold_five_challenges_matches_explicit_corner_sum() {
    let prefix = [
        Tower::from_repr(0x11),
        Tower::from_repr(0x22),
        Tower::from_repr(0x33),
        Tower::from_repr(0x44),
        Tower::from_repr(0x55),
    ];
    for num_vars in [11, 12] {
        let height = 1 << num_vars;
        for (name, mut instance) in [
            ("pair", Instance::honest(FixtureAir::Pair, height, 0xF01D)),
            (
                "merged",
                Instance::honest(FixtureAir::QuadraticInputs, height, 0xF01E),
            ),
        ] {
            let main_width = instance.air.width();
            for (row, values) in instance.main.values.chunks_mut(main_width).enumerate() {
                for (column, value) in values.iter_mut().enumerate() {
                    *value = gf4(plane_fold_fixture_bits(row, column, main_width));
                }
            }
            if let Some(preprocessed) = instance.preprocessed.as_mut() {
                for (row, value) in preprocessed.values.iter_mut().enumerate() {
                    *value = gf4(plane_fold_fixture_bits(row, 1, main_width));
                }
            }
            with_state(&[instance], no_lookups(), |mut state, _| {
                if name == "merged" {
                    let periodic = (0..height)
                        .map(|row| gf4(plane_fold_fixture_bits(row, 2, main_width)))
                        .collect();
                    state.periodic[0] = Some(Table::new(RowMajorMatrix::new(periodic, height)));
                }
                let trace = state
                    .sliced_trace::<Gf4>()
                    .expect("the GF(4) trace should fit the sliced path");
                let columns = state.tables[0]
                    .iter_polys()
                    .chain(
                        state.preprocessed[0]
                            .into_iter()
                            .flat_map(Table::iter_polys),
                    )
                    .chain(state.periodic[0].iter().flat_map(Table::iter_polys))
                    .collect::<Vec<_>>();
                assert_eq!(trace.width, columns.len(), "{name} width");
                let fold =
                    PlaneFold::<Tower, MAX_PLANE_FOLD_CORNERS>::new::<Gf4, Tower>(&trace, &prefix);
                let remaining = num_vars - prefix.len();
                let words = 1 << remaining.saturating_sub(LANE_VARIABLES);
                let mut actual = vec![Tower::ZERO; SLICED_LANES];
                for (column, column_values) in columns.iter().enumerate().take(trace.width) {
                    for word in 0..words {
                        fold.fold_word(&trace.cells, column, word, &mut actual);
                        for row in 0..SLICED_LANES {
                            let mut expected = Tower::ZERO;
                            for corner in 0..1 << prefix.len() {
                                let mut weight = Tower::ONE;
                                for (i, &challenge) in prefix.iter().enumerate() {
                                    weight *= if (corner >> (prefix.len() - 1 - i)) & 1 == 0 {
                                        Tower::ONE - challenge
                                    } else {
                                        challenge
                                    };
                                }
                                expected += weight
                                    * column_values
                                        [(corner << remaining) + word * SLICED_LANES + row];
                            }
                            assert_eq!(
                                Tower::from(actual[row]),
                                expected,
                                "{name} column {column}"
                            );
                        }
                    }
                }
            });
        }
    }
}

fn plane_fold_fixture_bits(row: usize, column: usize, width: usize) -> usize {
    let mut mixed = 0xD1CE_BA5E_1234_5678u64
        .wrapping_add((row as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_add((column as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9))
        .wrapping_add(width as u64);
    mixed ^= mixed >> 30;
    mixed = mixed.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    mixed ^= mixed >> 27;
    mixed = mixed.wrapping_mul(0x94D0_49BB_1331_11EB);
    mixed ^= mixed >> 31;
    (mixed & 3) as usize
}

fn plane_fold_trace_fixture(
    num_vars: usize,
    width: usize,
    boolean: bool,
) -> (SlicedTrace, Vec<Tower>) {
    let height = 1 << num_vars;
    let words = height / SLICED_LANES;
    let mut values = vec![Tower::ZERO; height * width];
    let mut cells = vec![[0; 2]; words * width];
    for word in 0..words {
        for column in 0..width {
            for lane in 0..SLICED_LANES {
                let row = word * SLICED_LANES + lane;
                let bits = plane_fold_fixture_bits(row, column, width);
                let bits = if boolean { bits & 1 } else { bits };
                values[row * width + column] = gf4(bits);
                cells[word * width + column][0] |= u64::from(bits & 1 != 0) << lane;
                cells[word * width + column][1] |= u64::from(bits & 2 != 0) << lane;
            }
        }
    }
    (
        SlicedTrace {
            num_vars,
            width,
            cells,
            successors: vec![],
            boundary: vec![],
            rounds: 0,
        },
        values,
    )
}

#[test]
fn plane_fold_reference_covers_prefixes_widths_and_special_challenges() {
    let special = [
        Tower::ZERO,
        Tower::ONE,
        Tower::interpolation_node(2),
        Tower::from_repr(0x1234),
    ];
    for num_vars in [11, 12] {
        for width in [1, 3, 9] {
            let (_, values) = plane_fold_trace_fixture(num_vars, width, false);
            let remaining = num_vars - 5;
            let differs_after_reversal = (0usize..(1 << 5)).any(|corner| {
                let reversed = corner.reverse_bits() >> (usize::BITS as usize - 5);
                (0..1 << remaining).any(|suffix| {
                    (0..width).any(|column| {
                        values[((corner << remaining) | suffix) * width + column]
                            != values[((reversed << remaining) | suffix) * width + column]
                    })
                })
            });
            assert!(
                differs_after_reversal,
                "fixture must distinguish five-prefix reversal at n={num_vars}, width={width}"
            );
        }
    }
    for prefix_len in 3..=5 {
        for width in [1, 3, 9] {
            for boolean in [true, false] {
                let (trace, values) = plane_fold_trace_fixture(11, width, boolean);
                let remaining = trace.num_vars - prefix_len;
                let words = 1 << (remaining - LANE_VARIABLES);
                for special_index in 0..prefix_len {
                    for &special_challenge in &special {
                        let prefix = (0..prefix_len)
                            .map(|index| {
                                if index == special_index {
                                    special_challenge
                                } else {
                                    Tower::from_repr((0x100 + index) as u128)
                                }
                            })
                            .collect::<Vec<_>>();
                        let fold = PlaneFold::<Tower, MAX_PLANE_FOLD_CORNERS>::new::<Gf4, Tower>(
                            &trace, &prefix,
                        );
                        let mut actual = vec![Tower::ZERO; SLICED_LANES];
                        for column in 0..width {
                            for word in 0..words {
                                fold.fold_word(&trace.cells, column, word, &mut actual);
                                for row in 0..SLICED_LANES {
                                    let mut expected = Tower::ZERO;
                                    for corner in 0..1 << prefix_len {
                                        let mut weight = Tower::ONE;
                                        for (i, &challenge) in prefix.iter().enumerate() {
                                            weight *= if (corner >> (prefix_len - 1 - i)) & 1 == 0 {
                                                Tower::ONE - challenge
                                            } else {
                                                challenge
                                            };
                                        }
                                        expected += weight
                                            * values[((corner << remaining)
                                                + word * SLICED_LANES
                                                + row)
                                                * width
                                                + column];
                                    }
                                    assert_eq!(Tower::from(actual[row]), expected);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
#[should_panic(
    expected = "a plane fold's corner buffers must hold every corner of its bound prefix"
)]
fn a_default_plane_fold_refuses_the_delayed_five_challenge_prefix() {
    // Only the delayed boundary path's unslice gathers five challenges' worth of corners.
    let (trace, _) = plane_fold_trace_fixture(11, 1, true);
    let prefix = [Tower::from_repr(0x1234); MAX_PLANE_FOLD_ROUNDS];
    let _ = PlaneFold::<Tower>::new::<Gf4, Tower>(&trace, &prefix);
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

#[test]
fn tensor4_path_retains_all_entries_and_replays_cached_rounds() {
    let height = 1 << 10;
    let instances = [Instance::honest(FixtureAir::Pair, height, 0x007E_5001)];
    with_state(&instances, no_lookups(), |mut state, eq_suffix| {
        let _first = state
            .round_poly_sliced_with_strategy::<Gf4, Ghash128>(
                eq_suffix,
                SlicedStrategy::TensorBoundary,
            )
            .expect("eligible pair stage should build tensor4");
        assert!(state.has_sliced_tensor());
        let tensor = state
            .sliced
            .as_ref()
            .and_then(|columns| columns.tensor.as_ref())
            .expect("tensor cache should be installed");
        assert_eq!(tensor.depth, 4);
        assert_eq!(tensor.values.len(), 1);
        assert_eq!(tensor.values[0].len(), 81);
        assert!(state.sliced.as_ref().unwrap().challenges.is_empty());

        let mut state = state.fold_sliced::<Ghash128>(challenge(0));
        assert_eq!(state.round, 1);
        assert!(
            matches!(&state.columns, ExtColumns::Sliced(columns) if columns.challenges.len() == 1)
        );
        let tau = state.tau.as_slice().to_vec();
        let eq_suffix = Poly::new_from_point(&tau[2..], Tower::ONE);
        let round_one = state
            .round_poly_sliced::<Gf4>(&eq_suffix)
            .expect("tensor cache should serve round one");
        let replay = state
            .round_poly_sliced::<Gf4>(&eq_suffix)
            .expect("cached round should be replayable");
        assert_eq!(round_one, replay);
        assert!(state.has_sliced_tensor());

        assert!(state.fold_sliced(challenge(1)));
        assert_eq!(state.round, 2);
        assert!(
            matches!(&state.columns, ExtColumns::Sliced(columns) if columns.challenges.len() == 2)
        );
        let tau = state.tau.as_slice().to_vec();
        let eq_suffix = Poly::new_from_point(&tau[3..], Tower::ONE);
        let round_two = state
            .round_poly_sliced::<Gf4>(&eq_suffix)
            .expect("tensor cache should serve round two");
        assert_eq!(
            round_two,
            state.round_poly_sliced::<Gf4>(&eq_suffix).unwrap()
        );
        assert!(state.fold_sliced(challenge(2)));
        assert_eq!(state.round, 3);
        assert!(
            matches!(&state.columns, ExtColumns::Sliced(columns) if columns.challenges.len() == 3)
        );
        let tau = state.tau.as_slice().to_vec();
        let eq_suffix = Poly::new_from_point(&tau[4..], Tower::ONE);
        let round_three = state
            .round_poly_sliced::<Gf4>(&eq_suffix)
            .expect("tensor cache should serve round three");
        assert_eq!(
            round_three,
            state.round_poly_sliced::<Gf4>(&eq_suffix).unwrap()
        );
        assert!(state.has_sliced_tensor());
        assert!(state.fold_boundary::<Gf4>(challenge(3)));
        assert_eq!(state.round, 4);
        assert_eq!(state.num_evals(), height >> 4);
        assert!(!state.has_sliced_tensor());
    });
}

#[test]
fn late_boundary_keeps_planes_until_fifth_challenge() {
    for height in [1 << 11, 1 << 12] {
        let instances = [Instance::honest(
            FixtureAir::Pair,
            height,
            0x007E_50A0 + height as u64,
        )];
        with_state(&instances, no_lookups(), |mut state, eq_suffix| {
            state
                .round_poly_sliced_with_strategy::<Gf4, Ghash128>(
                    eq_suffix,
                    SlicedStrategy::TensorBoundaryLate,
                )
                .expect("late tensor4 should build");
            assert!(matches!(
                &state.sliced,
                Some(columns) if columns.late_boundary
                    && columns.tensor.is_some()
                    && columns.num_evals() == height
            ));

            let mut state = state.fold_sliced::<Ghash128>(challenge(0));
            assert!(
                matches!(&state.columns, ExtColumns::Sliced(columns) if columns.challenges.len() == 1 && columns.tensor.is_some())
            );
            for round in 1..=2 {
                let tau = state.tau.as_slice().to_vec();
                let suffix = Poly::new_from_point(&tau[round + 1..], Tower::ONE);
                assert!(state.round_poly_sliced::<Gf4>(&suffix).is_some());
                assert!(state.fold_sliced(challenge(round)));
            }
            assert!(
                matches!(&state.columns, ExtColumns::Sliced(columns) if columns.challenges.len() == 3 && columns.tensor.is_some())
            );

            let tau = state.tau.as_slice().to_vec();
            let suffix = Poly::new_from_point(&tau[4..], Tower::ONE);
            assert!(state.round_poly_sliced::<Gf4>(&suffix).is_some());
            assert!(state.fold_late_boundary::<Gf4>(challenge(3)));
            assert!(
                matches!(&state.columns, ExtColumns::Sliced(columns) if columns.challenges.len() == 4 && columns.late_boundary && columns.tensor.is_none() && columns.num_evals() == height >> 4)
            );

            let tau = state.tau.as_slice().to_vec();
            let suffix = Poly::new_from_point(&tau[5..], Tower::ONE);
            let round_four = state
                .round_poly_late_boundary::<Gf4>(&suffix)
                .expect("late plane evaluator should serve round four");
            assert_eq!(
                round_four,
                state
                    .round_poly_late_boundary::<Gf4>(&suffix)
                    .expect("late round should replay deterministically")
            );

            assert!(state.fold_late_boundary::<Gf4>(challenge(4)));
            assert!(
                matches!(&state.columns, ExtColumns::Scalar(columns) if columns.len() == 3 && columns.iter().all(|column| column.num_evals() == height >> 5))
            );
            assert_eq!(state.round, 5);

            for round in 5..tau.len() {
                let tau = state.tau.as_slice().to_vec();
                let suffix = Poly::new_from_point(&tau[round + 1..], Tower::ONE);
                state.round_poly_repr(&suffix);
                state.fold_repr(challenge(round));
            }
            assert!(!state.into_openings().is_empty());
        });

        assert_eq!(
            collect_late_boundary_rounds(
                &instances,
                [
                    challenge(0),
                    challenge(1),
                    challenge(2),
                    challenge(3),
                    challenge(4)
                ],
                true,
            ),
            collect_late_boundary_rounds(
                &instances,
                [
                    challenge(0),
                    challenge(1),
                    challenge(2),
                    challenge(3),
                    challenge(4)
                ],
                false,
            ),
            "late lifecycle must match incumbent through openings at {height}"
        );
    }
}

#[test]
fn late_boundary_marker_requires_the_n11_height_floor() {
    // The n11 of the late tests is `MIN_LATE_BOUNDARY_VARS`, the delayed path's floor.
    assert_eq!(MIN_LATE_BOUNDARY_VARS, 11);
    for (height, expected) in [(1 << 9, false), (1 << 10, false), (1 << 11, true)] {
        let instances = [Instance::honest(
            FixtureAir::Pair,
            height,
            0x007E_50B0 + height as u64,
        )];
        with_state(&instances, no_lookups(), |mut state, eq_suffix| {
            state
                .round_poly_sliced_with_strategy::<Gf4, Ghash128>(
                    eq_suffix,
                    SlicedStrategy::TensorBoundaryLate,
                )
                .expect("late strategy should retain the ordinary sliced fallback");
            assert_eq!(
                state
                    .sliced
                    .as_ref()
                    .is_some_and(|columns| columns.late_boundary),
                expected,
                "height {height}"
            );
            if height == 1 << 10 {
                assert!(
                    state.has_sliced_tensor(),
                    "n10 keeps ordinary tensor4 state"
                );
            }
        });
    }
}

#[test]
#[should_panic(expected = "a plane round needs a whole word pair")]
fn a_forced_late_round_below_the_height_floor_panics_instead_of_summing_no_rows() {
    // Four bound challenges leave 2^10 rows a single word, so round four has no word pair.
    let instances = [Instance::honest(FixtureAir::Pair, 1 << 10, 0x007E_50E0)];
    with_state(&instances, no_lookups(), |mut state, eq_suffix| {
        state
            .round_poly_sliced_with_strategy::<Gf4, Ghash128>(
                eq_suffix,
                SlicedStrategy::TensorBoundaryLate,
            )
            .expect("tensor4 should build at 2^10");
        let columns = state.sliced.as_mut().expect("the stage is on its planes");
        assert!(
            !columns.late_boundary,
            "the height floor withholds the marker"
        );
        columns.late_boundary = true;

        let mut state = state.fold_sliced::<Ghash128>(challenge(0));
        let tau = state.tau.as_slice().to_vec();
        for round in 1..=2 {
            let suffix = Poly::new_from_point(&tau[round + 1..], Tower::ONE);
            assert!(state.round_poly_sliced::<Gf4>(&suffix).is_some());
            assert!(state.fold_sliced(challenge(round)));
        }
        let suffix = Poly::new_from_point(&tau[4..], Tower::ONE);
        assert!(state.round_poly_sliced::<Gf4>(&suffix).is_some());
        assert!(state.fold_late_boundary::<Gf4>(challenge(3)));

        let suffix = Poly::new_from_point(&tau[5..], Tower::ONE);
        let _ = state.round_poly_late_boundary::<Gf4>(&suffix);
    });
}

#[test]
fn late_boundary_refusal_unslices_with_the_four_recorded_challenges() {
    let height = 1 << 11;
    let instances = [Instance::honest(FixtureAir::Pair, height, 0x007E_50B1)];
    let fallback = with_state(&instances, no_lookups(), |mut state, eq_suffix| {
        let first = state
            .round_poly_sliced_with_strategy::<Gf4, Ghash128>(
                eq_suffix,
                SlicedStrategy::TensorBoundaryLate,
            )
            .expect("late tensor4 should build");
        let mut round_polys = vec![first];
        let mut state = state.fold_sliced::<Ghash128>(challenge(0));
        for round in 1..=2 {
            let tau = state.tau.as_slice().to_vec();
            let suffix = Poly::new_from_point(&tau[round + 1..], Tower::ONE);
            round_polys.push(
                state
                    .round_poly_sliced::<Gf4>(&suffix)
                    .expect("tensor round should be evaluated before its fold"),
            );
            assert!(state.fold_sliced(challenge(round)));
        }
        let tau = state.tau.as_slice().to_vec();
        let suffix = Poly::new_from_point(&tau[4..], Tower::ONE);
        round_polys.push(
            state
                .round_poly_sliced::<Gf4>(&suffix)
                .expect("tensor round should be evaluated before its fold"),
        );
        assert!(state.fold_late_boundary::<Gf4>(challenge(3)));
        if let ExtColumns::Sliced(columns) = &mut state.columns {
            columns.late_boundary = false;
        } else {
            panic!("late refusal test must still hold planes");
        }
        let tau = state.tau.as_slice().to_vec();
        let suffix = Poly::new_from_point(&tau[5..], Tower::ONE);
        assert!(state.round_poly_sliced::<Gf4>(&suffix).is_none());
        assert!(state.round_poly_boundary::<Gf4>(&suffix).is_none());
        state.unslice::<Gf4>();
        assert!(
            matches!(&state.columns, ExtColumns::Scalar(columns) if columns.iter().all(|column| column.num_evals() == height >> 4))
        );
        for round in 4..tau.len() {
            let tau = state.tau.as_slice().to_vec();
            let suffix = Poly::new_from_point(&tau[round + 1..], Tower::ONE);
            round_polys.push(state.round_poly_repr(&suffix));
            state.fold_repr(challenge(round));
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
        (round_polys, openings)
    });
    assert_eq!(
        fallback,
        collect_late_boundary_rounds(
            &instances,
            [
                challenge(0),
                challenge(1),
                challenge(2),
                challenge(3),
                challenge(4)
            ],
            false,
        ),
        "forced refusal must preserve the incumbent transcript and openings"
    );
}

#[test]
fn late_boundary_rejection_gates_are_isolated_at_height_eleven() {
    let height = 1 << 11;
    let assert_no_late = |instances: &[Instance], coupling| {
        with_state(instances, coupling, |mut state, eq_suffix| {
            let _ = state.round_poly_sliced_with_strategy::<Gf4, Ghash128>(
                eq_suffix,
                SlicedStrategy::TensorBoundaryLate,
            );
            assert!(
                !state
                    .sliced
                    .as_ref()
                    .is_some_and(|columns| columns.late_boundary)
            );
        });
    };

    for sliced_rounds in [0, 1, 2, 4] {
        let instance =
            Instance::honest(FixtureAir::Pair, height, 0x007E_50C0 + sliced_rounds as u64);
        with_state(&[instance], no_lookups(), |mut state, eq_suffix| {
            state.sliced_rounds = sliced_rounds;
            let _ = state.round_poly_sliced_with_strategy::<Gf4, Ghash128>(
                eq_suffix,
                SlicedStrategy::TensorBoundaryLate,
            );
            assert!(
                !state
                    .sliced
                    .as_ref()
                    .is_some_and(|columns| columns.late_boundary)
            );
        });
    }
    assert_no_late(
        &[Instance::honest(
            FixtureAir::Periodic {
                period: [gf4(2), gf4(3)],
            },
            height,
            0x007E_50C5,
        )],
        no_lookups(),
    );
    assert_no_late(
        &[Instance::honest(
            FixtureAir::QuadraticSuccessor,
            height,
            0x007E_50C6,
        )],
        no_lookups(),
    );
    assert_no_late(
        &[Instance::honest(
            FixtureAir::QuadraticPreprocessedSuccessor,
            height,
            0x007E_50CA,
        )],
        no_lookups(),
    );
    assert_no_late(
        &[Instance::honest(FixtureAir::Link, height, 0x007E_50C7)],
        link_coupling(),
    );
    let mut outside_main = Instance::honest(FixtureAir::QuadraticInputs, height, 0x007E_50C8);
    outside_main.main.values[0] = outside();
    assert_no_late(&[outside_main], no_lookups());
    let mut outside_preprocessed =
        Instance::honest(FixtureAir::QuadraticInputs, height, 0x007E_50CB);
    outside_preprocessed
        .preprocessed
        .as_mut()
        .expect("quadratic inputs has fixed data")
        .values[0] = outside();
    assert_no_late(&[outside_preprocessed], no_lookups());
    let mut outside_public = Instance::honest(FixtureAir::QuadraticInputs, height, 0x007E_50CC);
    outside_public.public_values[0] = outside();
    assert_no_late(&[outside_public], no_lookups());
    let mut outside_periodic = Instance::honest(
        FixtureAir::QuadraticInputsOutsidePeriodic,
        height,
        0x007E_50C9,
    );
    outside_periodic
        .preprocessed
        .as_mut()
        .expect("quadratic inputs has fixed data")
        .values
        .fill(Tower::ZERO);
    assert!(<Tower as HasSubfield<Gf4>>::all_in_subfield(
        &outside_periodic.main.values
    ));
    assert!(<Tower as HasSubfield<Gf4>>::all_in_subfield(
        &outside_periodic
            .preprocessed
            .as_ref()
            .expect("quadratic inputs has fixed data")
            .values
    ));
    assert!(<Tower as HasSubfield<Gf4>>::all_in_subfield(
        &outside_periodic.public_values
    ));
    assert_eq!(outside_periodic.air.periodic_columns()[0][0], outside());
    assert_no_late(&[outside_periodic], no_lookups());
}

#[test]
fn tensor4_installs_at_two_eligible_activation_heights() {
    for (height, seed) in [(1 << 12, 0x007E_5022), (1 << 10, 0x007E_5023)] {
        let instances = [Instance::honest(FixtureAir::Pair, height, seed)];
        with_state(&instances, no_lookups(), |mut state, eq_suffix| {
            state
                .round_poly_sliced_with_strategy::<Gf4, Ghash128>(
                    eq_suffix,
                    SlicedStrategy::TensorBoundary,
                )
                .expect("eligible height should build tensor4");
            assert!(state.has_sliced_tensor());
        });
    }
}

#[test]
fn tensor4_evaluates_mixed_linear_and_quadratic_airs() {
    let height = 1 << 10;
    let mut linear = Instance::honest(
        FixtureAir::Linear { scale: Tower::ONE },
        height,
        0x007E_5002,
    );
    linear.main.values[0] = gf4(2);
    let instances = [
        linear,
        Instance::honest(FixtureAir::Pair, height, 0x007E_5003),
    ];
    let tensor = collect_tensor_rounds(&instances, true);
    let sequential = collect_tensor_rounds(&instances, false);
    assert_eq!(
        tensor, sequential,
        "mixed degree tensor4 rounds and openings"
    );
}

/// Collect the complete direct-state transcript for the incumbent tensor4 path or the delayed
/// plane path, using caller-supplied first five fold challenges.
fn collect_late_boundary_rounds(instances: &[Instance], prefix: [Tower; 5], late: bool) -> Rounds {
    collect_late_boundary_rounds_with_tau(instances, prefix, late, None, None)
}

fn collect_late_boundary_rounds_with_tau(
    instances: &[Instance],
    prefix: [Tower; 5],
    late: bool,
    tau4: Option<Tower>,
    tau5: Option<Tower>,
) -> Rounds {
    with_state(instances, no_lookups(), |mut state, _eq_suffix| {
        let mut tau = state.tau.as_slice().to_vec();
        if let Some(value) = tau4 {
            tau[4] = value;
        }
        if let Some(value) = tau5 {
            tau[5] = value;
        }
        state.tau = Point::new(tau);
        let tau = state.tau.as_slice().to_vec();
        let eq_suffix = Poly::new_from_point(&tau[1..], Tower::ONE);
        let strategy = if late {
            SlicedStrategy::TensorBoundaryLate
        } else {
            SlicedStrategy::TensorBoundary
        };
        let first = state
            .round_poly_sliced_with_strategy::<Gf4, Ghash128>(&eq_suffix, strategy)
            .expect("eligible tensor stage should build");
        let mut state = state.fold_sliced::<Ghash128>(prefix[0]);
        let tau = state.tau.as_slice().to_vec();
        let mut round_polys = vec![first];
        for round in 1..tau.len() {
            let suffix = Poly::new_from_point(&tau[round + 1..], Tower::ONE);
            let round_poly = if round < 4 {
                state
                    .round_poly_sliced::<Gf4>(&suffix)
                    .expect("tensor should serve its four rounds")
            } else if late && round == 4 {
                state
                    .round_poly_late_boundary::<Gf4>(&suffix)
                    .expect("late boundary should serve round four")
            } else {
                state.round_poly_repr(&suffix)
            };
            round_polys.push(round_poly);

            let challenge = prefix
                .get(round)
                .copied()
                .unwrap_or_else(|| challenge(round));
            if round < 3 {
                assert!(state.fold_sliced(challenge));
            } else if late && round == 3 {
                assert!(state.fold_late_boundary::<Gf4>(challenge));
            } else if !late && round == 3 {
                assert!(state.fold_boundary::<Gf4>(challenge));
            } else if late && round == 4 {
                assert!(state.fold_late_boundary::<Gf4>(challenge));
            } else {
                state.fold_repr(challenge);
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
        (round_polys, openings)
    })
}

#[test]
fn late_boundary_matches_incumbent_for_all_special_prefix_coordinates() {
    let height = 1 << 11;
    let instances = [Instance::honest(FixtureAir::Pair, height, 0x007E_50B2)];
    let lambda = Tower::interpolation_node(2);
    let special = [Tower::ZERO, Tower::ONE, lambda, Tower::from_repr(0x1234)];
    let ordinary = [
        challenge(0),
        challenge(1),
        challenge(2),
        challenge(3),
        challenge(4),
    ];
    for coordinate in 0..5 {
        for &value in &special {
            let mut prefix = ordinary;
            prefix[coordinate] = value;
            assert_eq!(
                collect_late_boundary_rounds(&instances, prefix, true),
                collect_late_boundary_rounds(&instances, prefix, false),
                "special coordinate {coordinate}, value {value:?}"
            );
        }
    }
    for &r3 in &special {
        for &r4 in &special {
            let mut prefix = ordinary;
            prefix[3] = r3;
            prefix[4] = r4;
            assert_eq!(
                collect_late_boundary_rounds(&instances, prefix, true),
                collect_late_boundary_rounds(&instances, prefix, false),
                "special pair ({r3:?}, {r4:?})"
            );
        }
    }
}

#[test]
fn late_boundary_matches_incumbent_for_special_tau4_and_tau5() {
    let height = 1 << 11;
    let instances = [Instance::honest(FixtureAir::Pair, height, 0x007E_50B3)];
    let special = [
        Tower::ONE,
        Tower::interpolation_node(2),
        Tower::from_repr(0x1234),
    ];
    let prefix = [
        challenge(0),
        challenge(1),
        challenge(2),
        challenge(3),
        challenge(4),
    ];
    for &tau4 in &special {
        for &tau5 in &special {
            assert_eq!(
                collect_late_boundary_rounds_with_tau(
                    &instances,
                    prefix,
                    true,
                    Some(tau4),
                    Some(tau5),
                ),
                collect_late_boundary_rounds_with_tau(
                    &instances,
                    prefix,
                    false,
                    Some(tau4),
                    Some(tau5),
                ),
                "special tau4={tau4:?}, tau5={tau5:?}"
            );
        }
    }
}

#[test]
fn late_boundary_active_quadratic_inputs_have_nonzero_merged_nodes() {
    let height = 1 << 11;
    let instances = [Instance::honest(
        FixtureAir::QuadraticInputs,
        height,
        0x007E_50B4,
    )];
    let prefix = [
        challenge(0),
        challenge(1),
        challenge(2),
        challenge(3),
        challenge(4),
    ];
    let late = collect_late_boundary_rounds(&instances, prefix, true);
    let incumbent = collect_late_boundary_rounds(&instances, prefix, false);
    assert_eq!(late, incumbent, "active quadratic-input late transcript");
    assert!(
        late.0[4].iter().any(|&value| value != Tower::ZERO),
        "round-four merged nodes must be nonzero"
    );
}

fn explicit_eq_weight(point: &[Tower], mask: usize) -> Tower {
    point
        .iter()
        .enumerate()
        .map(|(index, &challenge)| {
            if (mask >> (point.len() - 1 - index)) & 1 == 0 {
                Tower::ONE - challenge
            } else {
                challenge
            }
        })
        .product()
}

/// Evaluate the Pair and Linear AIRs directly from their original row-major corner tables.
/// This deliberately does not call `PlaneFold`, a boundary evaluator, or a round kernel.
fn explicit_round4_raw_nodes(
    instances: &[Instance],
    prefix: &[Tower; 4],
    suffix: &[Tower],
) -> Vec<Vec<Tower>> {
    let suffix_len = suffix.len();
    let lambda = Tower::interpolation_node(2);
    instances
        .iter()
        .map(|instance| {
            let nodes = match &instance.air {
                FixtureAir::Pair => vec![Tower::ZERO, lambda],
                FixtureAir::Linear { .. } => vec![Tower::ZERO],
                _ => panic!("raw round-four oracle fixture must be Pair or Linear"),
            };
            nodes
                .iter()
                .map(|&node| {
                    let mut sum = Tower::ZERO;
                    for suffix_mask in 0..1 << suffix_len {
                        let suffix_weight = explicit_eq_weight(suffix, suffix_mask);
                        let mut columns = vec![Tower::ZERO; instance.main.width];
                        for prefix_mask in 0..1 << prefix.len() {
                            let prefix_weight = explicit_eq_weight(prefix, prefix_mask);
                            let row_low = ((prefix_mask << 1) << suffix_len) | suffix_mask;
                            let row_high = row_low | (1 << suffix_len);
                            for (column, value) in columns.iter_mut().enumerate() {
                                let low =
                                    instance.main.values[row_low * instance.main.width + column];
                                let high =
                                    instance.main.values[row_high * instance.main.width + column];
                                *value += prefix_weight * (low + node * (high - low));
                            }
                        }
                        let value = match &instance.air {
                            FixtureAir::Pair => columns[0] * columns[1] - columns[2],
                            FixtureAir::Linear { scale } => (columns[0] - columns[1]) * *scale,
                            _ => unreachable!(),
                        };
                        sum += suffix_weight * value;
                    }
                    sum
                })
                .collect()
        })
        .collect()
}

#[test]
fn late_boundary_round4_matches_independent_raw_pair_and_corrupted_linear_nodes() {
    let height = 1 << 11;
    let pair = Instance::honest(FixtureAir::Pair, height, 0x007E_50D0);
    let mut linear = Instance::honest(
        FixtureAir::Linear { scale: Tower::ONE },
        height,
        0x007E_50D1,
    );
    for row in 0..height {
        linear.main.values[2 * row..2 * row + 2].copy_from_slice(&[Tower::ZERO, Tower::ONE]);
    }
    let instances = [pair, linear];
    let prefix = [challenge(0), challenge(1), challenge(2), challenge(3)];

    let (late_raw, late_nodes, late_last_evals, late_claims, late_rounds, late_openings) =
        with_state(&instances, no_lookups(), |mut state, eq_suffix| {
            let first = state
                .round_poly_sliced_with_strategy::<Gf4, Ghash128>(
                    eq_suffix,
                    SlicedStrategy::TensorBoundaryLate,
                )
                .expect("late tensor4 should build");
            let mut round_polys = vec![first];
            let mut state = state.fold_sliced::<Ghash128>(prefix[0]);
            for round in 1..=2 {
                let tau = state.tau.as_slice().to_vec();
                let suffix = Poly::new_from_point(&tau[round + 1..], Tower::ONE);
                round_polys.push(
                    state
                        .round_poly_sliced::<Gf4>(&suffix)
                        .expect("tensor round should be evaluated before its fold"),
                );
                assert!(state.fold_sliced(prefix[round]));
            }
            let tau = state.tau.as_slice().to_vec();
            let suffix = Poly::new_from_point(&tau[4..], Tower::ONE);
            round_polys.push(
                state
                    .round_poly_sliced::<Gf4>(&suffix)
                    .expect("tensor round should be evaluated before its fold"),
            );
            assert!(state.fold_late_boundary::<Gf4>(prefix[3]));
            let tau = state.tau.as_slice().to_vec();
            let suffix = Poly::new_from_point(&tau[5..], Tower::ONE);
            let raw = {
                let ExtColumns::Sliced(columns) = &state.columns else {
                    panic!("late raw oracle requires retained planes")
                };
                let fold =
                    PlaneFold::<Ghash128>::new::<Gf4, Tower>(&columns.trace, &columns.challenges);
                state.boundary_evals_rows(&suffix, &fold).0
            };
            let expected = explicit_round4_raw_nodes(&instances, &prefix, &tau[5..]);
            assert_eq!(raw, expected, "independent raw sparse nodes");
            assert!(raw.iter().flatten().any(|&value| value != Tower::ZERO));
            let nodes = state.round_poly_late_boundary::<Gf4>(&suffix).unwrap();
            round_polys.push(nodes.clone());
            let last_evals = state
                .constraint_groups
                .iter()
                .map(|group| group.last_evals.clone())
                .collect::<Vec<_>>();
            let claims = state
                .constraint_groups
                .iter()
                .map(|group| group.claim)
                .collect::<Vec<_>>();
            assert!(state.fold_late_boundary::<Gf4>(challenge(4)));
            for round in 5..tau.len() {
                let tau = state.tau.as_slice().to_vec();
                let suffix = Poly::new_from_point(&tau[round + 1..], Tower::ONE);
                round_polys.push(state.round_poly_repr(&suffix));
                state.fold_repr(challenge(round));
            }
            let openings: Vec<[Vec<Tower>; 4]> = state
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
            (raw, nodes, last_evals, claims, round_polys, openings)
        });

    let (
        incumbent_nodes,
        incumbent_last_evals,
        incumbent_claims,
        incumbent_rounds,
        incumbent_openings,
    ) = with_state(&instances, no_lookups(), |mut state, eq_suffix| {
        let first = state
            .round_poly_sliced_with_strategy::<Gf4, Ghash128>(
                eq_suffix,
                SlicedStrategy::TensorBoundary,
            )
            .expect("incumbent tensor4 should build");
        let mut round_polys = vec![first];
        let mut state = state.fold_sliced::<Ghash128>(prefix[0]);
        for round in 1..=2 {
            let tau = state.tau.as_slice().to_vec();
            let suffix = Poly::new_from_point(&tau[round + 1..], Tower::ONE);
            round_polys.push(
                state
                    .round_poly_sliced::<Gf4>(&suffix)
                    .expect("tensor round should be evaluated before its fold"),
            );
            assert!(state.fold_sliced(prefix[round]));
        }
        let tau = state.tau.as_slice().to_vec();
        let suffix = Poly::new_from_point(&tau[4..], Tower::ONE);
        round_polys.push(
            state
                .round_poly_sliced::<Gf4>(&suffix)
                .expect("tensor round should be evaluated before its fold"),
        );
        assert!(state.fold_boundary::<Gf4>(prefix[3]));
        let tau = state.tau.as_slice().to_vec();
        let suffix = Poly::new_from_point(&tau[5..], Tower::ONE);
        let nodes = state.round_poly_repr(&suffix);
        round_polys.push(nodes.clone());
        let last_evals = state
            .constraint_groups
            .iter()
            .map(|group| group.last_evals.clone())
            .collect::<Vec<_>>();
        let claims = state
            .constraint_groups
            .iter()
            .map(|group| group.claim)
            .collect::<Vec<_>>();
        assert_eq!(state.round, 4);
        state.fold_repr(challenge(4));
        for round in 5..tau.len() {
            let tau = state.tau.as_slice().to_vec();
            let suffix = Poly::new_from_point(&tau[round + 1..], Tower::ONE);
            round_polys.push(state.round_poly_repr(&suffix));
            state.fold_repr(challenge(round));
        }
        let openings: Vec<[Vec<Tower>; 4]> = state
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
        (nodes, last_evals, claims, round_polys, openings)
    });

    assert_eq!(late_nodes, incumbent_nodes, "transmitted stage nodes");
    assert_eq!(late_last_evals, incumbent_last_evals, "stored sparse nodes");
    assert_eq!(late_claims, incumbent_claims, "group claims");
    assert_eq!(late_rounds, incumbent_rounds, "all round polynomials");
    assert_eq!(late_openings, incumbent_openings, "final openings");
    assert_eq!(late_raw.len(), 2);
}

#[test]
fn late_boundary_raw_round4_matches_tau5_zero_and_one_without_claim_folding() {
    let height = 1 << 11;
    let instances = [Instance::honest(FixtureAir::Pair, height, 0x007E_50D2)];
    let prefix = [challenge(0), challenge(1), challenge(2), challenge(3)];
    for tau5 in [Tower::ZERO, Tower::ONE] {
        with_state(&instances, no_lookups(), |mut state, _| {
            let mut tau = state.tau.as_slice().to_vec();
            tau[5] = tau5;
            state.tau = Point::new(tau.clone());
            let eq_suffix = Poly::new_from_point(&tau[1..], Tower::ONE);
            state
                .round_poly_sliced_with_strategy::<Gf4, Ghash128>(
                    &eq_suffix,
                    SlicedStrategy::TensorBoundaryLate,
                )
                .expect("late tensor4 should build");
            let mut state = state.fold_sliced::<Ghash128>(prefix[0]);
            for round in 1..=2 {
                let tau = state.tau.as_slice().to_vec();
                let suffix = Poly::new_from_point(&tau[round + 1..], Tower::ONE);
                state
                    .round_poly_sliced::<Gf4>(&suffix)
                    .expect("tensor round should be evaluated before its fold");
                assert!(state.fold_sliced(prefix[round]));
            }
            let tau = state.tau.as_slice().to_vec();
            let suffix = Poly::new_from_point(&tau[4..], Tower::ONE);
            state
                .round_poly_sliced::<Gf4>(&suffix)
                .expect("tensor round should be evaluated before its fold");
            assert!(state.fold_late_boundary::<Gf4>(prefix[3]));
            let suffix = Poly::new_from_point(&tau[5..], Tower::ONE);
            let ExtColumns::Sliced(columns) = &state.columns else {
                panic!("raw tau test requires retained planes")
            };
            let fold =
                PlaneFold::<Ghash128>::new::<Gf4, Tower>(&columns.trace, &columns.challenges);
            let raw = state.boundary_evals_rows(&suffix, &fold).0;
            assert_eq!(
                raw,
                explicit_round4_raw_nodes(&instances, &prefix, &tau[5..]),
                "raw sparse nodes for tau5={tau5:?}"
            );
            assert!(raw.iter().flatten().any(|&value| value != Tower::ZERO));
        });
    }
}

/// Collect every round and final opening from either the tensor or sequential sliced path.
fn collect_tensor_rounds(
    instances: &[Instance],
    tensor: bool,
) -> (Vec<Vec<Tower>>, Vec<[Vec<Tower>; 4]>) {
    with_state(instances, no_lookups(), |mut state, _| {
        let tau = state.tau.as_slice().to_vec();
        let eq_suffix = Poly::new_from_point(&tau[1..], Tower::ONE);
        let first = if tensor {
            state
                .round_poly_sliced_with_strategy::<Gf4, Ghash128>(
                    &eq_suffix,
                    SlicedStrategy::TensorBoundary,
                )
                .expect("tensor4 should be eligible")
        } else {
            state
                .round_poly_sliced::<Gf4, Ghash128>(&eq_suffix)
                .expect("sequential sliced path should be eligible")
        };
        if tensor {
            assert!(state.has_sliced_tensor());
        }
        let mut state = state.fold_sliced::<Ghash128>(challenge(0));
        let mut round_polys = vec![first];
        for round in 1..tau.len() {
            let tau = state.tau.as_slice().to_vec();
            let suffix = Poly::new_from_point(&tau[round + 1..], Tower::ONE);
            let round_poly = if (tensor && round < 4) || (!tensor && round < 3) {
                state
                    .round_poly_sliced::<Gf4>(&suffix)
                    .expect("tensor cache should serve cached round")
            } else if !tensor && round == 3 {
                state
                    .round_poly_boundary::<Gf4>(&suffix)
                    .expect("sequential boundary should serve round three")
            } else {
                assert!(
                    matches!(state.columns, ExtColumns::Scalar(_)),
                    "tensor={tensor} round={round} should be scalar"
                );
                state.round_poly_repr(&suffix)
            };
            round_polys.push(round_poly);
            if round < 3 {
                assert!(state.fold_sliced(challenge(round)));
            } else if round == 3 {
                let folded = state.fold_boundary::<Gf4>(challenge(round));
                assert!(folded, "tensor={tensor} round={round}");
                assert!(matches!(state.columns, ExtColumns::Scalar(_)));
            } else {
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
        (round_polys, openings)
    })
}

#[test]
fn tensor4_path_includes_fixed_periodic_and_boundary_inputs() {
    let height = 1 << 10;
    let instances = [Instance::honest(
        FixtureAir::QuadraticInputs,
        height,
        0x007E_5004,
    )];
    let tensor = collect_tensor_rounds(&instances, true);
    let sequential = collect_tensor_rounds(&instances, false);
    assert_eq!(tensor, sequential, "quadratic input rounds and openings");
    assert!(
        tensor
            .0
            .iter()
            .any(|poly| poly.iter().any(|&value| value != Tower::ZERO))
    );
}

#[test]
fn tensor4_contraction_matches_sequential_at_special_challenges() {
    let height = 1 << 10;
    let mut instance = Instance::honest(FixtureAir::Pair, height, 0x007E_5005);
    for (index, value) in instance.main.values.iter_mut().enumerate() {
        *value = gf4((index + 2) & 3);
    }
    let instances = [instance];
    let normal = [challenge(0), challenge(1), challenge(2), challenge(3)];
    let lambda = Tower::interpolation_node(2);
    let full_width = Tower::from_repr(0x1234);
    let challenges = [
        [Tower::ZERO, normal[1], normal[2], normal[3]],
        [Tower::ONE, normal[1], normal[2], normal[3]],
        [lambda, normal[1], normal[2], normal[3]],
        [full_width, normal[1], normal[2], normal[3]],
        [normal[0], Tower::ZERO, normal[2], normal[3]],
        [normal[0], Tower::ONE, normal[2], normal[3]],
        [normal[0], lambda, normal[2], normal[3]],
        [normal[0], full_width, normal[2], normal[3]],
        [normal[0], normal[1], Tower::ZERO, normal[3]],
        [normal[0], normal[1], Tower::ONE, normal[3]],
        [normal[0], normal[1], lambda, normal[3]],
        [normal[0], normal[1], full_width, normal[3]],
    ];
    for prefix_challenges in challenges {
        let tensor = with_state(&instances, no_lookups(), |mut state, _| {
            let mut tau = state.tau.as_slice().to_vec();
            tau[1] = Tower::ONE;
            tau[2] = gf4(2);
            tau[3] = Tower::from_repr(0x4567);
            state.tau = Point::new(tau);
            let eq_suffix = Poly::new_from_point(&state.tau.as_slice()[1..], Tower::ONE);
            let first = state
                .round_poly_sliced_with_strategy::<Gf4, Ghash128>(
                    &eq_suffix,
                    SlicedStrategy::TensorBoundary,
                )
                .expect("tensor4 should be eligible");
            let mut state = state.fold_sliced::<Ghash128>(prefix_challenges[0]);
            let tau = state.tau.as_slice().to_vec();
            let suffix = Poly::new_from_point(&tau[2..], Tower::ONE);
            let second = state
                .round_poly_sliced::<Gf4>(&suffix)
                .expect("tensor4 should serve round one");
            assert!(state.fold_sliced(prefix_challenges[1]));
            let tau = state.tau.as_slice().to_vec();
            let suffix = Poly::new_from_point(&tau[3..], Tower::ONE);
            let third = state
                .round_poly_sliced::<Gf4>(&suffix)
                .expect("tensor4 should serve round two");
            assert!(state.fold_sliced(prefix_challenges[2]));
            let tau = state.tau.as_slice().to_vec();
            let suffix = Poly::new_from_point(&tau[4..], Tower::ONE);
            let fourth = state
                .round_poly_sliced::<Gf4>(&suffix)
                .expect("tensor4 should serve round three");
            assert!(state.fold_boundary::<Gf4>(prefix_challenges[3]));
            assert!(!state.has_sliced_tensor());
            (first, second, third, fourth)
        });
        let sequential = with_state(&instances, no_lookups(), |mut state, _| {
            let mut tau = state.tau.as_slice().to_vec();
            tau[1] = Tower::ONE;
            tau[2] = gf4(2);
            tau[3] = Tower::from_repr(0x4567);
            state.tau = Point::new(tau);
            let eq_suffix = Poly::new_from_point(&state.tau.as_slice()[1..], Tower::ONE);
            let first = state
                .round_poly_sliced::<Gf4, Ghash128>(&eq_suffix)
                .expect("sequential sliced path should be eligible");
            let mut state = state.fold_sliced::<Ghash128>(prefix_challenges[0]);
            let tau = state.tau.as_slice().to_vec();
            let suffix = Poly::new_from_point(&tau[2..], Tower::ONE);
            let second = state
                .round_poly_sliced::<Gf4>(&suffix)
                .expect("sequential sliced path should serve round one");
            assert!(state.fold_sliced(prefix_challenges[1]));
            let tau = state.tau.as_slice().to_vec();
            let suffix = Poly::new_from_point(&tau[3..], Tower::ONE);
            let third = state
                .round_poly_sliced::<Gf4>(&suffix)
                .expect("sequential sliced path should serve round two");
            assert!(state.fold_sliced(prefix_challenges[2]));
            let tau = state.tau.as_slice().to_vec();
            let suffix = Poly::new_from_point(&tau[4..], Tower::ONE);
            let fourth = state
                .round_poly_boundary::<Gf4>(&suffix)
                .expect("sequential boundary path should serve round three");
            assert!(state.fold_boundary::<Gf4>(prefix_challenges[3]));
            (first, second, third, fourth)
        });
        assert_eq!(tensor, sequential, "challenge {prefix_challenges:?}");
    }
}

#[test]
fn tensor4_strategy_falls_back_for_short_or_cubic_stages() {
    let short = [Instance::honest(FixtureAir::Pair, 1 << 9, 0x007E_5006)];
    with_state(&short, no_lookups(), |mut state, eq_suffix| {
        state
            .round_poly_sliced_with_strategy::<Gf4, Ghash128>(
                eq_suffix,
                SlicedStrategy::TensorBoundary,
            )
            .expect("short pair should retain sequential sliced fallback");
        assert!(!state.has_sliced_tensor());
    });

    // The periodic stage reads no successor, so the degree clause alone rejects it.
    let cubic = [Instance::honest(
        FixtureAir::Periodic {
            period: [gf4(2), gf4(3)],
        },
        1 << 10,
        0x007E_5007,
    )];
    with_state(&cubic, no_lookups(), |mut state, eq_suffix| {
        state
            .round_poly_sliced_with_strategy::<Gf4, Ghash128>(
                eq_suffix,
                SlicedStrategy::TensorBoundary,
            )
            .expect("cubic periodic stage should retain sequential sliced fallback");
        assert!(!state.has_sliced_tensor());
    });
}

#[test]
fn tensor4_strategy_matches_the_generic_kernel_on_a_cubic_stage() {
    // The periodic stage is cubic, reads no successor, and fits GF(4) with three sliced rounds,
    // so only the degree clause keeps it off the tensor, whose contraction stops at node two.
    let instances = [Instance::honest(
        FixtureAir::Periodic {
            period: [gf4(2), gf4(3)],
        },
        1 << 10,
        0x007E_5026,
    )];
    let first_round = |tensor: bool| {
        with_state(&instances, no_lookups(), |mut state, eq_suffix| {
            let evals = if tensor {
                state
                    .round_poly_sliced_with_strategy::<Gf4, Ghash128>(
                        eq_suffix,
                        SlicedStrategy::TensorBoundary,
                    )
                    .expect("a cubic stage should retain the sequential sliced path")
            } else {
                state.round_poly(eq_suffix)
            };
            (
                evals,
                state
                    .constraint_groups
                    .iter()
                    .map(|group| (group.claim, group.last_evals.clone()))
                    .collect::<Vec<_>>(),
            )
        })
    };
    assert_eq!(first_round(true), first_round(false));
}

#[test]
fn tensor4_eligibility_gates_are_isolated_at_height_ten() {
    let height = 1 << 10;
    let assert_no_tensor = |instances: &[Instance], coupling| {
        with_state(instances, coupling, |mut state, eq_suffix| {
            let _ = state.round_poly_sliced_with_strategy::<Gf4, Ghash128>(
                eq_suffix,
                SlicedStrategy::TensorBoundary,
            );
            assert!(!state.has_sliced_tensor());
        });
    };

    for sliced_rounds in [0, 1, 2, 4] {
        let instance =
            Instance::honest(FixtureAir::Pair, height, 0x007E_5008 + sliced_rounds as u64);
        with_state(&[instance], no_lookups(), |mut state, eq_suffix| {
            state.sliced_rounds = sliced_rounds;
            let _ = state.round_poly_sliced_with_strategy::<Gf4, Ghash128>(
                eq_suffix,
                SlicedStrategy::TensorBoundary,
            );
            assert!(
                !state.has_sliced_tensor(),
                "configured sliced rounds {sliced_rounds}"
            );
        });
    }
    assert_no_tensor(
        &[Instance::honest(
            FixtureAir::Periodic {
                period: [gf4(2), gf4(3)],
            },
            height,
            0x007E_5009,
        )],
        no_lookups(),
    );
    assert_no_tensor(
        &[Instance::honest(
            FixtureAir::QuadraticSuccessor,
            height,
            0x007E_500A,
        )],
        no_lookups(),
    );
    assert_no_tensor(
        &[Instance::honest(FixtureAir::Link, height, 0x007E_500B)],
        link_coupling(),
    );

    let mut main = Instance::honest(FixtureAir::QuadraticInputs, height, 0x007E_500C);
    main.main.values[5] = outside();
    assert_no_tensor(&[main], no_lookups());

    let mut preprocessed = Instance::honest(FixtureAir::QuadraticInputs, height, 0x007E_500D);
    preprocessed
        .preprocessed
        .as_mut()
        .expect("quadratic inputs has fixed data")
        .values[5] = outside();
    assert_no_tensor(&[preprocessed], no_lookups());

    let mut public = Instance::honest(FixtureAir::QuadraticInputs, height, 0x007E_500E);
    public.public_values[0] = outside();
    assert_no_tensor(&[public], no_lookups());

    let mut periodic = Instance::honest(
        FixtureAir::QuadraticInputsOutsidePeriodic,
        height,
        0x007E_500F,
    );
    periodic
        .preprocessed
        .as_mut()
        .expect("quadratic inputs has fixed data")
        .values
        .fill(Tower::ZERO);
    assert!(<Tower as HasSubfield<Gf4>>::all_in_subfield(
        &periodic.main.values
    ));
    assert!(<Tower as HasSubfield<Gf4>>::all_in_subfield(
        &periodic
            .preprocessed
            .as_ref()
            .expect("quadratic inputs has fixed data")
            .values
    ));
    assert!(<Tower as HasSubfield<Gf4>>::all_in_subfield(
        &periodic.public_values
    ));
    assert_eq!(periodic.air.periodic_columns()[0][0], outside());
    assert_no_tensor(&[periodic], no_lookups());
}

#[test]
fn tensor4_poison_falls_back_transactionally_on_eligible_mixed_stage() {
    let height = 1 << 10;
    let instances = [
        Instance::honest(FixtureAir::Pair, height, 0x007E_5020),
        Instance::honest(FixtureAir::Linear { scale: outside() }, height, 0x007E_5021),
    ];
    let tensor = with_state(&instances, no_lookups(), |mut state, eq_suffix| {
        let evals = state
            .round_poly_sliced_with_strategy::<Gf4, Ghash128>(
                eq_suffix,
                SlicedStrategy::TensorBoundary,
            )
            .expect("poisoned tensor attempt should retain sequential fallback");
        assert!(!state.has_sliced_tensor());
        (
            evals,
            state
                .constraint_groups
                .iter()
                .map(|group| (group.claim, group.last_evals.clone()))
                .collect::<Vec<_>>(),
        )
    });
    let sequential = with_state(&instances, no_lookups(), |mut state, eq_suffix| {
        let evals = state
            .round_poly_sliced::<Gf4, Ghash128>(eq_suffix)
            .expect("sequential sliced path should remain eligible");
        (
            evals,
            state
                .constraint_groups
                .iter()
                .map(|group| (group.claim, group.last_evals.clone()))
                .collect::<Vec<_>>(),
        )
    });
    assert_eq!(tensor, sequential);
}

#[test]
fn late_tensor4_poison_falls_back_without_installing_late_state() {
    let height = 1 << 11;
    let instances = [
        Instance::honest(FixtureAir::Pair, height, 0x007E_5024),
        Instance::honest(FixtureAir::Linear { scale: outside() }, height, 0x007E_5025),
    ];
    let late = with_state(&instances, no_lookups(), |mut state, eq_suffix| {
        let evals = state
            .round_poly_sliced_with_strategy::<Gf4, Ghash128>(
                eq_suffix,
                SlicedStrategy::TensorBoundaryLate,
            )
            .expect("poisoned late tensor attempt should retain sliced fallback");
        assert!(!state.has_sliced_tensor());
        assert!(
            !state
                .sliced
                .as_ref()
                .is_some_and(|columns| columns.late_boundary)
        );
        (
            evals,
            state
                .constraint_groups
                .iter()
                .map(|group| (group.claim, group.last_evals.clone()))
                .collect::<Vec<_>>(),
        )
    });
    let sequential = with_state(&instances, no_lookups(), |mut state, eq_suffix| {
        let evals = state
            .round_poly_sliced::<Gf4, Ghash128>(eq_suffix)
            .expect("sequential sliced fallback should remain eligible");
        (
            evals,
            state
                .constraint_groups
                .iter()
                .map(|group| (group.claim, group.last_evals.clone()))
                .collect::<Vec<_>>(),
        )
    });
    assert_eq!(
        late, sequential,
        "poisoned late attempt must match fallback"
    );
}

#[test]
fn tensor4_contraction_skips_zero_degree_slots_in_mixed_stage() {
    let empty = FixtureAir::Empty;
    let pair = FixtureAir::Pair;
    let slots = vec![
        AirSlot {
            air: &empty,
            stage_index: 0,
            caller_index: 0,
            main_offset: 0,
            main_width: 1,
            preprocessed_offset: 0,
            preprocessed_width: 0,
            periodic_offset: 0,
            periodic_width: 0,
            main_next_columns: vec![],
            preprocessed_next_columns: vec![],
            constraint_degree: 0,
            interaction: None,
        },
        AirSlot {
            air: &pair,
            stage_index: 1,
            caller_index: 1,
            main_offset: 1,
            main_width: 3,
            preprocessed_offset: 0,
            preprocessed_width: 0,
            periodic_offset: 0,
            periodic_width: 0,
            main_next_columns: vec![],
            preprocessed_next_columns: vec![],
            constraint_degree: 2,
            interaction: None,
        },
    ];
    let tensor = SlicedTensor {
        values: vec![vec![Tower::ZERO; 81], vec![Tower::ONE; 81]],
        depth: 4,
    };
    let evals = tensor_round(&tensor, &slots, &[Tower::ZERO; 4], &[], 0);
    assert!(evals[0].is_empty());
    assert_eq!(evals[1].len(), 2);
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

#[test]
fn a_prefix_fold_matches_folding_every_corner() {
    type Sliced = SlicedGf4<Tower, Gf4>;
    let coordinates = [(false, false), (true, false), (false, true), (true, true)];
    let mut rng = SmallRng::seed_from_u64(28);
    for round in 0..=MAX_SLICED_ROUNDS {
        let planes = (0..2 << round)
            .map(|_| [rng.random(), rng.random()])
            .collect::<Vec<[u64; 2]>>();
        for index in 0..coordinates.len().pow(round as u32) {
            let prefix = (0..round)
                .map(|variable| coordinates[(index >> (2 * variable)) % coordinates.len()])
                .collect::<Vec<_>>();
            let mut every = planes
                .iter()
                .map(|&[low, high]| Sliced::from_planes(low, high))
                .collect::<Vec<_>>();
            let expected = fold_corners(&mut every, &prefix);

            let fold = PrefixFold::new(&prefix);
            assert_eq!(
                fold.corners.len(),
                2 << fold.nodes.len(),
                "prefix {prefix:?}"
            );
            let mut read = fold
                .corners
                .iter()
                .map(|&corner| Sliced::from_planes(planes[corner][0], planes[corner][1]))
                .collect::<Vec<_>>();
            let folded = fold_corners(&mut read, &fold.nodes);
            for (folded, expected) in [(folded.0, expected.0), (folded.1, expected.1)] {
                assert_eq!(
                    (0..SLICED_LANES)
                        .map(|lane| folded.lane(lane))
                        .collect::<Vec<_>>(),
                    (0..SLICED_LANES)
                        .map(|lane| expected.lane(lane))
                        .collect::<Vec<_>>(),
                    "prefix {prefix:?}"
                );
            }
        }
    }
}
