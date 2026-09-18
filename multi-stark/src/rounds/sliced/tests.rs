use alloc::vec;
use alloc::vec::Vec;

use p3_binary_field::{Ghash128, TowerLevel};
use p3_field::PrimeCharacteristicRing;

use super::*;
use crate::rounds::StageCoupling;
use crate::rounds::subfield::tests::{link_coupling, no_lookups, with_state};
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
fn successor_planes_shift_by_one_row_and_repeat_the_last() {
    // Two words: every third row is set, and so is the last row.
    let column = (0..2 * SLICED_LANES)
        .map(|row| gf4(usize::from(row % 3 == 0 || row + 1 == 2 * SLICED_LANES)))
        .collect::<Vec<_>>();
    let planes = pack_column::<Tower, Gf4>(&column).expect("bit cells fit");
    let successors = successor_planes(&planes);
    for row in 0..2 * SLICED_LANES {
        let successor = (row + 1).min(2 * SLICED_LANES - 1);
        let bit =
            |planes: &[[u64; 2]], row: usize| (planes[row / SLICED_LANES][0] >> (row % 64)) & 1;
        assert_eq!(bit(&successors, row), bit(&planes, successor), "row {row}");
    }
}
