use p3_binary_field::{BinaryField2, BinaryField128, Ghash128, TowerLevel};
use p3_field::PrimeCharacteristicRing;

use super::*;
use crate::rounds::subfield::tests::{first_challenge, no_lookups, with_state};
use crate::zerocheck::backend_tests::{FixtureAir, Instance};

type Tower = BinaryField128;

#[test]
fn the_fold_tables_fold_every_gf4_pair() {
    let r = Tower::from_repr(0xF01D_0000_0000_0000_0000_0000_0000_0007);
    let tables = SubfieldFoldTables::<Ghash128>::new::<BinaryField2, Tower, Tower>(r)
        .expect("the elements of GF(4) differ in their low byte");
    let gf4 = || (0..4).map(Tower::from_repr);
    for lo in gf4() {
        for hi in gf4() {
            assert_eq!(
                tables.low[table_index(lo)] + tables.scaled[table_index(hi - lo)],
                Ghash128::from(lo + r * (hi - lo)),
                "lo = {lo}, hi = {hi}"
            );
        }
    }
}

/// A stage of two AIRs reading no successor column.
fn pair_stage() -> [Instance; 2] {
    [
        Instance::honest(FixtureAir::Pair, 1 << 6, 0xB0),
        Instance::honest(FixtureAir::Pair, 1 << 6, 0xB1),
    ]
}

/// A stage of two AIRs reading a main and a preprocessed successor column.
fn gate_stage() -> [Instance; 2] {
    [
        Instance::honest(FixtureAir::Gate { scale: Tower::ONE }, 1 << 6, 0xB2),
        Instance::honest(FixtureAir::Gate { scale: Tower::ONE }, 1 << 6, 0xB3),
    ]
}

/// One later round of a stage bound into `Ghash128`, taken by the kernel `round` runs.
///
/// The fixture is rebuilt per call, so each kernel reads the same stage.
fn later_round(
    instances: &[Instance],
    round: impl Fn(&mut ReprState<'_, '_>, &Poly<Tower>) -> Vec<Tower>,
) -> Vec<Tower> {
    with_state(instances, no_lookups(), |state, _| {
        let mut state = state.fold_into::<Ghash128>(first_challenge());
        let tau = state.tau.as_slice().to_vec();
        let eq_suffix = Poly::new_from_point(&tau[2..], Tower::ONE);
        round(&mut state, &eq_suffix)
    })
}

/// A stage of the fixture above, bound into the polynomial basis.
type ReprState<'air, 'data> = RoundStateExt<'air, 'data, FixtureAir, Tower, Tower, Ghash128>;

/// The round polynomial of a stage, taken by the lane-group kernel and by the row kernel.
fn lane_and_row_rounds(instances: &[Instance]) -> (Vec<Tower>, Vec<Tower>) {
    let lanes = later_round(instances, |state, eq_suffix| {
        state.round_poly_repr(eq_suffix)
    });
    let rows = later_round(instances, |state, eq_suffix| {
        let lifted = Poly::new(
            eq_suffix
                .as_slice()
                .iter()
                .copied()
                .map(Ghash128::from)
                .collect(),
        );
        state.round_poly_unpacked(&lifted)
    });
    (lanes, rows)
}

#[test]
fn a_lane_group_round_is_the_round_the_rows_give() {
    let (lanes, rows) = lane_and_row_rounds(&pair_stage());
    assert_eq!(lanes, rows);
}

/// A stage reading successor columns takes the widened next-row load in every group but the
/// last, and the lane-by-lane path with the repeat-last tail in the last one.
#[test]
fn a_lane_group_round_over_successor_columns_is_the_round_the_rows_give() {
    let (lanes, rows) = lane_and_row_rounds(&gate_stage());
    assert_eq!(lanes, rows);
}

#[test]
fn a_lane_group_read_from_a_column_is_the_one_its_lanes_give() {
    let lanes = <Ghash128 as Field>::Packing::WIDTH;
    let column = (0..3 * lanes)
        .map(|row| {
            let seed = 0x9E37_79B9_7F4A_7C15_F39C_C060_5CED_C835_u128;
            Ghash128::from(Tower::from_repr(seed.wrapping_mul(row as u128 + 1)))
        })
        .collect::<Vec<_>>();

    for row in 0..=column.len() - lanes {
        let read = lane_rows::<Tower, Ghash128>(&column, row);
        let lanewise = lane_group::<Tower, Ghash128>(|lane| column[row + lane]);
        assert_eq!(read.0.as_slice(), lanewise.0.as_slice(), "row {row}");
    }
}

#[test]
fn a_lane_group_leaves_the_column_only_in_the_last_lane_of_the_last_group() {
    for log_evals in 1..10 {
        let num_evals = 1 << log_evals;
        let half = num_evals / 2;
        for lanes in [1, 2, 4, 8, 16] {
            if half % lanes != 0 {
                continue;
            }
            let groups = half / lanes;
            for group in 0..groups {
                let s = group * lanes;
                let in_column = next_rows_in_column(s, half, num_evals, lanes);

                // The group's last lane reads next row `s + half + lanes`.
                assert_eq!(
                    in_column,
                    s + half + lanes < num_evals,
                    "evals {num_evals}, lanes {lanes}, group {group}"
                );

                // Every group but the last stays inside the column.
                assert_eq!(
                    in_column,
                    group + 1 < groups,
                    "evals {num_evals}, lanes {lanes}, group {group}"
                );
            }
        }
    }
}
