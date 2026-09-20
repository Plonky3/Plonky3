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

/// One later round of a stage bound into `Ghash128`, taken by the kernel `round` runs.
///
/// The fixture is rebuilt per call, so each kernel reads the same stage.
fn later_round(round: impl Fn(&mut ReprState<'_, '_>, &Poly<Tower>) -> Vec<Tower>) -> Vec<Tower> {
    let instances = [
        Instance::honest(FixtureAir::Pair, 1 << 6, 0xB0),
        Instance::honest(FixtureAir::Pair, 1 << 6, 0xB1),
    ];
    with_state(&instances, no_lookups(), |state, _| {
        let mut state = state.fold_into::<Ghash128>(first_challenge());
        let tau = state.tau.as_slice().to_vec();
        let eq_suffix = Poly::new_from_point(&tau[2..], Tower::ONE);
        round(&mut state, &eq_suffix)
    })
}

/// A stage of the fixture above, bound into the polynomial basis.
type ReprState<'air, 'data> = RoundStateExt<'air, 'data, FixtureAir, Tower, Tower, Ghash128>;

#[test]
fn a_lane_group_round_is_the_round_the_rows_give() {
    let lanes = later_round(|state, eq_suffix| state.round_poly_repr(eq_suffix));
    let rows = later_round(|state, eq_suffix| {
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
