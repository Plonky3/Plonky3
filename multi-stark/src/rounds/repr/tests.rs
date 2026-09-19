use p3_binary_field::{BinaryField2, BinaryField128, Ghash128, TowerLevel};

use super::*;

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
