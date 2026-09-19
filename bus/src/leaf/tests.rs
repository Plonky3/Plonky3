use alloc::vec;
use alloc::vec::Vec;

use p3_baby_bear::BabyBear;
use p3_binary_field::BinaryField128;
use p3_field::PrimeCharacteristicRing;

use crate::BusDirection;
use crate::leaf::{BusLeafDeclaration, BusLeafError, BusLeaves, BusSelector};

#[test]
fn directions_and_boolean_selection_survive_characteristic_two() {
    // Two equal pushes remain two leaves rather than cancelling as signed counts would.
    let column = [BinaryField128::from_u64(7), BinaryField128::from_u64(7)];
    let selector = [BinaryField128::ONE, BinaryField128::ZERO];
    let columns = [&column[..]];
    let declarations = [
        BusLeafDeclaration {
            direction: BusDirection::Push,
            columns: &columns,
            selector: BusSelector::Always,
        },
        BusLeafDeclaration {
            direction: BusDirection::Pull,
            columns: &columns,
            selector: BusSelector::Boolean(&selector),
        },
    ];

    let leaves = BusLeaves::materialize(&declarations, &[], BinaryField128::from_u64(19)).unwrap();
    assert_eq!(leaves.pushes.len(), 2);
    assert_eq!(leaves.pulls.len(), 2);
    assert_eq!(leaves.pulls[1], BinaryField128::ONE);
}

#[test]
fn tuple_fingerprint_matches_direct_multilinear_evaluation() {
    // Four slots use the address order 00, 01, 10, 11.
    let columns = [
        [BabyBear::from_u64(2)],
        [BabyBear::from_u64(3)],
        [BabyBear::from_u64(5)],
        [BabyBear::from_u64(7)],
    ];
    let borrowed = columns
        .iter()
        .map(|column| column.as_slice())
        .collect::<Vec<_>>();
    let declaration = [BusLeafDeclaration {
        direction: BusDirection::Push,
        columns: &borrowed,
        selector: BusSelector::Always,
    }];
    let point = [BabyBear::from_u64(11), BabyBear::from_u64(13)];
    let offset = BabyBear::from_u64(17);

    let leaves = BusLeaves::materialize(&declaration, &point, offset).unwrap();
    let low_zero = columns[0][0] + point[0] * (columns[1][0] - columns[0][0]);
    let low_one = columns[2][0] + point[0] * (columns[3][0] - columns[2][0]);
    let fingerprint = low_zero + point[1] * (low_one - low_zero);

    assert_eq!(leaves.pushes, vec![offset - fingerprint]);
    assert!(leaves.pulls.is_empty());
}

#[test]
fn malformed_leaf_declarations_are_rejected() {
    // A non-Boolean selector must not become a fractional product multiplicity.
    let column = [BabyBear::ONE];
    let columns = [&column[..]];
    let selector = [BabyBear::TWO];
    let declarations = [BusLeafDeclaration {
        direction: BusDirection::Push,
        columns: &columns,
        selector: BusSelector::Boolean(&selector),
    }];

    assert!(matches!(
        BusLeaves::materialize(&declarations, &[], BabyBear::ZERO),
        Err(BusLeafError::NonBooleanSelector { .. })
    ));
}

#[test]
fn tuple_width_is_rejected_before_challenge_sized_allocation() {
    // A 24-coordinate point implies sixteen million slots.
    // A one-column declaration is rejected before those weights are allocated.
    let column = [BabyBear::ONE];
    let columns = [&column[..]];
    let declaration = [BusLeafDeclaration {
        direction: BusDirection::Push,
        columns: &columns,
        selector: BusSelector::Always,
    }];
    let point = vec![BabyBear::ZERO; 24];
    assert!(matches!(
        BusLeaves::materialize(&declaration, &point, BabyBear::ZERO),
        Err(BusLeafError::TupleWidthMismatch { .. })
    ));
}
