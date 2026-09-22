use alloc::vec::Vec;

use p3_binary_field::BinaryField128;
use p3_challenger::testing::{assert_seeds_pairwise_distinct, seed_digest};

use crate::product::{ProductGkrRootShape, ProductGkrShape};

#[test]
fn transcript_seed_binds_every_shape_dimension() {
    // Height-zero shapes must differ even when their message patterns alias.
    let shapes = [
        ProductGkrShape::new(0, 1, ProductGkrRootShape::Distinct).unwrap(),
        ProductGkrShape::new(0, 2, ProductGkrRootShape::FirstTwoShared).unwrap(),
        ProductGkrShape::new(4, 2, ProductGkrRootShape::Distinct).unwrap(),
        ProductGkrShape::new(5, 2, ProductGkrRootShape::Distinct).unwrap(),
        ProductGkrShape::new(4, 3, ProductGkrRootShape::Distinct).unwrap(),
        ProductGkrShape::new(4, 2, ProductGkrRootShape::FirstTwoShared).unwrap(),
    ];
    let seeds = shapes
        .iter()
        .enumerate()
        .map(|(index, shape)| {
            (
                index,
                seed_digest(&shape.domain_separator::<BinaryField128, BinaryField128>()),
            )
        })
        .collect::<Vec<_>>();
    assert_seeds_pairwise_distinct(&seeds);
}
