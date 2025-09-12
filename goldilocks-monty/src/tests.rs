//! Tests for equivalence between standard and Montgomery implementations

use p3_field::PrimeField64;
use p3_poseidon2::{ExternalLayerConstants, Poseidon2};
use p3_symmetric::Permutation;

use crate::Goldilocks;

/// Test that MDS and Poseidon2 implementations produce the expected results
/// by comparing with known good outputs from the standard Goldilocks implementation.
#[test]
fn test_poseidon2_equivalence_with_standard_goldilocks() {
    // Test input values - same as used in standard goldilocks tests
    let test_input = [0u64, 1, 2, 3, 4, 5, 6, 7];

    // Montgomery version
    let mut input_monty = test_input.map(|x| Goldilocks::new(x));
    let poseidon2_monty: crate::poseidon2::Poseidon2GoldilocksHL<8> = Poseidon2::new(
        ExternalLayerConstants::<Goldilocks, 8>::new_from_saved_array(
            crate::poseidon2::HL_GOLDILOCKS_MONTY_8_EXTERNAL_ROUND_CONSTANTS,
            Goldilocks::new_array,
        ),
        Goldilocks::new_array(crate::poseidon2::HL_GOLDILOCKS_MONTY_8_INTERNAL_ROUND_CONSTANTS)
            .to_vec(),
    );
    poseidon2_monty.permute_mut(&mut input_monty);
    let output_monty: [u64; 8] = input_monty.map(|x| x.as_canonical_u64());

    // Expected output from the standard goldilocks test for input [0,1,2,3,4,5,6,7]
    // This is from test_poseidon2_width_8_range test in standard goldilocks
    let expected_output = [
        14266028122062624699,
        5353147180106052723,
        15203350112844181434,
        17630919042639565165,
        16601551015858213987,
        10184091939013874068,
        16774100645754596496,
        12047415603622314780,
    ];

    assert_eq!(
        output_monty, expected_output,
        "Montgomery Poseidon2 output should match standard Goldilocks output"
    );
}

/// Test MDS matrix with known inputs
#[test]
fn test_mds_matrix_functionality() {
    use crate::mds::MdsMatrixGoldilocksMonty;

    let mds = MdsMatrixGoldilocksMonty::default();

    // Test with 8 elements
    let input_8 = [
        Goldilocks::new(1),
        Goldilocks::new(2),
        Goldilocks::new(3),
        Goldilocks::new(4),
        Goldilocks::new(5),
        Goldilocks::new(6),
        Goldilocks::new(7),
        Goldilocks::new(8),
    ];
    let output_8 = mds.permute(input_8);

    // Verify the output is not the same as input (MDS transforms the data)
    assert_ne!(
        input_8.map(|x| x.as_canonical_u64()),
        output_8.map(|x| x.as_canonical_u64()),
        "MDS should transform the input"
    );

    // Test with 12 elements
    let input_12 = [
        Goldilocks::new(1),
        Goldilocks::new(2),
        Goldilocks::new(3),
        Goldilocks::new(4),
        Goldilocks::new(5),
        Goldilocks::new(6),
        Goldilocks::new(7),
        Goldilocks::new(8),
        Goldilocks::new(9),
        Goldilocks::new(10),
        Goldilocks::new(11),
        Goldilocks::new(12),
    ];
    let output_12 = mds.permute(input_12);

    // Verify the output is not the same as input
    assert_ne!(
        input_12.map(|x| x.as_canonical_u64()),
        output_12.map(|x| x.as_canonical_u64()),
        "MDS should transform the input"
    );
}
