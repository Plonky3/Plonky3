use alloc::vec;
use alloc::vec::Vec;

use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_binary_field::{BinaryChallenger, BinaryField128};
use p3_challenger::testing::{assert_seeds_pairwise_distinct, seed_digest};
use p3_challenger::{DuplexChallenger, HashChallenger};
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_keccak::Keccak256Hash;
use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoroshiro128Plus;

use crate::leaf::{
    BusDirection, BusLeafDeclaration, BusLeafError, BusSelector, materialize_bus_leaves,
};
use crate::product::{
    ProductGkrError, ProductGkrLayerProof, ProductGkrRootShape, ProductGkrShape, prove_product_gkr,
    verify_product_gkr,
};

type PrimeBase = BabyBear;
type PrimeChallenge = BinomialExtensionField<PrimeBase, 4>;
type PrimePermutation = Poseidon2BabyBear<16>;
type PrimeChallenger = DuplexChallenger<PrimeBase, PrimePermutation, 16, 8>;
type Binary = BinaryField128;
type BinaryTranscript = BinaryChallenger<Binary, HashChallenger<u8, Keccak256Hash, 32>>;

/// Build a deterministic prime-field transcript for differential tests.
fn prime_challenger() -> PrimeChallenger {
    // Matching seeds make prover and verifier draws identical.
    let mut rng = Xoroshiro128Plus::seed_from_u64(0xB055_0001);
    PrimeChallenger::new(PrimePermutation::new_from_rng_128(&mut rng))
}

/// Build a deterministic binary-field transcript for differential tests.
fn binary_challenger() -> BinaryTranscript {
    // An empty prefix leaves the protocol domain separator as the first absorbed input.
    BinaryTranscript::from_hasher(Vec::new(), Keccak256Hash)
}

/// Evaluate an identity-padded prefix at a multilinear point.
fn naive_eval<F: Field>(prefix: &[F], log_height: usize, point: &[F]) -> F {
    // Materialization is test-only and provides a deliberately obvious reference.
    let mut table = vec![F::ONE; 1usize << log_height];
    table[..prefix.len()].copy_from_slice(prefix);

    for &coordinate in point {
        // Bind the lowest remaining variable at adjacent entries.
        let half = table.len() / 2;
        for row in 0..half {
            table[row] = table[2 * row] + coordinate * (table[2 * row + 1] - table[2 * row]);
        }
        table.truncate(half);
    }

    table[0]
}

/// Check roots and terminal claims against a materialized scalar reference.
fn check_output<F: Field>(
    inputs: &[Vec<F>],
    log_height: usize,
    roots: &[F],
    point: &[F],
    values: &[F],
) {
    for (tree, input) in inputs.iter().enumerate() {
        // Omitted suffix leaves are one and therefore do not change the root.
        assert_eq!(roots[tree], input.iter().copied().product());
        assert_eq!(values[tree], naive_eval(input, log_height, point));
    }
}

#[test]
fn binary_field_matches_the_scalar_reference_across_depths() {
    // Depth zero exercises the root-only transcript.
    // Depths one through ten exercise odd and even radix-four schedules.
    let mut rng = Xoroshiro128Plus::seed_from_u64(0xB1A4_0001);
    for log_height in 0..=10 {
        let capacity = 1usize << log_height;
        let inputs = (0..3)
            .map(|tree| {
                let explicit = capacity.saturating_sub(tree.min(capacity));
                (0..explicit)
                    .map(|_| Binary::from_u64(rng.random::<u64>()))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let borrowed = inputs.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let shape = ProductGkrShape::new(log_height, 3, ProductGkrRootShape::Distinct).unwrap();

        let mut prover_challenger = binary_challenger();
        let (proof, prover_output) =
            prove_product_gkr::<Binary, Binary, _>(&borrowed, shape, &mut prover_challenger);
        let mut verifier_challenger = binary_challenger();
        let verifier_output =
            verify_product_gkr::<Binary, Binary, _>(&proof, shape, &mut verifier_challenger)
                .unwrap();

        assert_eq!(prover_output, verifier_output);
        check_output(
            &inputs,
            log_height,
            &prover_output.roots,
            &prover_output.point,
            &prover_output.values,
        );
    }
}

#[test]
fn baby_bear_extension_matches_the_scalar_reference() {
    // Unequal prefix lengths exercise identity padding in every tree.
    let mut rng = Xoroshiro128Plus::seed_from_u64(0xBA8E_0001);
    let log_height = 8;
    let inputs = [256, 191, 1]
        .map(|length| {
            (0..length)
                .map(|_| PrimeChallenge::from_u64(rng.random::<u64>()))
                .collect::<Vec<_>>()
        })
        .to_vec();
    let borrowed = inputs.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let shape = ProductGkrShape::new(log_height, 3, ProductGkrRootShape::Distinct).unwrap();

    let mut prover_challenger = prime_challenger();
    let (proof, prover_output) =
        prove_product_gkr::<PrimeBase, PrimeChallenge, _>(&borrowed, shape, &mut prover_challenger);
    let mut verifier_challenger = prime_challenger();
    let verifier_output =
        verify_product_gkr::<PrimeBase, PrimeChallenge, _>(&proof, shape, &mut verifier_challenger)
            .unwrap();

    assert_eq!(prover_output, verifier_output);
    check_output(
        &inputs,
        log_height,
        &prover_output.roots,
        &prover_output.point,
        &prover_output.values,
    );
}

#[test]
fn shared_roots_are_structural() {
    // The first two products agree even though their leaf order differs.
    let left = vec![
        Binary::from_u64(2),
        Binary::from_u64(3),
        Binary::from_u64(5),
    ];
    let right = vec![left[2], left[0], left[1]];
    let count = vec![Binary::from_u64(7), Binary::from_u64(11)];
    let inputs = [left.as_slice(), right.as_slice(), count.as_slice()];
    let shape = ProductGkrShape::new(3, 3, ProductGkrRootShape::FirstTwoShared).unwrap();

    let mut prover_challenger = binary_challenger();
    let (proof, output) =
        prove_product_gkr::<Binary, Binary, _>(&inputs, shape, &mut prover_challenger);

    // One proof value represents both balancing roots.
    assert_eq!(proof.roots.len(), 2);
    assert_eq!(output.roots[0], output.roots[1]);

    let mut verifier_challenger = binary_challenger();
    assert_eq!(
        verify_product_gkr::<Binary, Binary, _>(&proof, shape, &mut verifier_challenger).unwrap(),
        output,
    );
}

#[test]
fn identity_padding_and_zero_roots_are_valid() {
    // Empty and all-one prefixes both denote the constant-one logical tree.
    // A zero leaf makes the other tree's product zero without invalidating GKR.
    let empty = Vec::<Binary>::new();
    let ones = vec![Binary::ONE; 3];
    let zero = vec![Binary::ZERO];
    let inputs = [empty.as_slice(), ones.as_slice(), zero.as_slice()];
    let shape = ProductGkrShape::new(3, 3, ProductGkrRootShape::Distinct).unwrap();

    let mut prover_challenger = binary_challenger();
    let (proof, output) =
        prove_product_gkr::<Binary, Binary, _>(&inputs, shape, &mut prover_challenger);
    assert_eq!(output.roots, vec![Binary::ONE, Binary::ONE, Binary::ZERO]);

    let mut verifier_challenger = binary_challenger();
    assert_eq!(
        verify_product_gkr::<Binary, Binary, _>(&proof, shape, &mut verifier_challenger).unwrap(),
        output,
    );
}

#[test]
fn tampering_each_message_family_is_rejected() {
    // Height four includes both round-polynomial and child messages.
    let inputs = [
        (1..=16).map(Binary::from_u64).collect::<Vec<_>>(),
        (17..=32).map(Binary::from_u64).collect::<Vec<_>>(),
    ];
    let borrowed = inputs.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let shape = ProductGkrShape::new(4, 2, ProductGkrRootShape::Distinct).unwrap();
    let mut prover_challenger = binary_challenger();
    let (proof, _) =
        prove_product_gkr::<Binary, Binary, _>(&borrowed, shape, &mut prover_challenger);

    // Mutation 1: change a root before the first batching challenge.
    let mut root_tamper = proof.clone();
    root_tamper.roots[0] += Binary::ONE;
    let mut challenger = binary_challenger();
    assert!(verify_product_gkr::<Binary, Binary, _>(&root_tamper, shape, &mut challenger).is_err());

    // Mutation 2: change one degree-five sumcheck evaluation.
    let mut round_tamper = proof.clone();
    let ProductGkrLayerProof::RadixFour { round_polys, .. } = &mut round_tamper.layers[1] else {
        panic!("height four has a radix-four leaf layer");
    };
    round_polys[0][0] += Binary::ONE;
    let mut challenger = binary_challenger();
    assert!(
        verify_product_gkr::<Binary, Binary, _>(&round_tamper, shape, &mut challenger).is_err()
    );

    // Mutation 3: change one closing child claim.
    let mut child_tamper = proof;
    let ProductGkrLayerProof::RadixFour { children, .. } = &mut child_tamper.layers[0] else {
        panic!("height four starts with a radix-four layer");
    };
    children[0][0] += Binary::ONE;
    let mut challenger = binary_challenger();
    assert!(
        verify_product_gkr::<Binary, Binary, _>(&child_tamper, shape, &mut challenger).is_err()
    );
}

#[test]
fn malformed_proof_lengths_return_errors_without_panicking() {
    // A two-layer proof supplies both a layer count and per-layer child counts to attack.
    let input = vec![Binary::from_u64(9); 16];
    let shape = ProductGkrShape::new(4, 1, ProductGkrRootShape::Distinct).unwrap();
    let mut prover_challenger = binary_challenger();
    let (proof, _) =
        prove_product_gkr::<Binary, Binary, _>(&[input.as_slice()], shape, &mut prover_challenger);

    let mut truncated = proof.clone();
    truncated.layers.pop();
    let mut challenger = binary_challenger();
    assert!(matches!(
        verify_product_gkr::<Binary, Binary, _>(&truncated, shape, &mut challenger),
        Err(ProductGkrError::LayerCountMismatch { .. })
    ));

    let mut malformed = proof;
    let ProductGkrLayerProof::RadixFour { children, .. } = &mut malformed.layers[0] else {
        panic!("height four starts with a radix-four layer");
    };
    children.clear();
    let mut challenger = binary_challenger();
    assert!(matches!(
        verify_product_gkr::<Binary, Binary, _>(&malformed, shape, &mut challenger),
        Err(ProductGkrError::MalformedLayer { layer: 0 })
    ));
}

#[test]
fn transcript_seed_binds_every_shape_dimension() {
    // Each verifier-derived knob changes the interaction pattern.
    let shapes = [
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
                seed_digest(&shape.domain_separator::<Binary, Binary>()),
            )
        })
        .collect::<Vec<_>>();
    assert_seeds_pairwise_distinct(&seeds);
}

#[test]
fn directions_and_boolean_selection_survive_characteristic_two() {
    // Two equal pushes remain two leaves rather than cancelling as signed counts would.
    let column = [Binary::from_u64(7), Binary::from_u64(7)];
    let selector = [Binary::ONE, Binary::ZERO];
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

    let leaves = materialize_bus_leaves(&declarations, &[], Binary::from_u64(19)).unwrap();
    assert_eq!(leaves.pushes.len(), 2);
    assert_eq!(leaves.pulls.len(), 2);
    assert_eq!(leaves.pulls[1], Binary::ONE);
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

    let leaves =
        materialize_bus_leaves::<BabyBear, BabyBear>(&declaration, &point, offset).unwrap();
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
        materialize_bus_leaves::<BabyBear, BabyBear>(&declarations, &[], BabyBear::ZERO),
        Err(BusLeafError::NonBooleanSelector { .. })
    ));
}
