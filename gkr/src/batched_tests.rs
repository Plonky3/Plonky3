use alloc::borrow::Cow;
use alloc::vec;
use alloc::vec::Vec;

use p3_binary_field::{BinaryChallenger, BinaryField128, Poly64, Poly192};
use p3_challenger::fs::TranscriptField;
use p3_challenger::{CanSample, FieldChallenger, HashChallenger};
use p3_field::{ExtensionField, Field, PrimeCharacteristicRing};
use p3_keccak::Keccak256Hash;
use rand::distr::{Distribution, StandardUniform};
use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoroshiro128Plus;

use crate::prover::{ProductLayers, RadixFourBatch};
use crate::{
    DenseLeaves, ProductGkrError, ProductGkrLayerProof, ProductGkrProof, ProductGkrShape,
    ProductGkrShapeError, ProductLeaves,
};

type Binary = BinaryField128;

fn challenger<F: Field>() -> BinaryChallenger<F, HashChallenger<u8, Keccak256Hash, 32>> {
    BinaryChallenger::from_hasher(Vec::new(), Keccak256Hash)
}

/// The multilinear extension of `table`, padded with ones, at a point whose first coordinate is most significant.
fn mle<F: Field>(table: &[F], num_variables: usize, point: &[F]) -> F {
    let mut values = vec![F::ONE; 1 << num_variables];
    values[..table.len()].copy_from_slice(table);
    for &coordinate in point.iter().rev() {
        values = values
            .chunks(2)
            .map(|pair| pair[0] + coordinate * (pair[1] - pair[0]))
            .collect();
    }
    values[0]
}

/// Prove and verify a batched table, checking every output against its definition.
fn round_trip<F, EF>(
    leaves: &[EF],
    log_height: usize,
    batch_variables: usize,
) -> ProductGkrProof<EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    BinaryChallenger<F, HashChallenger<u8, Keccak256Hash, 32>>: FieldChallenger<F>,
{
    let shape = ProductGkrShape::batched(log_height, batch_variables).unwrap();
    let mut prover = challenger::<F>();
    let (proof, output) = ProductGkrProof::prove_batched::<F, _, _>(
        &DenseLeaves(leaves.to_vec()),
        shape,
        &mut prover,
    );

    let mut padded = leaves.to_vec();
    padded.resize(1 << (batch_variables + log_height), EF::ONE);
    let roots: Vec<EF> = padded
        .chunks(1 << log_height)
        .map(|tree| tree.iter().copied().product())
        .collect();
    assert_eq!(output.roots, roots);
    assert_eq!(output.point.len(), batch_variables + log_height);
    assert_eq!(
        output.values,
        [mle(leaves, batch_variables + log_height, &output.point)]
    );

    let mut verifier = challenger::<F>();
    assert_eq!(proof.verify::<F, _>(shape, &mut verifier).unwrap(), output);
    assert_eq!(
        CanSample::<F>::sample(&mut prover),
        CanSample::<F>::sample(&mut verifier)
    );
    proof
}

fn random<T>(rng: &mut Xoroshiro128Plus, len: usize) -> Vec<T>
where
    StandardUniform: Distribution<T>,
{
    (0..len).map(|_| rng.random()).collect()
}

#[test]
fn batched_tables_prove_and_verify_at_every_height() {
    let mut rng = Xoroshiro128Plus::seed_from_u64(0xBA7C_0001);
    for batch_variables in 1..=3 {
        for log_height in 0..=9 {
            let len = 1 << (batch_variables + log_height);
            // The full table, and one whose identity suffix is left implicit.
            round_trip::<Binary, Binary>(
                &random::<Binary>(&mut rng, len),
                log_height,
                batch_variables,
            );
            round_trip::<Binary, Binary>(
                &random::<Binary>(&mut rng, len / 2 + 1),
                log_height,
                batch_variables,
            );
            round_trip::<Poly64, Poly192>(
                &random::<Poly192>(&mut rng, len),
                log_height,
                batch_variables,
            );
        }
    }
}

#[test]
fn odd_batched_heights_sum_the_binary_layer() {
    let mut rng = Xoroshiro128Plus::seed_from_u64(0xBA7C_0002);
    let proof = round_trip::<Binary, Binary>(&random::<Binary>(&mut rng, 1 << 7), 5, 2);
    let ProductGkrLayerProof::BinarySumcheck { round_polys, .. } = &proof.layers[0] else {
        panic!("an odd batched height starts with a summed binary layer");
    };
    assert_eq!(round_polys.len(), 2);
    assert!(matches!(
        proof.layers[1],
        ProductGkrLayerProof::RadixFour { .. }
    ));
}

/// Leaves that prove their bottom `own` radix-four layers through the hook.
struct OwnBottom {
    leaves: Vec<Poly192>,
    log_height: usize,
    own: usize,
}

impl ProductLeaves<Poly192> for OwnBottom {
    type Layer<'a> = RadixFourBatch<Poly192, 4>;

    fn own_layers(&self) -> usize {
        self.own
    }

    fn materialize(&self) -> Cow<'_, [Poly192]> {
        let mut level = self.leaves.clone();
        for _ in 0..self.own {
            level = level
                .chunks(4)
                .map(|group| group.iter().copied().product())
                .collect();
        }
        Cow::Owned(level)
    }

    fn layer(&self, j: usize, logical_len: usize) -> Self::Layer<'_> {
        let mut layers = [ProductLayers::new(&self.leaves, self.log_height)];
        RadixFourBatch::new_consuming(&mut layers, 2 * j, logical_len)
    }
}

#[test]
fn own_bottom_layers_leave_proofs_and_transcripts_unchanged() {
    let mut rng = Xoroshiro128Plus::seed_from_u64(0xBA7C_0003);
    for batch_variables in 1..=2 {
        for log_height in 0..=9 {
            let leaves = random::<Poly192>(&mut rng, 1 << (batch_variables + log_height));
            let shape = ProductGkrShape::batched(log_height, batch_variables).unwrap();
            let mut dense = challenger::<Poly64>();
            let expected = ProductGkrProof::prove_batched::<Poly64, _, _>(
                &DenseLeaves(leaves.clone()),
                shape,
                &mut dense,
            );
            for own in 0..=log_height / 2 {
                let hooked_leaves = OwnBottom {
                    leaves: leaves.clone(),
                    log_height,
                    own,
                };
                let mut hooked = challenger::<Poly64>();
                let actual = ProductGkrProof::prove_batched::<Poly64, _, _>(
                    &hooked_leaves,
                    shape,
                    &mut hooked,
                );
                assert_eq!(actual, expected, "height {log_height}, own {own}");
                assert_eq!(
                    CanSample::<Poly64>::sample(&mut hooked),
                    CanSample::<Poly64>::sample(&mut dense.clone())
                );
            }
        }
    }
}

#[test]
fn tampering_any_batched_message_or_root_is_rejected() {
    let mut rng = Xoroshiro128Plus::seed_from_u64(0xBA7C_0004);
    let shape = ProductGkrShape::batched(5, 2).unwrap();
    let proof = round_trip::<Binary, Binary>(&random::<Binary>(&mut rng, 1 << 7), 5, 2);
    let rejects = |proof: &ProductGkrProof<Binary>| {
        matches!(
            proof.verify::<Binary, _>(shape, &mut challenger::<Binary>()),
            Err(ProductGkrError::LayerConsistency { .. })
        )
    };
    for index in 0..proof.roots.len() {
        let mut bad = proof.clone();
        bad.roots[index] += Binary::ONE;
        assert!(rejects(&bad), "root {index}");
    }
    for layer in 0..proof.layers.len() {
        let (rounds, children) = match &proof.layers[layer] {
            ProductGkrLayerProof::BinarySumcheck { round_polys, .. } => (round_polys.len(), 2),
            ProductGkrLayerProof::RadixFour { round_polys, .. } => (round_polys.len(), 4),
            ProductGkrLayerProof::Binary { .. } => {
                panic!("a batched table has no plain binary layer")
            }
        };
        for (round, node) in (0..rounds).flat_map(|round| (0..5).map(move |node| (round, node))) {
            let mut bad = proof.clone();
            match &mut bad.layers[layer] {
                ProductGkrLayerProof::BinarySumcheck { round_polys, .. }
                | ProductGkrLayerProof::RadixFour { round_polys, .. } => {
                    round_polys[round][node] += Binary::ONE;
                }
                ProductGkrLayerProof::Binary { .. } => unreachable!(),
            }
            assert!(rejects(&bad), "layer {layer} round {round} node {node}");
        }
        for child in 0..children {
            let mut bad = proof.clone();
            match &mut bad.layers[layer] {
                ProductGkrLayerProof::BinarySumcheck { children, .. } => {
                    children[0][child] += Binary::ONE;
                }
                ProductGkrLayerProof::RadixFour { children, .. } => {
                    children[0][child] += Binary::ONE;
                }
                ProductGkrLayerProof::Binary { .. } => unreachable!(),
            }
            assert!(rejects(&bad), "layer {layer} child {child}");
        }
    }
}

#[test]
fn a_binary_layer_of_the_wrong_kind_is_malformed() {
    let mut rng = Xoroshiro128Plus::seed_from_u64(0xBA7C_0005);
    let shape = ProductGkrShape::batched(3, 1).unwrap();
    let mut proof = round_trip::<Binary, Binary>(&random::<Binary>(&mut rng, 1 << 4), 3, 1);
    let ProductGkrLayerProof::BinarySumcheck { children, .. } = &proof.layers[0] else {
        panic!("an odd batched height starts with a summed binary layer");
    };
    proof.layers[0] = ProductGkrLayerProof::Binary {
        children: children.clone(),
    };
    assert_eq!(
        proof.verify::<Binary, _>(shape, &mut challenger::<Binary>()),
        Err(ProductGkrError::MalformedLayer { layer: 0 })
    );
}

#[test]
fn a_batched_table_must_fit_one_word() {
    let maximum = usize::BITS as usize - 1;
    assert_eq!(
        ProductGkrShape::batched(maximum, 1),
        Err(ProductGkrShapeError::HeightOverflow {
            log_height: maximum + 1,
            maximum,
        })
    );
    assert_eq!(
        ProductGkrShape::batched(3, 2).unwrap().root_message_len(),
        4
    );
}

#[test]
fn a_batched_shape_charges_its_batch_point() {
    use p3_security::gkr::{PRODUCT_GKR_BATCH_POINT_LABEL, PRODUCT_GKR_BATCHING_LABEL};

    let terms = ProductGkrShape::batched(5, 3)
        .unwrap()
        .security_profile()
        .components(192);
    let labels: Vec<_> = terms.iter().map(|term| term.label).collect();
    assert!(labels.contains(&PRODUCT_GKR_BATCH_POINT_LABEL));
    // One table means no random combination of trees.
    assert!(!labels.contains(&PRODUCT_GKR_BATCHING_LABEL));
}
