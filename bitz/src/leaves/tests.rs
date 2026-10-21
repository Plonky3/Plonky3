use alloc::vec::Vec;

use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_binary_field::{BinaryChallenger, BinaryField128, Poly64, Poly192};
use p3_challenger::fs::TranscriptField;
use p3_challenger::{CanSample, DuplexChallenger, FieldChallenger, HashChallenger};
use p3_field::extension::BinomialExtensionField;
use p3_field::{ExtensionField, Field};
use p3_gkr::{DenseLeaves, ProductGkrProof, ProductGkrShape};
use p3_keccak::Keccak256Hash;
use proptest::prelude::*;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;
use rand::{Rng, RngExt, SeedableRng};

use super::*;

/// A Keccak transcript over a binary field.
fn keccak<F: Field>() -> BinaryChallenger<F, HashChallenger<u8, Keccak256Hash, 32>> {
    BinaryChallenger::from_hasher(Vec::new(), Keccak256Hash)
}

/// A Poseidon2 transcript over BabyBear.
fn poseidon() -> DuplexChallenger<BabyBear, Poseidon2BabyBear<16>, 16, 8> {
    DuplexChallenger::new(Poseidon2BabyBear::new_from_rng_128(
        &mut SmallRng::seed_from_u64(3),
    ))
}

/// Every leaf, written out, as the dense prover reads them.
fn dense<F, EF: Field>(leaves: &BitLeaves<'_, F, EF>) -> Vec<EF> {
    let (rows, columns) = (leaves.shape.rows(), leaves.shape.columns());
    let mut out = Vec::with_capacity(leaves.images.len() * columns * rows);
    for images in leaves.images {
        for c in 0..columns {
            for (b, &image) in images.iter().enumerate() {
                let word = leaves.words[c * leaves.shape.column_words() + b / 64];
                out.push(if word >> (b % 64) & 1 == 1 {
                    image
                } else {
                    EF::ONE
                });
            }
        }
    }
    out
}

/// The proofs, outputs and transcript states of the dense and the low-entropy prover agree.
fn agree<F, EF, C>(
    sponge: fn() -> C,
    seed: u64,
    log_rows: usize,
    log_columns: usize,
    claims: usize,
    density: u32,
) where
    F: TranscriptField,
    EF: ExtensionField<F>,
    StandardUniform: Distribution<EF>,
    C: FieldChallenger<F>,
{
    let mut rng = SmallRng::seed_from_u64(seed);
    let shape = BitShape::new(log_rows, log_columns).unwrap();
    // Sparse, dense and uniform words, so some tiles are all zeros or all ones.
    let words: Vec<u64> = (0..shape.words())
        .map(|_| match density {
            0 => rng.next_u64() & rng.next_u64() & rng.next_u64(),
            1 => rng.next_u64() | rng.next_u64(),
            _ => rng.next_u64(),
        })
        .collect();
    let images: Vec<Vec<EF>> = (0..claims)
        .map(|_| (0..shape.rows()).map(|_| rng.random()).collect())
        .collect();
    let leaves = BitLeaves::<F, EF> {
        words: &words,
        shape,
        images: &images,
        base: PhantomData,
    };
    let log_trees =
        (claims.next_power_of_two().trailing_zeros() as usize).max(usize::from(log_columns == 0));
    let gkr = ProductGkrShape::batched(log_rows, log_trees + log_columns).unwrap();

    let mut expected_sponge = sponge();
    let expected = ProductGkrProof::prove_batched::<F, _, _>(
        &DenseLeaves(dense(&leaves)),
        gkr,
        &mut expected_sponge,
    );
    let mut actual_sponge = sponge();
    let actual = ProductGkrProof::prove_batched::<F, _, _>(&leaves, gkr, &mut actual_sponge);
    assert_eq!(actual, expected);
    assert_eq!(
        CanSample::<F>::sample(&mut actual_sponge),
        CanSample::<F>::sample(&mut expected_sponge)
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    #[test]
    fn poly192_proofs_match_the_dense_prover(
        seed: u64, log_rows in 6usize..=14, log_columns in 0usize..=8, claims in 1usize..=3, density in 0u32..3,
    ) {
        prop_assume!(log_rows + log_columns <= 14);
        agree::<Poly64, Poly192, _>(keccak, seed, log_rows, log_columns, claims, density);
    }

    #[test]
    fn tower_proofs_match_the_dense_prover(
        seed: u64, log_rows in 6usize..=11, log_columns in 0usize..=4, claims in 1usize..=2, density in 0u32..3,
    ) {
        agree::<BinaryField128, BinaryField128, _>(keccak, seed, log_rows, log_columns, claims, density);
    }

    #[test]
    fn odd_characteristic_proofs_match_the_dense_prover(
        seed: u64, log_rows in 6usize..=11, log_columns in 0usize..=4, claims in 1usize..=2, density in 0u32..3,
    ) {
        agree::<BabyBear, BinomialExtensionField<BabyBear, 4>, _>(poseidon, seed, log_rows, log_columns, claims, density);
    }
}

#[test]
fn wide_shapes_match_the_dense_prover() {
    // Full tiles of eight words, several column chunks, and a padded tree.
    agree::<Poly64, Poly192, _>(keccak, 1, 14, 2, 3, 2);
    agree::<Poly64, Poly192, _>(keccak, 2, 9, 9, 1, 2);
}

#[test]
fn word_products_are_the_level_above_three_layers() {
    let mut rng = SmallRng::seed_from_u64(7);
    let shape = BitShape::new(10, 3).unwrap();
    let words: Vec<u64> = (0..shape.words()).map(|_| rng.next_u64()).collect();
    let images: Vec<Vec<Poly192>> = (0..2)
        .map(|_| (0..shape.rows()).map(|_| rng.random()).collect())
        .collect();
    let leaves = BitLeaves::<Poly64, Poly192> {
        words: &words,
        shape,
        images: &images,
        base: PhantomData,
    };
    let expected: Vec<Poly192> = dense(&leaves)
        .chunks(64)
        .map(|word| word.iter().copied().product())
        .collect();
    assert_eq!(leaves.materialize().into_owned(), expected);
}
