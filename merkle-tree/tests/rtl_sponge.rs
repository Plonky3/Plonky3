use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_commit::Mmcs;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CryptographicHasher, RtlPaddingFreeSponge, TruncatedPermutation};
use rand::SeedableRng;
use rand::rngs::SmallRng;

type F = BabyBear;
type Packing = <F as Field>::Packing;
type Perm = Poseidon2BabyBear<16>;
type Hash = RtlPaddingFreeSponge<Perm, 16, 8, 8>;
type Compress = TruncatedPermutation<Perm, 2, 8, 16>;
type PackedMmcs = MerkleTreeMmcs<Packing, Packing, Hash, Compress, 2, 8>;
type ScalarMmcs = MerkleTreeMmcs<F, F, Hash, Compress, 2, 8>;

fn setup() -> (Hash, PackedMmcs, ScalarMmcs) {
    let perm = Perm::new_from_rng_128(&mut SmallRng::seed_from_u64(1));
    let hash = Hash::new(perm.clone());
    let compress = Compress::new(perm);
    let packed = PackedMmcs::new(hash.clone(), compress.clone(), 0);
    let scalar = ScalarMmcs::new(hash.clone(), compress, 0);
    (hash, packed, scalar)
}

#[test]
fn single_leaf_root_is_the_sponge_digest() {
    // One row of 13 cells: a partial last block, so the zero padding is exercised.
    let (hash, packed, _) = setup();
    let row: Vec<F> = (1..=13).map(F::from_u32).collect();

    let (commitment, _) = packed.commit_matrix(RowMajorMatrix::new_row(row.clone()));
    assert_eq!(commitment.roots(), [hash.hash_slice(&row)]);
}

#[test]
fn commits_open_and_verify() {
    // Two heights, so the shorter matrix is injected into an upper layer.
    // Widths 7, 8 and 13 cover a partial block, a whole block, and more than one block.
    let (_, packed, scalar) = setup();
    let mut rng = SmallRng::seed_from_u64(2);
    let matrices = vec![
        RowMajorMatrix::<F>::rand(&mut rng, 64, 7),
        RowMajorMatrix::<F>::rand(&mut rng, 64, 8),
        RowMajorMatrix::<F>::rand(&mut rng, 16, 13),
    ];
    let dims: Vec<_> = matrices.iter().map(Matrix::dimensions).collect();

    // The packed and scalar trees hash through different entry points and must agree.
    let (commitment, data) = packed.commit(matrices.clone());
    assert_eq!(scalar.commit(matrices).0, commitment);

    for index in [0, 17, 63] {
        let opening = packed.open_batch(index, &data);
        packed
            .verify_batch(&commitment, &dims, index, (&opening).into())
            .unwrap();

        // Any change to an opened cell breaks the proof.
        let mut tampered = opening.clone();
        tampered.opened_values[2][0] += F::ONE;
        assert!(
            packed
                .verify_batch(&commitment, &dims, index, (&tampered).into())
                .is_err()
        );
    }
}
