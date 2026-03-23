//! Blake3 leaves + Poseidon2 inner compression for `MerkleTreeMmcs`.
//!
//! When [`BabyBear`]'s [`Field::Packing`](p3_field::Field::Packing) is `Self` (Monty31 falls back to
//! scalar packing if NEON/AVX target features are off), a single [`FieldLeafFromByteHash`] is enough.
//!
//! When SIMD packing is enabled, `P != F` and the [`byte_hash_merkle_leaf_pair`] macro supplies both
//! leaf modes. See `p3_monty_31::monty_31` `Field` impl for `MontyField31`.

#[cfg(not(any(
    all(target_arch = "aarch64", target_feature = "neon"),
    all(target_arch = "x86_64", target_feature = "avx2"),
    all(target_arch = "x86_64", target_feature = "avx512f"),
)))]
mod mixed_merkle_without_simd_packing {
    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_blake3::Blake3;
    use p3_commit::Mmcs;
    use p3_field::Field;
    use p3_matrix::Matrix;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_symmetric::{FieldLeafFromByteHash, TruncatedPermutation};
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    type F = BabyBear;
    type P = <F as Field>::Packing;

    trait Same {}
    impl<T> Same for T {}

    fn _field_packing_equals_scalar() {
        fn eq<A: Same<B>, B>() {}
        eq::<F, P>();
    }

    type Perm = Poseidon2BabyBear<16>;
    type Compress = TruncatedPermutation<Perm, 2, 8, 16>;
    type Leaf = FieldLeafFromByteHash<Blake3, F, 32, 8>;
    type MmcsTy = MerkleTreeMmcs<F, F, Leaf, Compress, 2, 8>;

    #[test]
    fn blake3_leaf_poseidon2_compress_without_simd_packing() {
        _field_packing_equals_scalar();

        let mut rng = SmallRng::seed_from_u64(11);
        let perm = Perm::new_from_rng_128(&mut rng);
        let compress = Compress::new(perm);
        let leaf = Leaf::new(Blake3);
        let mmcs = MmcsTy::new(leaf, compress, 0);

        let mat = RowMajorMatrix::<F>::rand(&mut rng, 32, 4);
        let dims = [mat.dimensions()];
        let (commit, prover_data) = mmcs.commit(vec![mat]);
        let index = 11;
        let opening = mmcs.open_batch(index, &prover_data);
        mmcs.verify_batch(&commit, &dims, index, (&opening).into())
            .expect("Blake3 leaf + Poseidon2 compress (scalar packing) should verify");
    }
}

#[cfg(any(
    all(target_arch = "aarch64", target_feature = "neon"),
    all(target_arch = "x86_64", target_feature = "avx2"),
    all(target_arch = "x86_64", target_feature = "avx512f"),
))]
mod mixed_merkle_with_simd_packing {
    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_blake3::Blake3;
    use p3_commit::Mmcs;
    use p3_field::Field;
    use p3_matrix::Matrix;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_symmetric::{TruncatedPermutation, byte_hash_merkle_leaf_pair};
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    type PackedBabyBear = p3_baby_bear::PackedBabyBearNeon;

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        not(target_feature = "avx512f")
    ))]
    type PackedBabyBear = p3_baby_bear::PackedBabyBearAVX2;

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    type PackedBabyBear = p3_baby_bear::PackedBabyBearAVX512;

    type F = BabyBear;
    type PF = <F as Field>::Packing;

    trait SameAs<B> {}
    impl<T> SameAs<T> for T {}

    #[test]
    fn field_packing_alias_matches_baby_bear() {
        fn check<T: SameAs<U>, U>() {}
        check::<PF, PackedBabyBear>();
    }

    byte_hash_merkle_leaf_pair!(
        /// Blake3 row hash reduced to eight BabyBear limbs; pairs with SIMD `F::Packing`.
        pub HybridBlake3Leaf,
        Blake3,
        F,
        PackedBabyBear,
        32,
        8
    );

    type Perm = Poseidon2BabyBear<16>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;

    #[test]
    fn blake3_leaf_poseidon2_compress_with_simd_packing() {
        let mut rng = SmallRng::seed_from_u64(7);
        let perm = Perm::new_from_rng_128(&mut rng);
        let compress = MyCompress::new(perm);

        let leaf = HybridBlake3Leaf(Blake3);
        type HybridMmcs = MerkleTreeMmcs<PF, PF, HybridBlake3Leaf, MyCompress, 2, 8>;
        let mmcs = HybridMmcs::new(leaf, compress, 0);

        let mat = RowMajorMatrix::<F>::rand(&mut rng, 32, 4);
        let dims = [mat.dimensions()];
        let (commit, prover_data) = mmcs.commit(vec![mat]);
        let index = 11;
        let opening = mmcs.open_batch(index, &prover_data);
        mmcs.verify_batch(&commit, &dims, index, (&opening).into())
            .expect("Blake3 leaf + Poseidon2 compress (SIMD packing) should verify");
    }
}
