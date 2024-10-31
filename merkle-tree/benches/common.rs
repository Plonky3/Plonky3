#![allow(dead_code)]

use std::collections::HashMap;

use p3_baby_bear::{BabyBear, DiffusionMatrixBabyBear};
use p3_blake3::Blake3;
use p3_matrix::dense::RowMajorMatrix;
use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
use p3_symmetric::{CompressionFunctionFromHasher, PaddingFreeSponge, TruncatedPermutation};
use rand::{thread_rng, Rng};

type PermPoseidon =
    Poseidon2<BabyBear, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 16, 7>;
type HPoseidon = PaddingFreeSponge<PermPoseidon, 16, 8, 8>;
type CPoseidon = TruncatedPermutation<PermPoseidon, 2, 8, 16>;

type CBlake3 = CompressionFunctionFromHasher<Blake3, 2, 32>;

pub fn get_random_leaves(
    num_matrices: usize,
    max_rows: usize,
    max_cols: usize,
) -> Vec<RowMajorMatrix<BabyBear>> {
    let mut pow2_to_size = HashMap::new();
    (0..num_matrices)
        .map(|_| {
            let mut n_rows = rand::thread_rng().gen_range(1..max_rows);
            let n_cols = rand::thread_rng().gen_range(1..max_cols);

            // Same-power-of-two row numbers must match
            n_rows = pow2_to_size
                .entry(n_rows.next_power_of_two())
                .or_insert(n_rows)
                .to_owned();

            RowMajorMatrix::<BabyBear>::rand(&mut thread_rng(), n_rows, n_cols)
        })
        .collect()
}

pub fn poseidon2_permutation() -> PermPoseidon {
    PermPoseidon::new_from_rng_128(
        Poseidon2ExternalMatrixGeneral,
        DiffusionMatrixBabyBear::default(),
        &mut thread_rng(),
    )
}

pub fn poseidon2_hasher() -> HPoseidon {
    HPoseidon::new(poseidon2_permutation())
}

pub fn poseidon2_compressor() -> CPoseidon {
    CPoseidon::new(poseidon2_permutation())
}

pub fn blake3_compressor() -> CBlake3 {
    CBlake3::new(Blake3 {})
}

#[macro_export]
macro_rules! bench_merkle_tree {
    ($group:expr, $name:expr, $h:expr, $c:expr, $leaves:ident, $tree_type:ident, $p:ident, $pw:ident) => {
        $group.bench_function(BenchmarkId::from_parameter($name), |b| {
            b.iter_batched(
                || $leaves.clone(),
                |input| $tree_type::new::<$p, $pw, _, _>($h, $c, input),
                BatchSize::SmallInput,
            )
        })
    };
}

#[macro_export]
macro_rules! bench_hybrid_merkle_tree {
    ($group:expr, $name:expr, $h:expr, $c:expr, $leaves:ident) => {
        bench_merkle_tree!(
            $group,
            $name,
            $h,
            $c,
            $leaves,
            HybridMerkleTree,
            BabyBear,
            BabyBear
        )
    };
}

#[macro_export]
macro_rules! bench_plain_merkle_tree {
    ($group:expr, $name:expr, $h:expr, $c:expr, $leaves:ident, $width:expr) => {
        if $width == 1 {
            bench_merkle_tree!($group, $name, $h, $c, $leaves, MerkleTree, BabyBear, BabyBear)
        } else if $width == 4 {
            bench_merkle_tree!(
                $group,
                $name,
                $h,
                $c,
                $leaves,
                MerkleTree,
                BabyBearPacking,
                BabyBearPacking
            )
        } else {
            panic!("Unsupported width: {}", $width);
        }
    };
}
