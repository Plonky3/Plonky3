use alloc::vec;
use alloc::vec::Vec;

use p3_baby_bear::BabyBear;
use p3_commit::{BatchOpeningRef, Mmcs};
use p3_field::Field;
use p3_goldilocks::Goldilocks;
use p3_keccak::{KeccakF, VECTOR_LEN};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_symmetric::{
    CompressionFunctionFromHasher, CryptographicHasher, PaddingFreeSponge, SerializingHasher,
};
use rand::SeedableRng;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::{SmallRng, StdRng};
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::{MerkleTreeHidingMmcs, MerkleTreeMmcs};

// Keep the pre-staging iterator path as an independent commitment reference.
#[derive(Clone)]
struct Unstaged<H>(H);

impl<T: Clone, Out, H: CryptographicHasher<T, Out>> CryptographicHasher<T, Out> for Unstaged<H> {
    fn hash_iter<I: IntoIterator<Item = T>>(&self, input: I) -> Out {
        self.0.hash_iter(input)
    }
}

type Sponge = PaddingFreeSponge<KeccakF, 25, 17, 4>;
type Hasher = SerializingHasher<Sponge>;
type Compression<const N: usize> = CompressionFunctionFromHasher<Sponge, N, 4>;
type MmcsImpl<F, H, const N: usize> =
    MerkleTreeMmcs<[F; VECTOR_LEN], [u64; VECTOR_LEN], H, Compression<N>, N, 4>;

#[test]
fn contiguous_input_hint_targets_packed_32_bit_serialization() {
    let hints = [
        <Hasher as CryptographicHasher<BabyBear, [u64; 4]>>::PREFER_CONTIGUOUS_INPUT,
        <Hasher as CryptographicHasher<[BabyBear; VECTOR_LEN], [[u64; VECTOR_LEN]; 4]>>::PREFER_CONTIGUOUS_INPUT,
        <Hasher as CryptographicHasher<[Goldilocks; VECTOR_LEN], [[u64; VECTOR_LEN]; 4]>>::PREFER_CONTIGUOUS_INPUT,
    ];
    assert_eq!(hints, [false, VECTOR_LEN > 1, false]);
}

fn check_layout<F, const N: usize>(dimensions: &[(usize, usize)], cap_height: usize)
where
    F: Field + Serialize + DeserializeOwned,
    StandardUniform: Distribution<F>,
{
    let mut rng = SmallRng::seed_from_u64(73);
    let matrices: Vec<_> = dimensions
        .iter()
        .map(|&(height, width)| RowMajorMatrix::<F>::rand(&mut rng, height, width))
        .collect();
    let sponge = Sponge::new(KeccakF);
    let hasher = Hasher::new(sponge);
    let staged = MmcsImpl::<F, _, N>::new(hasher, Compression::new(sponge), cap_height);
    let reference =
        MmcsImpl::<F, _, N>::new(Unstaged(hasher), Compression::new(sponge), cap_height);
    let (cap, data) = staged.commit(matrices.clone());
    let (reference_cap, reference_data) = reference.commit(matrices);

    assert_eq!(cap, reference_cap);
    assert_eq!(data.digest_layers, reference_data.digest_layers);
    assert_eq!(data.arity_schedule, reference_data.arity_schedule);

    let dims: Vec<_> = data.leaves.iter().map(Matrix::dimensions).collect();
    let height = dimensions.iter().map(|d| d.0).max().unwrap();
    for index in [0, height / 2, height - 1] {
        let opening = staged.open_batch(index, &data);
        let reference_opening = reference.open_batch(index, &reference_data);
        assert_eq!(opening.opened_values, reference_opening.opened_values);
        assert_eq!(opening.opening_proof, reference_opening.opening_proof);
        reference
            .verify_batch(
                &cap,
                &dims,
                index,
                BatchOpeningRef::new(&opening.opened_values, &opening.opening_proof),
            )
            .unwrap();

        let mut corrupted = opening.opened_values.clone();
        corrupted[0][0] += F::ONE;
        assert!(
            reference
                .verify_batch(
                    &cap,
                    &dims,
                    index,
                    BatchOpeningRef::new(&corrupted, &opening.opening_proof)
                )
                .is_err()
        );
    }
}

fn check_shapes<F, const N: usize>()
where
    F: Field + Serialize + DeserializeOwned,
    StandardUniform: Distribution<F>,
{
    // Odd field counts cross u64 serialization boundaries; 34 BabyBear elements
    // fill the sponge rate exactly. Odd heights exercise the scalar remainder.
    for height in [1, VECTOR_LEN, VECTOR_LEN + 1, 31, 128] {
        for widths in [
            vec![1, 1],
            vec![3, 5],
            vec![17, 17],
            vec![33, 35, 1],
            vec![2633, 1],
        ] {
            let dims: Vec<_> = widths.into_iter().map(|w| (height, w)).collect();
            check_layout::<F, N>(&dims, 0);
        }
    }
    for cap_height in [0, 1, 2] {
        // Multiple injected rows at two heights, with a binary bridge for N=4.
        check_layout::<F, N>(
            &[(65, 3), (33, 17), (65, 5), (17, 1), (33, 19), (17, 2)],
            cap_height,
        );
    }
    // More packed batches than the worker pool normally needs.
    check_layout::<F, N>(&[(4097, 3), (4097, 5), (2049, 17), (2049, 19)], 0);
}

#[test]
fn packed_babybear_rows_preserve_binary_commitments() {
    check_shapes::<BabyBear, 2>();
}

#[test]
fn packed_babybear_rows_preserve_quaternary_commitments() {
    check_shapes::<BabyBear, 4>();
}

#[test]
fn packed_goldilocks_rows_preserve_binary_commitments() {
    check_shapes::<Goldilocks, 2>();
}

#[test]
fn packed_goldilocks_rows_preserve_quaternary_commitments() {
    check_shapes::<Goldilocks, 4>();
}

#[test]
fn packed_rows_preserve_salted_commitments() {
    type Hiding<H> = MerkleTreeHidingMmcs<
        [BabyBear; VECTOR_LEN],
        [u64; VECTOR_LEN],
        H,
        Compression<4>,
        StdRng,
        4,
        4,
        5,
    >;
    let mut rng = SmallRng::seed_from_u64(73);
    let matrices: Vec<_> = [(65, 3), (33, 17), (65, 5), (33, 19)]
        .into_iter()
        .map(|(height, width)| RowMajorMatrix::<BabyBear>::rand(&mut rng, height, width))
        .collect();
    let dims: Vec<_> = matrices.iter().map(Matrix::dimensions).collect();
    let sponge = Sponge::new(KeccakF);
    let hasher = Hasher::new(sponge);
    // Independently seed both MMCSs: cloning a hiding MMCS forks its salt stream.
    let staged = Hiding::new(
        hasher,
        Compression::new(sponge),
        1,
        StdRng::seed_from_u64(91),
    );
    let reference = Hiding::new(
        Unstaged(hasher),
        Compression::new(sponge),
        1,
        StdRng::seed_from_u64(91),
    );
    let (cap, data) = staged.commit(matrices.clone());
    let (reference_cap, reference_data) = reference.commit(matrices);
    assert_eq!(cap, reference_cap);
    assert_eq!(data.digest_layers, reference_data.digest_layers);
    for index in [0, 32, 64] {
        let opening = staged.open_batch(index, &data);
        let reference_opening = reference.open_batch(index, &reference_data);
        assert_eq!(opening.opened_values, reference_opening.opened_values);
        assert_eq!(opening.opening_proof, reference_opening.opening_proof);
        reference
            .verify_batch(
                &cap,
                &dims,
                index,
                BatchOpeningRef::new(&opening.opened_values, &opening.opening_proof),
            )
            .unwrap();
    }
}
