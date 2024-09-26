use alloc::vec;
use alloc::vec::Vec;
use core::array;
use core::cmp::Reverse;

use itertools::Itertools;
use p3_field::{AbstractField, Field, PackedField};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::{Matrix, MatrixRowSlices};
use p3_maybe_rayon::prelude::*;
use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use tracing::instrument;

/// A binary Merkle tree for field data. It has leaves of type `F` and digests of type
/// `[F; DIGEST_ELEMS]`.
///
/// This generally shouldn't be used directly. If you're using a Merkle tree as an MMCS,
/// see `FieldMerkleTreeMmcs`.
#[derive(Clone, Debug, Deserialize, Eq, Serialize, PartialEq)]
#[serde(bound(serialize = "[F; DIGEST_ELEMS]: Serialize"))]
#[serde(bound(deserialize = "[F; DIGEST_ELEMS]: DeserializeOwned"))]
pub struct FieldMerkleTree<F: Field, const DIGEST_ELEMS: usize> {
    pub(crate) leaves: Vec<RowMajorMatrix<F>>,
    pub(crate) digest_layers: Vec<Vec<[F; DIGEST_ELEMS]>>,
}

impl<F: Field, const DIGEST_ELEMS: usize> FieldMerkleTree<F, DIGEST_ELEMS> {
    /// Matrix heights need not be powers of two. However, if the heights of two given matrices
    /// round up to the same power of two, they must be equal.
    #[instrument(name = "build merkle tree", level = "debug", skip_all,
                 fields(dimensions = alloc::format!("{:?}", leaves.iter().map(|l| l.dimensions()).collect::<Vec<_>>())))]
    pub fn new<P, H, C>(h: &H, c: &C, leaves: Vec<RowMajorMatrix<F>>) -> Self
    where
        P: PackedField<Scalar = F>,
        H: CryptographicHasher<F, [F; DIGEST_ELEMS]>,
        H: CryptographicHasher<P, [P; DIGEST_ELEMS]>,
        H: Sync,
        C: PseudoCompressionFunction<[F; DIGEST_ELEMS], 2>,
        C: PseudoCompressionFunction<[P; DIGEST_ELEMS], 2>,
        C: Sync,
    {
        assert!(!leaves.is_empty(), "No matrices given?");

        // check height property
        assert!(
            leaves
                .iter()
                .map(|m| m.height())
                .sorted()
                .tuple_windows()
                .all(|(curr, next)| curr == next
                    || curr.next_power_of_two() != next.next_power_of_two()),
            "matrix heights that round up to the same power of two must be equal"
        );

        let mut leaves_largest_first = leaves
            .iter()
            .sorted_by_key(|l| Reverse(l.height()))
            .peekable();

        let max_height = leaves_largest_first.peek().unwrap().height();
        let tallest_matrices = leaves_largest_first
            .peeking_take_while(|m| m.height() == max_height)
            .collect_vec();

        let mut digest_layers = vec![first_digest_layer::<P, H, DIGEST_ELEMS>(
            h,
            tallest_matrices,
        )];
        loop {
            let prev_layer = digest_layers.last().map(Vec::as_slice).unwrap_or_default();
            if prev_layer.len() == 1 {
                break;
            }
            let next_layer_len = prev_layer.len() / 2;

            // The matrices that get injected at this layer.
            let matrices_to_inject = leaves_largest_first
                .peeking_take_while(|m| m.height().next_power_of_two() == next_layer_len)
                .collect_vec();

            let next_digests =
                compress_and_inject::<P, H, C, DIGEST_ELEMS>(prev_layer, matrices_to_inject, h, c);
            digest_layers.push(next_digests);
        }

        Self {
            leaves,
            digest_layers,
        }
    }

    #[must_use]
    pub fn root(&self) -> [F; DIGEST_ELEMS] {
        self.digest_layers.last().unwrap()[0]
    }
}

fn first_digest_layer<P, H, const DIGEST_ELEMS: usize>(
    h: &H,
    tallest_matrices: Vec<&RowMajorMatrix<P::Scalar>>,
) -> Vec<[P::Scalar; DIGEST_ELEMS]>
where
    P: PackedField,
    H: CryptographicHasher<P::Scalar, [P::Scalar; DIGEST_ELEMS]>,
    H: CryptographicHasher<P, [P; DIGEST_ELEMS]>,
    H: Sync,
{
    let width = P::WIDTH;
    let max_height = tallest_matrices[0].height();
    let max_height_padded = max_height.next_power_of_two();

    let default_digest = [P::Scalar::zero(); DIGEST_ELEMS];
    let mut digests = vec![default_digest; max_height_padded];

    digests[0..max_height]
        .par_chunks_exact_mut(width)
        .enumerate()
        .for_each(|(i, digests_chunk)| {
            let first_row = i * width;
            let packed_digest: [P; DIGEST_ELEMS] = h.hash_iter(
                tallest_matrices
                    .iter()
                    .flat_map(|m| m.packed_row(first_row)),
            );
            for (dst, src) in digests_chunk.iter_mut().zip(unpack_array(packed_digest)) {
                *dst = src;
            }
        });

    // If our packing width did not divide max_height, fall back to single-threaded scalar code
    // for the last bit.
    #[allow(clippy::needless_range_loop)]
    for i in (max_height / width * width)..max_height {
        digests[i] = h.hash_iter_slices(tallest_matrices.iter().map(|m| m.row_slice(i)));
    }

    digests
}

/// Compress `n` digests from the previous layer into `n/2` digests, while potentially mixing in
/// some leaf data, if there are input matrices with (padded) height `n/2`.
fn compress_and_inject<P, H, C, const DIGEST_ELEMS: usize>(
    prev_layer: &[[P::Scalar; DIGEST_ELEMS]],
    matrices_to_inject: Vec<&RowMajorMatrix<P::Scalar>>,
    h: &H,
    c: &C,
) -> Vec<[P::Scalar; DIGEST_ELEMS]>
where
    P: PackedField,
    H: CryptographicHasher<P::Scalar, [P::Scalar; DIGEST_ELEMS]>,
    H: CryptographicHasher<P, [P; DIGEST_ELEMS]>,
    H: Sync,
    C: PseudoCompressionFunction<[P::Scalar; DIGEST_ELEMS], 2>,
    C: PseudoCompressionFunction<[P; DIGEST_ELEMS], 2>,
    C: Sync,
{
    if matrices_to_inject.is_empty() {
        return compress::<P, C, DIGEST_ELEMS>(prev_layer, c);
    }

    let width = P::WIDTH;
    let next_len = matrices_to_inject[0].height();
    let next_len_padded = prev_layer.len() / 2;

    let default_digest = [P::Scalar::zero(); DIGEST_ELEMS];
    let mut next_digests = vec![default_digest; next_len_padded];

    next_digests[0..next_len]
        .par_chunks_exact_mut(width)
        .enumerate()
        .for_each(|(i, digests_chunk)| {
            let first_row = i * width;
            let left = array::from_fn(|j| P::from_fn(|k| prev_layer[2 * (first_row + k)][j]));
            let right = array::from_fn(|j| P::from_fn(|k| prev_layer[2 * (first_row + k) + 1][j]));
            let mut packed_digest = c.compress([left, right]);
            let tallest_digest = h.hash_iter(
                matrices_to_inject
                    .iter()
                    .flat_map(|m| m.packed_row(first_row)),
            );
            packed_digest = c.compress([packed_digest, tallest_digest]);
            for (dst, src) in digests_chunk.iter_mut().zip(unpack_array(packed_digest)) {
                *dst = src;
            }
        });

    // If our packing width did not divide next_len, fall back to single-threaded scalar code
    // for the last bit.
    for i in (next_len / width * width)..next_len {
        let left = prev_layer[2 * i];
        let right = prev_layer[2 * i + 1];
        let digest = c.compress([left, right]);
        let rows_digest = h.hash_iter_slices(matrices_to_inject.iter().map(|m| m.row_slice(i)));
        next_digests[i] = c.compress([digest, rows_digest]);
    }

    // At this point, we've exceeded the height of the matrices to inject, so we continue the
    // process above except with default_digest in place of an input digest.
    for i in next_len..next_len_padded {
        let left = prev_layer[2 * i];
        let right = prev_layer[2 * i + 1];
        let digest = c.compress([left, right]);
        next_digests[i] = c.compress([digest, default_digest]);
    }

    next_digests
}

/// Compress `n` digests from the previous layer into `n/2` digests.
fn compress<P, C, const DIGEST_ELEMS: usize>(
    prev_layer: &[[P::Scalar; DIGEST_ELEMS]],
    c: &C,
) -> Vec<[P::Scalar; DIGEST_ELEMS]>
where
    P: PackedField,
    C: PseudoCompressionFunction<[P::Scalar; DIGEST_ELEMS], 2>,
    C: PseudoCompressionFunction<[P; DIGEST_ELEMS], 2>,
    C: Sync,
{
    debug_assert!(prev_layer.len().is_power_of_two());
    let width = P::WIDTH;
    let next_len = prev_layer.len() / 2;

    let default_digest = [P::Scalar::zero(); DIGEST_ELEMS];
    let mut next_digests = vec![default_digest; next_len];

    next_digests[0..next_len]
        .par_chunks_exact_mut(width)
        .enumerate()
        .for_each(|(i, digests_chunk)| {
            let first_row = i * width;
            let left = array::from_fn(|j| P::from_fn(|k| prev_layer[2 * (first_row + k)][j]));
            let right = array::from_fn(|j| P::from_fn(|k| prev_layer[2 * (first_row + k) + 1][j]));
            let packed_digest = c.compress([left, right]);
            for (dst, src) in digests_chunk.iter_mut().zip(unpack_array(packed_digest)) {
                *dst = src;
            }
        });

    // If our packing width did not divide next_len, fall back to single-threaded scalar code
    // for the last bit.
    for i in (next_len / width * width)..next_len {
        let left = prev_layer[2 * i];
        let right = prev_layer[2 * i + 1];
        let digest = c.compress([left, right]);
        next_digests[i] = digest;
    }

    next_digests
}

/// Converts a packed array `[P; N]` into its underlying `P::WIDTH` scalar arrays.
#[inline]
fn unpack_array<P: PackedField, const N: usize>(
    packed_digest: [P; N],
) -> impl Iterator<Item = [P::Scalar; N]> {
    (0..P::WIDTH).map(move |j| packed_digest.map(|p| p.as_slice()[j]))
}
