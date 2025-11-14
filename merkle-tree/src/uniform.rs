use alloc::vec;
use alloc::vec::Vec;
use core::array;

use p3_field::PackedValue;
use p3_matrix::Matrix;
use p3_maybe_rayon::prelude::*;
use p3_symmetric::{PseudoCompressionFunction, StatefulSponge};

/// Compress a layer of digests in a uniform Merkle tree.
///
/// Assumes the layer length is a power of two and returns exactly half as many digests.
pub fn compress_uniform<P, C, const DIGEST_ELEMS: usize>(
    prev_layer: &[[P::Value; DIGEST_ELEMS]],
    c: &C,
) -> Vec<[P::Value; DIGEST_ELEMS]>
where
    P: PackedValue,
    C: PseudoCompressionFunction<[P::Value; DIGEST_ELEMS], 2>
        + PseudoCompressionFunction<[P; DIGEST_ELEMS], 2>
        + Sync,
{
    let width = P::WIDTH;
    let next_len = prev_layer.len() / 2;

    let default_digest = [P::Value::default(); DIGEST_ELEMS];
    let mut next_digests = vec![default_digest; next_len];

    next_digests
        .par_chunks_exact_mut(width)
        .enumerate()
        .for_each(|(i, digests_chunk)| {
            let first_state = i * width;
            let left = array::from_fn(|j| P::from_fn(|k| prev_layer[2 * (first_state + k)][j]));
            let right =
                array::from_fn(|j| P::from_fn(|k| prev_layer[2 * (first_state + k) + 1][j]));
            let packed_digest = c.compress([left, right]);
            for (dst, src) in digests_chunk.iter_mut().zip(unpack_array(packed_digest)) {
                *dst = src;
            }
        });

    for i in (next_len / width * width)..next_len {
        let left = prev_layer[2 * i];
        let right = prev_layer[2 * i + 1];
        next_digests[i] = c.compress([left, right]);
    }

    next_digests
}

/// Build the leaf digests that would appear at the base of a uniform Merkle tree.
///
/// Matrices are processed from shortest to tallest. For each matrix we rebuild a scratch slice of
/// scalar states sized to that matrix, pack it into SIMD lanes, absorb the matrix-provided inputs
/// into the sponge state, then write the updated lanes back into the canonical per-leaf state
/// buffer. This routine assumes every matrix height and the packing width `P::WIDTH` are powers of
/// two.
pub fn build_uniform_leaves<
    P,
    H,
    M,
    const WIDTH: usize,
    const RATE: usize,
    const DIGEST_ELEMS: usize,
>(
    mut matrices: Vec<M>,
    sponge: &H,
) -> Vec<[P::Value; DIGEST_ELEMS]>
where
    P: PackedValue + Default,
    H: StatefulSponge<P, WIDTH, RATE> + StatefulSponge<P::Value, WIDTH, RATE> + Sync,
    M: Matrix<P::Value>,
{
    assert!(!matrices.is_empty(), "matrices cannot be empty");

    matrices.sort_by_key(|m| m.height());

    let _: () = { assert!(P::WIDTH.is_power_of_two()) };

    let initial_height = matrices[0].height();

    // Sanity-check matrix heights once so the main loop can assume clean invariants.
    for matrix in &matrices {
        let height = matrix.height();
        debug_assert!(height > 0, "matrix height must be non-zero");
        debug_assert!(
            height.is_power_of_two(),
            "matrix height {} must be a power of two",
            height
        );
    }

    let final_height = matrices.last().unwrap().height();
    let final_packed = final_height.div_ceil(P::WIDTH);

    let scalar_default = [P::Value::default(); WIDTH];
    let mut states = vec![scalar_default; final_height]; // canonical per-leaf scalar states
    let mut temp_states = vec![scalar_default; final_height]; // scratch buffer when growing matrices
    let mut packed_states = vec![[P::default(); WIDTH]; final_packed.max(1)]; // SIMD-aligned states

    let mut active_height = initial_height;

    // Process matrices from shortest to tallest, expanding the canonical states as we go.
    for matrix in matrices.iter() {
        let height = matrix.height();

        if height < P::WIDTH {
            for lane in 0..height {
                let state = &mut temp_states[lane];
                let row = matrix.row(lane).expect("row must exist for narrow absorb");
                sponge.absorb(state, row.into_iter());
            }
            states[..height].copy_from_slice(&temp_states[..height]);
        } else {
            let scaling_factor = height / active_height;

            // Copy `states` into `temp_slice`, repeating each entry `scaling_factor` times
            temp_states[..height]
                .par_chunks_mut(scaling_factor)
                .zip(states[..active_height].par_iter())
                .for_each(|(chunk, state)| chunk.fill(*state));

            // Pack the replicated scalar states into SIMD-friendly buffers.
            let packed_height = height.div_ceil(P::WIDTH);
            packed_states
                .par_iter_mut()
                .enumerate()
                .for_each(|(packed_idx, packed_state)| {
                    let base_state = packed_idx * P::WIDTH;
                    *packed_state = array::from_fn(|col| {
                        P::from_fn(|lane| temp_states[base_state + lane][col])
                    });
                });

            // Absorb the packed rows from the matrix into the sponge states.
            (0..packed_height).into_par_iter().for_each(|packed_idx| {
                let base_state = packed_idx * P::WIDTH;
                sponge.absorb(
                    &mut packed_states[packed_idx],
                    matrix.vertically_packed_row(base_state),
                );
            });

            // Scatter the updated SIMD states back into the canonical scalar layout.
            states
                .par_chunks_mut(P::WIDTH)
                .zip(packed_states.par_iter())
                .for_each(|(chunk, packed)| {
                    let lane_count = chunk.len();
                    for lane in 0..lane_count {
                        for col in 0..WIDTH {
                            chunk[lane][col] = packed[col].as_slice()[lane];
                        }
                    }
                });
        }

        active_height = height;
    }

    states
        .into_iter()
        .map(|mut state| sponge.squeeze::<DIGEST_ELEMS>(&mut state))
        .collect()
}

/// Build all digest layers from leaf digests up to the root.
pub fn build_digest_layers<P, C, const DIGEST_ELEMS: usize>(
    leaf_digests: Vec<[P::Value; DIGEST_ELEMS]>,
    c: &C,
) -> Vec<Vec<[P::Value; DIGEST_ELEMS]>>
where
    P: PackedValue,
    C: PseudoCompressionFunction<[P::Value; DIGEST_ELEMS], 2>
        + PseudoCompressionFunction<[P; DIGEST_ELEMS], 2>
        + Sync,
{
    assert!(!leaf_digests.is_empty(), "leaf_digests cannot be empty");

    let mut digest_layers = vec![leaf_digests];

    loop {
        let prev_layer = digest_layers.last().unwrap();
        if prev_layer.len() == 1 {
            break;
        }

        let next_layer = compress_uniform::<P, C, DIGEST_ELEMS>(prev_layer, c);
        digest_layers.push(next_layer);
    }

    digest_layers
}

/// Unpack a packed digest into an iterator of scalar digests.
///
/// Each call yields one lane worth of scalars across all digest elements.
fn unpack_array<P: PackedValue, const N: usize>(
    packed_digest: [P; N],
) -> impl Iterator<Item = [P::Value; N]> {
    (0..P::WIDTH).map(move |j| packed_digest.map(|p| p.as_slice()[j]))
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_field::Field;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_matrix::Matrix;
    use p3_symmetric::{PaddingFreeSponge, StatefulSponge, TruncatedPermutation};
    use rand::{rngs::SmallRng, SeedableRng};

    use super::*;

    type F = BabyBear;
    type Packed = <F as Field>::Packing;

    const WIDTH: usize = 16;
    const RATE: usize = 8;
    const DIGEST: usize = 8;
    const PACK_WIDTH: usize = <Packed as PackedValue>::WIDTH;

    fn poseidon_components() -> (
        PaddingFreeSponge<Poseidon2BabyBear<WIDTH>, WIDTH, RATE, DIGEST>,
        TruncatedPermutation<Poseidon2BabyBear<WIDTH>, 2, DIGEST, WIDTH>,
    ) {
        let mut rng = SmallRng::seed_from_u64(1);
        let permutation = Poseidon2BabyBear::<WIDTH>::new_from_rng_128(&mut rng);
        let sponge = PaddingFreeSponge::<_, WIDTH, RATE, DIGEST>::new(permutation.clone());
        let compressor = TruncatedPermutation::<_, 2, DIGEST, WIDTH>::new(permutation);
        (sponge, compressor)
    }

    fn field_matrix(rows: usize, cols: usize, offset: u32) -> RowMajorMatrix<F> {
        let data = (0..rows * cols)
            .map(|i| F::new(offset + i as u32))
            .collect::<Vec<_>>();
        RowMajorMatrix::new(data, cols)
    }

    fn reference_uniform_leaves(
        mut matrices: Vec<RowMajorMatrix<F>>,
        sponge: &PaddingFreeSponge<Poseidon2BabyBear<WIDTH>, WIDTH, RATE, DIGEST>,
    ) -> Vec<[F; DIGEST]> {
        matrices.sort_by_key(|m| m.height());
        assert!(!matrices.is_empty());

        let final_height = matrices.last().unwrap().height();
        let mut states = vec![[F::ZERO; WIDTH]; final_height];
        let mut scratch = states.clone();
        let mut active_height = matrices.first().unwrap().height();

        for matrix in matrices.iter() {
            let height = matrix.height();

            if height < PACK_WIDTH {
                for row in 0..height {
                    let mut state = scratch[row];
                    let row_iter = matrix.row(row).expect("row exists").into_iter();
                    sponge.absorb(&mut state, row_iter);
                    scratch[row] = state;
                }
                states[..height].copy_from_slice(&scratch[..height]);
            } else {
                let growth = height / active_height;
                for row in 0..height {
                    scratch[row] = states[row / growth];
                }
                for row in 0..height {
                    let mut state = scratch[row];
                    let row_iter = matrix.row(row).expect("row exists").into_iter();
                    sponge.absorb(&mut state, row_iter);
                    scratch[row] = state;
                }
                states[..height].copy_from_slice(&scratch[..height]);
            }

            active_height = height;
        }

        states
            .iter_mut()
            .map(|state| sponge.squeeze::<DIGEST>(state))
            .collect()
    }

    #[test]
    fn uniform_leaves_match_reference() {
        let (sponge, _) = poseidon_components();

        let small_height = if PACK_WIDTH > 1 { PACK_WIDTH / 2 } else { 1 };
        let large_height = PACK_WIDTH * 2;
        assert!(small_height.is_power_of_two() && small_height > 0);
        assert!(large_height.is_power_of_two());

        let small = field_matrix(small_height, 3, 1);
        let large = field_matrix(large_height, 5, 1_000);

        let matrices = vec![small.clone(), large.clone()];
        let leaves =
            build_uniform_leaves::<Packed, _, _, WIDTH, RATE, DIGEST>(matrices.clone(), &sponge);

        let expected = reference_uniform_leaves(matrices, &sponge);
        assert_eq!(leaves, expected);
    }

    #[test]
    fn digest_layers_match_truncated_poseidon() {
        let (sponge, compressor) = poseidon_components();

        let matrix = field_matrix(PACK_WIDTH * 2, 4, 10_000);
        let matrices = vec![matrix.clone()];

        let leaves =
            build_uniform_leaves::<Packed, _, _, WIDTH, RATE, DIGEST>(matrices.clone(), &sponge);
        let reference = reference_uniform_leaves(matrices, &sponge);
        assert_eq!(leaves, reference);

        let mut naive_layers = vec![reference.clone()];
        let mut current = reference;
        while current.len() > 1 {
            let mut next = Vec::with_capacity(current.len() / 2);
            for pair in current.chunks_exact(2) {
                next.push(compressor.compress([pair[0], pair[1]]));
            }
            naive_layers.push(next.clone());
            current = next;
        }

        let actual_layers = build_digest_layers::<Packed, _, DIGEST>(leaves, &compressor);
        assert_eq!(actual_layers, naive_layers);
    }
}
