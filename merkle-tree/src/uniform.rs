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

    let pack_width = P::WIDTH;
    debug_assert!(
        pack_width.is_power_of_two(),
        "packing width {} must be a power of two",
        pack_width
    );

    let initial_height = matrices[0].height();

    // Sanity-check matrix heights once so the main loop can assume clean invariants.
    let mut prev_height = initial_height;
    for (idx, matrix) in matrices.iter().enumerate() {
        let height = matrix.height();
        debug_assert!(height > 0, "matrix height must be non-zero");
        debug_assert!(
            height.is_power_of_two(),
            "matrix height {} must be a power of two",
            height
        );
        if idx > 0 {
            debug_assert!(
                height % prev_height == 0,
                "matrix height {} must be a multiple of the previous height {}",
                height,
                prev_height
            );
        }
        prev_height = height;
    }
    let final_height = matrices.last().unwrap().height();
    let final_packed = final_height.div_ceil(pack_width);

    let scalar_default = [P::Value::default(); WIDTH];
    let mut scalar_states = vec![scalar_default; final_height]; // canonical per-leaf scalar states
    let mut working_scalar = vec![scalar_default; final_height]; // scratch buffer when growing matrices
    let mut packed_states = vec![[P::default(); WIDTH]; final_packed.max(1)]; // SIMD-aligned states

    let mut active_height = initial_height;

    // Process matrices from shortest to tallest, expanding the canonical states as we go.
    for matrix in matrices.iter() {
        let height = matrix.height();

        let growth = height / active_height;

        // Replicate each canonical state `growth` times into the scratch buffer so we can absorb
        // the wider matrix input.
        working_scalar[..height]
            .par_chunks_mut(growth)
            .zip(scalar_states[..active_height].par_iter())
            .for_each(|(chunk, state)| chunk.fill(*state));

        let required_packed = height.div_ceil(pack_width);
        if height < pack_width {
            // Narrow matrices fit in a single SIMD packet; absorb lane-by-lane using scalar states.
            (0..height).for_each(|lane| {
                let state = &mut working_scalar[lane];
                let row_slice = matrix
                    .row_slice(lane)
                    .expect("row must exist for narrow sponge absorb");
                let row_iter = row_slice.iter().copied();
                sponge.absorb(state, row_iter);
            });
            scalar_states[..height].copy_from_slice(&working_scalar[..height]);
        } else {
            // Full-width matrices align with SIMD lanes; pack and absorb every chunk without padding.
            pack_full_states::<P, WIDTH>(
                &working_scalar[..height],
                &mut packed_states[..required_packed],
            );

            (0..required_packed).into_par_iter().for_each(|packed_idx| {
                let base_state = packed_idx * pack_width;
                sponge.absorb(
                    &mut packed_states[packed_idx],
                    matrix.vertically_packed_row(base_state),
                );
            });

            unpack_packed_states::<P, WIDTH>(
                &packed_states[..required_packed],
                &mut scalar_states[..height],
            );
        }

        active_height = height;
    }

    scalar_states
        .into_iter()
        .map(|state| sponge.squeee(state))
        .collect();

    // Refresh the packed representation so the SIMD states mirror `scalar_states`.
    let packed_len = final_height.div_ceil(pack_width);
    if final_height < pack_width {
        pack_small_state::<P, WIDTH>(&scalar_states[..final_height], &mut packed_states[0]);
    } else {
        pack_full_states::<P, WIDTH>(
            &scalar_states[..final_height],
            &mut packed_states[..packed_len],
        );
    }

    // Squeeze digests from each packed state and expand them back into scalar leaves.
    packed_states[..packed_len]
        .par_iter_mut()
        .map(|state| sponge.squeeze::<DIGEST_ELEMS>(state))
        .collect::<Vec<_>>()
        .into_iter()
        .flat_map(unpack_array)
        .take(final_height)
        .collect()
}

/// Pack scalar states that occupy fewer lanes than `P::WIDTH`, zero-padding the remainder.
fn pack_small_state<P, const WIDTH: usize>(
    scalar_slice: &[[P::Value; WIDTH]],
    packed_state: &mut [P; WIDTH],
) where
    P: PackedValue,
{
    let lane_count = scalar_slice.len();
    debug_assert!(
        lane_count < P::WIDTH,
        "pack_small_state expects fewer lanes than the packing width"
    );
    for col in 0..WIDTH {
        packed_state[col] = P::from_fn(|lane| {
            if lane < lane_count {
                scalar_slice[lane][col]
            } else {
                P::Value::default()
            }
        });
    }
}

/// Pack a slice of scalar states whose length is an exact multiple of the SIMD width.
fn pack_full_states<P, const WIDTH: usize>(
    scalar_slice: &[[P::Value; WIDTH]],
    packed_states: &mut [[P; WIDTH]],
) where
    P: PackedValue,
{
    let pack_width = P::WIDTH;
    debug_assert!(
        scalar_slice.len() == packed_states.len() * pack_width,
        "packed slice length mismatch: {} vs {} * {}",
        scalar_slice.len(),
        packed_states.len(),
        pack_width
    );

    packed_states
        .par_iter_mut()
        .enumerate()
        .for_each(|(packed_idx, packed_state)| {
            let base_state = packed_idx * pack_width;
            *packed_state =
                array::from_fn(|col| P::from_fn(|lane| scalar_slice[base_state + lane][col]));
        });
}

/// Copy packed sponge states back into their scalar representation.
///
/// Each packed state column is expanded lane-by-lane so the scalar buffer mirrors the SIMD
/// layout.
fn unpack_packed_states<P, const WIDTH: usize>(
    packed_states: &[[P; WIDTH]],
    scalar_slice: &mut [[P::Value; WIDTH]],
) where
    P: PackedValue,
{
    let pack_width = P::WIDTH;

    // Scatter each packed state back into contiguous scalar lanes in parallel.
    scalar_slice
        .par_chunks_mut(pack_width)
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
    use core::array;

    use p3_matrix::{dense::RowMajorMatrix, Matrix};
    use p3_symmetric::{CryptographicPermutation, Permutation};

    use super::*;

    #[derive(Clone, Copy, Default)]
    struct AdditiveSponge;

    impl StatefulSponge<[u64; 4], 4, 4> for AdditiveSponge {
        type Permutation = MockPermutation;

        fn permutation(&self) -> &Self::Permutation {
            static PERM: MockPermutation = MockPermutation;
            &PERM
        }

        fn absorb<I>(&self, state: &mut [[u64; 4]; 4], input: I)
        where
            I: IntoIterator<Item = [u64; 4]>,
        {
            for word in input {
                for lane in 0..4 {
                    for col in 0..4 {
                        state[lane][col] = state[lane][col].wrapping_add(word[col]);
                    }
                }
                self.permutation().permute_mut(state);
            }
        }

        fn squeeze<const OUT: usize>(&self, state: &mut [[u64; 4]; 4]) -> [[u64; 4]; OUT] {
            core::array::from_fn(|i| state[i % 4])
        }
    }

    #[derive(Clone, Copy, Default)]
    struct MockPermutation;

    impl Permutation<[[u64; 4]; 4]> for MockPermutation {
        fn permute_mut(&self, state: &mut [[u64; 4]; 4]) {
            for lane in state.iter_mut() {
                lane.rotate_left(1);
            }
        }
    }

    impl Permutation<[u64; 4]> for MockPermutation {
        fn permute_mut(&self, state: &mut [u64; 4]) {
            state.rotate_left(1);
        }
    }

    impl CryptographicPermutation<[[u64; 4]; 4]> for MockPermutation {}
    impl CryptographicPermutation<[u64; 4]> for MockPermutation {}

    impl StatefulSponge<u64, 4, 4> for AdditiveSponge {
        type Permutation = MockPermutation;

        fn permutation(&self) -> &Self::Permutation {
            static PERM: MockPermutation = MockPermutation;
            &PERM
        }

        fn absorb<I>(&self, state: &mut [u64; 4], input: I)
        where
            I: IntoIterator<Item = u64>,
        {
            let mut input = input.into_iter();
            'outer: loop {
                for col in 0..4 {
                    if let Some(x) = input.next() {
                        state[col] = state[col].wrapping_add(x);
                    } else {
                        if col != 0 {
                            self.permutation().permute_mut(state);
                        }
                        break 'outer;
                    }
                }
                self.permutation().permute_mut(state);
            }
        }

        fn squeeze<const OUT: usize>(&self, state: &mut [u64; 4]) -> [u64; OUT] {
            array::from_fn(|i| state[i % 4])
        }
    }

    #[test]
    fn uniform_leaves_match_concatenated_matrix() {
        type Packed = [u64; 4];
        const WIDTH: usize = 4;
        const RATE: usize = 4;
        const DIGEST: usize = 2;

        let sponge = AdditiveSponge;

        let small = RowMajorMatrix::new(vec![1, 2, 3, 4], 2); // height 2, width 2
        let large = RowMajorMatrix::new(vec![10, 20, 30, 40, 50, 60, 70, 80], 2); // height 4, width 2

        let leaves_sorted = build_uniform_leaves::<Packed, _, _, WIDTH, RATE, DIGEST>(
            vec![small.clone(), large.clone()],
            &sponge,
        );

        let leaves_unsorted =
            build_uniform_leaves::<Packed, _, _, WIDTH, RATE, DIGEST>(vec![large, small], &sponge);

        assert_eq!(leaves_sorted, leaves_unsorted);
    }

    #[derive(Clone, Copy, Default)]
    struct LaneMixSponge;

    impl StatefulSponge<[u64; 4], 4, 4> for LaneMixSponge {
        type Permutation = LaneMixPermutation;

        fn permutation(&self) -> &Self::Permutation {
            static PERM: LaneMixPermutation = LaneMixPermutation;
            &PERM
        }

        fn absorb<I>(&self, state: &mut [[u64; 4]; 4], input: I)
        where
            I: IntoIterator<Item = [u64; 4]>,
        {
            for word in input {
                let lane_sum = word.iter().copied().fold(0u64, u64::wrapping_add);
                for lane in 0..4 {
                    for col in 0..4 {
                        state[lane][col] = state[lane][col]
                            .wrapping_add(word[col])
                            .wrapping_add(lane_sum);
                    }
                }
                self.permutation().permute_mut(state);
            }
        }

        fn squeeze<const OUT: usize>(&self, state: &mut [[u64; 4]; 4]) -> [[u64; 4]; OUT] {
            array::from_fn(|i| state[(i + 1) % 4])
        }
    }

    impl StatefulSponge<u64, 4, 4> for LaneMixSponge {
        type Permutation = LaneMixPermutation;

        fn permutation(&self) -> &Self::Permutation {
            static PERM: LaneMixPermutation = LaneMixPermutation;
            &PERM
        }

        fn absorb<I>(&self, state: &mut [u64; 4], input: I)
        where
            I: IntoIterator<Item = u64>,
        {
            let mut input = input.into_iter();
            'outer: loop {
                let mut lane_sum = 0u64;
                for col in 0..4 {
                    if let Some(x) = input.next() {
                        lane_sum = lane_sum.wrapping_add(x);
                        state[col] = state[col].wrapping_add(x);
                    } else {
                        if col != 0 {
                            self.permutation().permute_mut(state);
                        }
                        break 'outer;
                    }
                }
                for col in 0..4 {
                    state[col] = state[col].wrapping_add(lane_sum);
                }
                self.permutation().permute_mut(state);
            }
        }

        fn squeeze<const OUT: usize>(&self, state: &mut [u64; 4]) -> [u64; OUT] {
            array::from_fn(|i| state[(i + 1) % 4])
        }
    }

    #[derive(Clone, Copy, Default)]
    struct LaneMixPermutation;

    impl Permutation<[[u64; 4]; 4]> for LaneMixPermutation {
        fn permute_mut(&self, state: &mut [[u64; 4]; 4]) {
            let snapshot = *state;
            for lane in 0..4 {
                for col in 0..4 {
                    let neighbor = snapshot[(lane + 1) % 4][(lane + col) % 4];
                    state[lane][col] = state[lane][col].wrapping_add(neighbor);
                }
            }
            state.rotate_left(1);
        }
    }

    impl Permutation<[u64; 4]> for LaneMixPermutation {
        fn permute_mut(&self, state: &mut [u64; 4]) {
            let snapshot = *state;
            for col in 0..4 {
                let neighbor = snapshot[(col + 1) % 4];
                state[col] = state[col].wrapping_add(neighbor);
            }
            state.rotate_left(1);
        }
    }

    impl CryptographicPermutation<[[u64; 4]; 4]> for LaneMixPermutation {}
    impl CryptographicPermutation<[u64; 4]> for LaneMixPermutation {}

    #[test]
    fn uniform_leaves_zero_padding_matches_reference() {
        type Packed = [u64; 4];
        const WIDTH: usize = 4;
        const RATE: usize = 4;
        const DIGEST: usize = 2;

        let sponge = LaneMixSponge;

        // height 2 < pack width, exercises the zero-padding path
        let matrix = RowMajorMatrix::new(vec![1, 2, 3, 4, 10, 20, 30, 40], 4);

        let leaves = build_uniform_leaves::<Packed, _, _, WIDTH, RATE, DIGEST>(
            vec![matrix.clone()],
            &sponge,
        );

        let expected =
            reference_zero_padded_leaves::<Packed, _, WIDTH, RATE, DIGEST>(&matrix, &sponge);

        assert_eq!(leaves, expected);
    }

    /// Reference implementation that hashes a short matrix using explicit zero padding.
    fn reference_zero_padded_leaves<
        P,
        H,
        const WIDTH: usize,
        const RATE: usize,
        const DIGEST_ELEMS: usize,
    >(
        matrix: &RowMajorMatrix<u64>,
        sponge: &H,
    ) -> Vec<[u64; DIGEST_ELEMS]>
    where
        P: PackedValue<Value = u64> + Default,
        H: StatefulSponge<P, WIDTH, RATE>,
    {
        let pack_width = P::WIDTH;
        let height = matrix.height();
        assert!(
            height <= pack_width,
            "reference helper only supports a single SIMD packet",
        );
        let mut scalar_states = vec![[0u64; WIDTH]; height];
        let mut working_scalar = scalar_states.clone();

        for lane in 0..height {
            let mut state = working_scalar[lane];
            let row_iter = matrix
                .row_slice(lane)
                .expect("row within height")
                .iter()
                .copied();
            sponge.absorb(&mut state, row_iter);
            working_scalar[lane] = state;
        }

        scalar_states[..height].copy_from_slice(&working_scalar[..height]);

        scalar_states
            .iter_mut()
            .map(|state| sponge.squeeze::<DIGEST_ELEMS>(state))
            .collect()
    }
}
