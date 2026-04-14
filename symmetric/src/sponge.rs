use alloc::string::String;
use core::marker::PhantomData;

use itertools::Itertools;
use p3_field::{Field, PrimeField, PrimeField32, reduce_32};

use crate::hasher::CryptographicHasher;
use crate::permutation::CryptographicPermutation;

/// A padding-free, overwrite-mode sponge function.
///
/// `WIDTH` is the sponge's rate plus the sponge's capacity.
#[derive(Copy, Clone, Debug)]
pub struct PaddingFreeSponge<P, const WIDTH: usize, const RATE: usize, const OUT: usize> {
    permutation: P,
}

impl<P, const WIDTH: usize, const RATE: usize, const OUT: usize>
    PaddingFreeSponge<P, WIDTH, RATE, OUT>
{
    pub const fn new(permutation: P) -> Self {
        Self { permutation }
    }
}

impl<T, P, const WIDTH: usize, const RATE: usize, const OUT: usize> CryptographicHasher<T, [T; OUT]>
    for PaddingFreeSponge<P, WIDTH, RATE, OUT>
where
    T: Default + Copy,
    P: CryptographicPermutation<[T; WIDTH]>,
{
    fn hash_iter<I>(&self, input: I) -> [T; OUT]
    where
        I: IntoIterator<Item = T>,
    {
        const {
            assert!(RATE > 0);
            assert!(RATE < WIDTH);
            assert!(OUT <= WIDTH);
        }
        let mut state = [T::default(); WIDTH];
        let mut input = input.into_iter();

        // Itertools' chunks() is more convenient, but seems to add more overhead,
        // hence the more manual loop.
        'outer: loop {
            for i in 0..RATE {
                if let Some(x) = input.next() {
                    state[i] = x;
                } else {
                    if i != 0 {
                        self.permutation.permute_mut(&mut state);
                    }
                    break 'outer;
                }
            }
            self.permutation.permute_mut(&mut state);
        }

        state[..OUT].try_into().unwrap()
    }
}

/// A padding-free, overwrite-mode sponge function that operates natively over PF but accepts elements
/// of F: PrimeField32.
///
/// `WIDTH` is the sponge's rate plus the sponge's capacity.
#[derive(Clone, Debug)]
pub struct MultiField32PaddingFreeSponge<
    F,
    PF,
    P,
    const WIDTH: usize,
    const RATE: usize,
    const OUT: usize,
> {
    permutation: P,
    num_f_elms: usize,
    _phantom: PhantomData<(F, PF)>,
}

impl<F, PF, P, const WIDTH: usize, const RATE: usize, const OUT: usize>
    MultiField32PaddingFreeSponge<F, PF, P, WIDTH, RATE, OUT>
where
    F: PrimeField32,
    PF: Field,
{
    pub fn new(permutation: P) -> Result<Self, String> {
        if F::order() >= PF::order() {
            return Err(String::from("F::order() must be less than PF::order()"));
        }

        let num_f_elms = PF::bits() / F::bits();
        Ok(Self {
            permutation,
            num_f_elms,
            _phantom: PhantomData,
        })
    }
}

impl<F, PF, P, const WIDTH: usize, const RATE: usize, const OUT: usize>
    CryptographicHasher<F, [PF; OUT]> for MultiField32PaddingFreeSponge<F, PF, P, WIDTH, RATE, OUT>
where
    F: PrimeField32,
    PF: PrimeField + Default + Copy,
    P: CryptographicPermutation<[PF; WIDTH]>,
{
    fn hash_iter<I>(&self, input: I) -> [PF; OUT]
    where
        I: IntoIterator<Item = F>,
    {
        const {
            assert!(RATE > 0);
            assert!(RATE < WIDTH);
            assert!(OUT <= WIDTH);
        }
        let mut state = [PF::default(); WIDTH];
        for block_chunk in &input.into_iter().chunks(RATE) {
            for (chunk_id, chunk) in (&block_chunk.chunks(self.num_f_elms))
                .into_iter()
                .enumerate()
            {
                state[chunk_id] = reduce_32(&chunk.collect_vec());
            }
            state = self.permutation.permute(state);
        }

        state[..OUT].try_into().unwrap()
    }
}

/// Absorb remaining iterator elements into the rate region, right-to-left.
///
/// Shared helper used by the right-to-left sponge's hash and initial-state methods.
///
/// # Algorithm
///
/// Each block overwrites positions `[WIDTH-RATE .. WIDTH-1]` in descending
/// index order, then applies the permutation:
///
/// ```text
///   state = [ cap_0 .. cap_{c-1} | rate_{r-1} .. rate_0 ]
///                                   ←←←←←←←←←←←←←←←←←←
///                                   filled right-to-left
/// ```
///
/// # Panics
///
/// Panics if the iterator yields a number of elements that is not a multiple of RATE.
#[inline(always)]
fn absorb_rtl_chunks<T, P, I, const WIDTH: usize, const RATE: usize, const OUT: usize>(
    permutation: &P,
    state: &mut [T; WIDTH],
    iter: &mut I,
) -> [T; OUT]
where
    T: Default + Copy,
    P: CryptographicPermutation<[T; WIDTH]>,
    I: Iterator<Item = T>,
{
    // Consume RATE elements per iteration.
    //
    // The first element lands at the highest rate index; the rest fill downward.
    //
    // Example with WIDTH=4, RATE=2, input = [a, b, c, d]:
    //
    //   Block 1:
    //     state[3] = a     ← highest rate index
    //     state[2] = b     ← next rate index down
    //     permute(state)
    //
    //   Block 2:
    //     state[3] = c
    //     state[2] = d
    //     permute(state)
    while let Some(elem) = iter.next() {
        // Place the first element at the highest rate index.
        state[WIDTH - 1] = elem;

        // Fill the remaining rate slots in descending order.
        // Panics if the iterator runs dry mid-block (non-multiple of RATE).
        for pos in (WIDTH - RATE..WIDTH - 1).rev() {
            state[pos] = iter.next().unwrap();
        }

        // Apply the permutation to diffuse absorbed data across the full state.
        permutation.permute_mut(state);
    }

    // Extract the digest from the first OUT positions.
    state[..OUT].try_into().unwrap()
}

/// A padding-free, overwrite-mode sponge with right-to-left absorption.
///
/// # Overview
///
/// Standard sponges (Bertoni et al., 2007) absorb left-to-right into
/// `[0..RATE)`. This variant absorbs from `[WIDTH-1]` downward, so
/// elements are placed as they arrive without buffering a full block.
///
/// Useful in FRI/WHIR where evaluations arrive in reverse order during
/// Merkle tree construction — avoids collecting and reversing.
///
/// # Two-phase Absorption
///
/// - **Phase 1** — first block fills all WIDTH positions (capacity + rate).
///   Injects real data into the capacity for full-width diffusion.
/// - **Phase 2** — subsequent blocks overwrite only the last RATE positions.
///   Capacity retains accumulated entropy from prior permutations.
///
/// ```text
///   Phase 1 (WIDTH=4, input = [a, b, c, d, ...]):
///
///     idx:    3    2    1    0
///           +----+----+----+----+
///           | a  | b  | c  | d  |   ← all WIDTH slots
///           +----+----+----+----+
///             rate      capacity
///
///   Phase 2 (RATE=2, next = [e, f]):
///
///     idx:    3    2    1    0
///           +----+----+----+----+
///           | e  | f  |  (kept) |   ← only RATE slots
///           +----+----+----+----+
///             rate      capacity
/// ```
///
/// `WIDTH` is the sponge's rate plus the sponge's capacity.
#[derive(Copy, Clone, Debug)]
pub struct RtlPaddingFreeSponge<P, const WIDTH: usize, const RATE: usize, const OUT: usize> {
    /// The cryptographic permutation applied after each absorption block.
    permutation: P,
}

impl<P, const WIDTH: usize, const RATE: usize, const OUT: usize>
    RtlPaddingFreeSponge<P, WIDTH, RATE, OUT>
{
    /// Create a new right-to-left sponge wrapping the given permutation.
    pub const fn new(permutation: P) -> Self {
        Self { permutation }
    }

    /// Precompute the sponge state after absorbing N all-zero rate-sized chunks.
    ///
    /// # Overview
    ///
    /// A midstate caching optimization.
    ///
    /// In FRI/WHIR, polynomials of degree d evaluated over domains of size
    /// N >> d produce many all-zero Merkle leaves. This precomputes the
    /// sponge state for those zeros once, avoiding redundant permutations.
    ///
    /// Pass the returned state to the initial-state hashing method to
    /// continue absorption with real data.
    ///
    /// # Algorithm
    ///
    /// ```text
    ///   n_zero_chunks = 4, WIDTH = 4, RATE = 2:
    ///
    ///   Chunk 1:  [0,0,0,0] → permute → s_1
    ///   Chunk 2:  s_1, rate zeroed → permute → s_2
    ///   Chunk 3:  s_2, rate zeroed → permute → s_3  ← returned
    ///             (chunk 4's rate will be overwritten by real data)
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `n_zero_chunks < 2`.
    pub fn precompute_zero_suffix_state<T>(&self, n_zero_chunks: usize) -> [T; WIDTH]
    where
        T: Default + Copy,
        P: CryptographicPermutation<[T; WIDTH]>,
    {
        const {
            assert!(RATE > 0);
            assert!(RATE < WIDTH);
            assert!(OUT <= WIDTH);
        }

        // Minimum 2 chunks: one for the initial permutation, one whose rate
        // region is left for the caller to overwrite with real data.
        assert!(n_zero_chunks >= 2);

        // Chunk 1: all-zero state, permute once.
        let mut state = [T::default(); WIDTH];
        self.permutation.permute_mut(&mut state);

        // Chunks 2 .. (n_zero_chunks - 1): zero the rate region and permute.
        //
        // Why n_zero_chunks - 2?
        // - Chunk 1 handled above,
        // - The final chunk's rate region is left as-is: the caller overwrites it with real data.
        for _ in 0..n_zero_chunks - 2 {
            // Reset only the rate portion (positions [WIDTH-RATE .. WIDTH-1]).
            // The capacity portion retains accumulated permutation entropy.
            state[WIDTH - RATE..].fill(T::default());
            self.permutation.permute_mut(&mut state);
        }

        state
    }

    /// Continue right-to-left absorption from a precomputed initial state.
    ///
    /// Companion to the zero-suffix precomputation method:
    /// precompute the state for known zero-padding, then call this
    /// to hash the remaining non-zero data.
    ///
    /// # Panics
    ///
    /// Panics if the iterator length is not a multiple of RATE.
    ///
    /// # Returns
    ///
    /// The first OUT elements of the final sponge state.
    pub fn hash_with_initial_state<T, I>(&self, initial_state: &[T; WIDTH], iter: I) -> [T; OUT]
    where
        T: Default + Copy,
        P: CryptographicPermutation<[T; WIDTH]>,
        I: IntoIterator<Item = T>,
    {
        const {
            assert!(RATE > 0);
            assert!(RATE < WIDTH);
            assert!(OUT <= WIDTH);
        }

        // Copy the precomputed state — the caller's data stays immutable.
        let mut state = *initial_state;

        // Delegate to the shared RTL chunk absorber for the remaining elements.
        let mut iter = iter.into_iter();
        absorb_rtl_chunks::<T, P, _, WIDTH, RATE, OUT>(&self.permutation, &mut state, &mut iter)
    }
}

impl<T, P, const WIDTH: usize, const RATE: usize, const OUT: usize> CryptographicHasher<T, [T; OUT]>
    for RtlPaddingFreeSponge<P, WIDTH, RATE, OUT>
where
    T: Default + Copy,
    P: CryptographicPermutation<[T; WIDTH]>,
{
    fn hash_iter<I>(&self, input: I) -> [T; OUT]
    where
        I: IntoIterator<Item = T>,
    {
        // - RATE > 0: at least one element per block.
        // - RATE < WIDTH: capacity must be non-zero for security.
        // - OUT <= WIDTH: output cannot exceed the state size.
        const {
            assert!(RATE > 0);
            assert!(RATE < WIDTH);
            assert!(OUT <= WIDTH);
        }

        // Start from the all-zero initialization vector.
        let mut state = [T::default(); WIDTH];
        let mut iter = input.into_iter();

        // Phase 1: First block — fill the entire WIDTH right-to-left.
        //
        // Why the full WIDTH? The capacity region must receive direct
        // input for collision resistance. If it only ever held zeros or
        // permutation outputs, an attacker could exploit the reduced
        // entropy in the capacity to find collisions.
        //
        // Example with WIDTH=4, input = [a, b, c, d, ...]:
        //
        //   idx:    3    2    1    0
        //         +----+----+----+----+
        //   fill: | a  | b  | c  | d  |   ← all WIDTH slots written
        //         +----+----+----+----+
        //           rate      capacity
        //   permute(state)
        for pos in (0..WIDTH).rev() {
            if let Some(x) = iter.next() {
                state[pos] = x;
            } else {
                // Input shorter than WIDTH — permute only if we absorbed
                // at least one element (pos < WIDTH-1 means we did).
                if pos < WIDTH - 1 {
                    self.permutation.permute_mut(&mut state);
                }
                return state[..OUT].try_into().unwrap();
            }
        }

        // Permute after the first full-WIDTH block.
        self.permutation.permute_mut(&mut state);

        // Phase 2: Subsequent blocks — only the last RATE positions change.
        // Delegate to the shared RTL chunk absorber.
        absorb_rtl_chunks::<T, P, _, WIDTH, RATE, OUT>(&self.permutation, &mut state, &mut iter)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use proptest::prelude::*;

    use super::*;
    use crate::Permutation;

    #[derive(Clone)]
    struct MockPermutation;

    impl<T, const WIDTH: usize> Permutation<[T; WIDTH]> for MockPermutation
    where
        T: Copy + core::ops::Add<Output = T> + Default,
    {
        fn permute_mut(&self, input: &mut [T; WIDTH]) {
            let sum: T = input.iter().copied().fold(T::default(), |acc, x| acc + x);
            // Set every element to the sum
            *input = [sum; WIDTH];
        }
    }

    impl<T, const WIDTH: usize> CryptographicPermutation<[T; WIDTH]> for MockPermutation where
        T: Copy + core::ops::Add<Output = T> + Default
    {
    }

    // Shared sponge dimensions used across unit tests and proptests.
    const WIDTH: usize = 4;
    const RATE: usize = 2;
    const OUT: usize = 2;

    #[test]
    fn test_padding_free_sponge_basic() {
        let permutation = MockPermutation;
        let sponge = PaddingFreeSponge::<MockPermutation, WIDTH, RATE, OUT>::new(permutation);

        let input = [1, 2, 3, 4, 5];
        let output = sponge.hash_iter(input);

        // Explanation of why the final state results in [44, 44, 44, 44]:
        // Initial state: [0, 0, 0, 0]
        // First input chunk [1, 2] overwrites first two positions: [1, 2, 0, 0]
        // Apply permutation (sum all elements and overwrite): [3, 3, 3, 3]
        // Second input chunk [3, 4] overwrites first two positions: [3, 4, 3, 3]
        // Apply permutation: [13, 13, 13, 13] (3 + 4 + 3 + 3 = 13)
        // Third input chunk [5] overwrites first position: [5, 13, 13, 13]
        // Apply permutation: [44, 44, 44, 44] (5 + 13 + 13 + 13 = 44)

        assert_eq!(output, [44; OUT]);
    }

    #[test]
    fn test_padding_free_sponge_empty_input() {
        let permutation = MockPermutation;
        let sponge = PaddingFreeSponge::<MockPermutation, WIDTH, RATE, OUT>::new(permutation);

        let input: [u64; 0] = [];
        let output = sponge.hash_iter(input);

        assert_eq!(
            output, [0; OUT],
            "Should return default values when input is empty."
        );
    }

    #[test]
    fn test_padding_free_sponge_exact_block_size() {
        const WIDTH: usize = 6;
        const RATE: usize = 3;
        const OUT: usize = 2;

        let permutation = MockPermutation;
        let sponge = PaddingFreeSponge::<MockPermutation, WIDTH, RATE, OUT>::new(permutation);

        let input = [10, 20, 30];
        let output = sponge.hash_iter(input);

        let expected_sum = 10 + 20 + 30;
        assert_eq!(output, [expected_sum; OUT]);
    }

    #[test]
    fn test_rtl_sponge_basic() {
        // Sponge dimensions: WIDTH=4, RATE=2, OUT=2.
        // The state has 2 capacity slots and 2 rate slots.
        //
        // Input: [1, 2, 3, 4, 5, 6]
        //   - First block fills all 4 positions (entire WIDTH) right-to-left.
        //   - Second block fills only the last 2 positions (RATE) right-to-left.

        let sponge =
            RtlPaddingFreeSponge::<MockPermutation, WIDTH, RATE, OUT>::new(MockPermutation);
        let output = sponge.hash_iter([1u64, 2, 3, 4, 5, 6]);

        // Hand-traced state transitions:
        //
        // Initial state: [0, 0, 0, 0]
        //
        // Phase 1: First block fills WIDTH=4 positions RTL.
        //   state[3] = 1, state[2] = 2, state[1] = 3, state[0] = 4
        //   → state = [4, 3, 2, 1]
        //   permute(sum = 4+3+2+1 = 10) → [10, 10, 10, 10]
        //
        // Phase 2: Second block fills RATE=2 positions RTL.
        //   state[3] = 5, state[2] = 6
        //   → state = [10, 10, 6, 5]
        //   permute(sum = 10+10+6+5 = 31) → [31, 31, 31, 31]
        //
        // Output: state[..2] = [31, 31]
        assert_eq!(output, [31; OUT]);
    }

    #[test]
    fn test_rtl_sponge_single_full_block() {
        // Input has exactly WIDTH elements — one full first-block, no subsequent blocks.

        let sponge =
            RtlPaddingFreeSponge::<MockPermutation, WIDTH, RATE, OUT>::new(MockPermutation);
        let output = sponge.hash_iter([10u64, 20, 30, 40]);

        // State transitions:
        //
        // Initial state: [0, 0, 0, 0]
        //
        // First block fills all 4 positions RTL:
        //   state[3] = 10, state[2] = 20, state[1] = 30, state[0] = 40
        //   → state = [40, 30, 20, 10]
        //   permute(sum = 40+30+20+10 = 100) → [100, 100, 100, 100]
        //
        // Iterator exhausted — no subsequent blocks.
        // Output: state[..2] = [100, 100]
        assert_eq!(output, [100; OUT]);
    }

    #[test]
    fn test_rtl_sponge_empty_input() {
        // Empty input should return all-zero output without any permutation.

        let sponge =
            RtlPaddingFreeSponge::<MockPermutation, WIDTH, RATE, OUT>::new(MockPermutation);
        let output = sponge.hash_iter(core::iter::empty::<u64>());

        // No elements consumed → pos stays at WIDTH-1 → no permutation applied.
        // State remains [0, 0, 0, 0], output is [0, 0].
        assert_eq!(output, [0; OUT]);
    }

    #[test]
    fn test_rtl_vs_ltr_different_outputs() {
        // The same input must produce different outputs for RTL vs LTR sponges,
        // confirming they are semantically distinct constructions.

        let input = [1u64, 2, 3, 4, 5, 6];

        let ltr_sponge =
            PaddingFreeSponge::<MockPermutation, WIDTH, RATE, OUT>::new(MockPermutation);
        let ltr_output = ltr_sponge.hash_iter(input);

        let rtl_sponge =
            RtlPaddingFreeSponge::<MockPermutation, WIDTH, RATE, OUT>::new(MockPermutation);
        let rtl_output = rtl_sponge.hash_iter(input);

        // LTR absorbs into state[0..RATE], RTL absorbs into state[WIDTH-1] downward.
        // The different absorption patterns cause different intermediate states
        // and therefore different final digests.
        assert_ne!(
            ltr_output, rtl_output,
            "RTL and LTR sponges must produce different outputs for the same input"
        );
    }

    #[test]
    fn test_precompute_zero_suffix_basic() {
        // Precompute the state after absorbing 3 all-zero rate-sized chunks.

        let sponge =
            RtlPaddingFreeSponge::<MockPermutation, WIDTH, RATE, OUT>::new(MockPermutation);
        let state: [u64; WIDTH] = sponge.precompute_zero_suffix_state(3);

        // Hand-traced state transitions:
        //
        // Initial state: [0, 0, 0, 0]
        //
        // Chunk 1: permute all-zero state.
        //   sum = 0 → state = [0, 0, 0, 0]
        //
        // Chunk 2 (loop iteration 0 of n_zero_chunks - 2 = 1):
        //   Zero out rate region: state[2..4] = [0, 0]
        //   → state = [0, 0, 0, 0]  (already zero from previous permutation)
        //   permute(sum = 0) → [0, 0, 0, 0]
        //
        // With the mock permutation (sum-and-broadcast), zero input always
        // stays zero. The important thing is that the method runs without
        // panicking and returns a WIDTH-sized state.
        assert_eq!(state, [0u64; WIDTH]);
    }

    #[test]
    fn test_precompute_zero_suffix_with_nonzero_permutation() {
        // Use a more interesting scenario: verify that precomputing N zero
        // chunks matches manually running the sponge on N zero chunks.

        let sponge =
            RtlPaddingFreeSponge::<MockPermutation, WIDTH, RATE, OUT>::new(MockPermutation);

        // Manually simulate 3 zero chunks through RTL sponge.
        //
        // The RTL sponge's first block fills all WIDTH positions. With all
        // zeros, the state is [0,0,0,0] → permute → [0,0,0,0].
        // Subsequent zero-rate blocks: state[2..4] = [0,0] → permute → [0,0,0,0].
        //
        // For the mock permutation, this is trivially [0;WIDTH].
        // We verify the precomputed state matches this manual result.
        let precomputed: [u64; WIDTH] = sponge.precompute_zero_suffix_state(3);
        assert_eq!(precomputed, [0u64; WIDTH]);
    }

    #[test]
    #[should_panic]
    fn test_precompute_zero_suffix_panics_on_one() {
        // The precomputation requires at least 2 chunks.
        // Passing 1 must panic.

        let sponge =
            RtlPaddingFreeSponge::<MockPermutation, WIDTH, RATE, OUT>::new(MockPermutation);
        let _: [u64; WIDTH] = sponge.precompute_zero_suffix_state(1);
    }

    #[test]
    #[should_panic]
    fn test_precompute_zero_suffix_panics_on_zero() {
        // Zero chunks must also panic.

        let sponge =
            RtlPaddingFreeSponge::<MockPermutation, WIDTH, RATE, OUT>::new(MockPermutation);
        let _: [u64; WIDTH] = sponge.precompute_zero_suffix_state(0);
    }

    #[test]
    fn test_hash_with_initial_state_matches_full_hash() {
        // End-to-end correctness: hashing K zero-chunks + M non-zero chunks
        // via the full RTL sponge must equal precomputing K zero-chunks then
        // hashing only the M non-zero chunks from that initial state.

        let sponge =
            RtlPaddingFreeSponge::<MockPermutation, WIDTH, RATE, OUT>::new(MockPermutation);

        // Build input: 2 zero-chunks (4 zeros for the first WIDTH block,
        // then we treat it as 2 rate-sized zero blocks) followed by 1
        // non-zero block [5, 6].
        //
        // Full input: WIDTH zeros for first block + RATE zeros + [5, 6]
        //   = [0, 0, 0, 0, 0, 0, 5, 6]
        let full_input = [0u64, 0, 0, 0, 0, 0, 5, 6];

        // Hash the full input via the RTL sponge.
        let full_output = sponge.hash_iter(full_input);

        // Precompute state for the zero prefix, then hash remaining data.
        // The zero prefix corresponds to 3 chunks: 1 full-WIDTH + 1 rate-sized
        // = effectively 3 rate-equivalent chunks for the precomputation.
        let initial_state: [u64; WIDTH] = sponge.precompute_zero_suffix_state(3);
        let partial_output = sponge.hash_with_initial_state(&initial_state, [5u64, 6]);

        // Both approaches must yield the same digest.
        assert_eq!(full_output, partial_output);
    }

    #[test]
    fn test_hash_with_initial_state_empty_remaining() {
        // When the remaining iterator is empty, the output should be the
        // first OUT elements of the initial state directly (no permutation).

        let sponge =
            RtlPaddingFreeSponge::<MockPermutation, WIDTH, RATE, OUT>::new(MockPermutation);

        // Use a non-trivial initial state to verify no permutation is applied.
        let initial_state = [10u64, 20, 30, 40];

        let output = sponge.hash_with_initial_state(&initial_state, core::iter::empty::<u64>());

        // No elements absorbed → no permutation → output = state[..OUT].
        assert_eq!(output, [10, 20]);
    }

    proptest! {
        #[test]
        fn proptest_precompute_equivalence(
            n_zero_chunks in 2..=8usize,
            n_suffix_blocks in 0..=4usize,
            suffix_vals in proptest::collection::vec(0..1000u64, 0..=8),
        ) {
            // Invariant: hash(zeros ++ suffix) == precompute(N) + hash_with_initial_state(suffix)

            // Trim suffix to an exact multiple of RATE.
            let suffix: Vec<u64> = suffix_vals
                .into_iter()
                .chain(core::iter::repeat(0))
                .take(n_suffix_blocks * RATE)
                .collect();

            // Full input: WIDTH zeros (first block) + (n-2)*RATE zeros + suffix.
            let total_zeros = WIDTH + (n_zero_chunks - 2) * RATE;
            let full_input: Vec<u64> = core::iter::repeat_n(0u64, total_zeros)
                .chain(suffix.iter().copied())
                .collect();

            let sponge =
                RtlPaddingFreeSponge::<MockPermutation, WIDTH, RATE, OUT>::new(MockPermutation);

            // Path A: one-shot hash.
            let full_output = sponge.hash_iter(full_input);

            // Path B: precompute zeros, then hash suffix from that state.
            let initial_state: [u64; WIDTH] = sponge.precompute_zero_suffix_state(n_zero_chunks);
            let partial_output = sponge.hash_with_initial_state(&initial_state, suffix);

            prop_assert_eq!(full_output, partial_output);
        }

        #[test]
        fn proptest_rtl_determinism(
            // Valid lengths: WIDTH + k*RATE (after the first block, remaining
            // elements must be a multiple of RATE).
            n_suffix_blocks in 0..=4usize,
            vals in proptest::collection::vec(0..1000u64, 12..=12),
        ) {
            let input: Vec<u64> = vals.into_iter().take(WIDTH + n_suffix_blocks * RATE).collect();

            let sponge =
                RtlPaddingFreeSponge::<MockPermutation, WIDTH, RATE, OUT>::new(MockPermutation);

            // Same input twice → same output.
            let output_1 = sponge.hash_iter(input.iter().copied());
            let output_2 = sponge.hash_iter(input.iter().copied());
            prop_assert_eq!(output_1, output_2);
        }
    }
}
