//! A sponge that absorbs its input from the last block to the first, under a length IV.
//!
//! Reading the input backwards lets a caller stream it from a reversed iterator, with no buffer.
//! When the input ends in a run of zeros, the state after absorbing that run depends only on the length.
//! That state can be computed once and shared by every input of that length.
//!
//! # State layout
//!
//! ```text
//!     +--------------------------+-------------------------------------+
//!     |   state[0 .. r]          |   state[r]    state[r+1 .. b]       |
//!     |   rate: input, output    |   capacity: length L, then zeros    |
//!     +--------------------------+-------------------------------------+
//! ```
//!
//! - Input only ever overwrites the rate.
//! - The digest is read from the rate, at most r cells.
//! - The capacity starts as (L, 0, ..., 0) for an input of L elements.
//! - The capacity is never written by input and never read as output.
//!
//! # Absorption
//!
//! - Pad the input with zeros up to n(L) = max(1, ceil(L / r)) blocks.
//! - Absorb block n-1 first and block 0 last.
//! - Each block overwrites the rate, then the permutation runs.
//!
//! ```text
//!     b = 4, r = 2, input = [x0, x1, x2]   (L = 3, n = 2, padded = [x0, x1, x2, 0])
//!
//!         start      [ 0   0  | 3  0 ]
//!         block 1    [ x2  0  | 3  0 ]  -> P
//!         block 0    [ x0  x1 | c  c ]  -> P  -> digest = rate cells
//! ```
//!
//! # Security
//!
//! Model: the permutation is random, and every input length comes from a set S fixed in advance.
//!
//! - Reversing the block order is a fixed bijection on inputs of a given length.
//! - The length IV gives every L in S its own root: the capacity (L, 0, ..., 0).
//! - Every path from the root of L absorbs exactly n(L) blocks.
//! - Zero padding is injective once L is fixed.
//! - Overwriting makes the state entering the permutation (block, capacity).
//! - That state does not depend on what the rate held before.
//!
//! So two distinct inputs reach the same permutation input only through one of two events.
//!
//! After N permutation calls:
//!
//! ```text
//!     capacity collision:  two permutation outputs share a capacity     <= N^2 / |F|^c
//!     root hit:            a permutation output's capacity is a root    <= N * |S| / |F|^c
//! ```
//!
//! Here c = WIDTH - RATE.
//!
//! These are the two events of the sponge-graph argument of \[BDPA08\].
//!
//! Here the graph has one root per allowed length.
//!
//! The root term counts every allowed length, not only the lengths queried.
//!
//! An adaptive adversary can aim at any root in S.
//!
//! So the bound needs |S| far below |F|^c:
//!
//! - with c = 1 and every length below p allowed, the root term is about N: no security;
//! - with c >= 2, any set of lengths below p keeps the root term below N / |F|^(c - 1).
//!
//! The digest must stay inside the rate.
//! A capacity cell in the output is no longer hidden, and the bound above stops counting it.
//! A full-state output lets anyone invert the last permutation.
//! For a one-block input that inversion returns the input itself.
//!
//! The length is encoded as a field element, so it is injective only for L < p.
//!
//! \[BDPA08\] Bertoni, Daemen, Peeters, Van Assche.
//! On the indifferentiability of the sponge construction. Eurocrypt 2008.
//! <https://keccak.team/files/SpongeIndifferentiability.pdf>

use alloc::vec::Vec;
use core::iter;

use p3_field::PrimeCharacteristicRing;

use crate::hasher::CryptographicHasher;
use crate::permutation::CryptographicPermutation;

/// An overwrite-mode sponge that absorbs right to left, under a length IV.
///
/// Inputs of any length are accepted.
/// The input is zero-padded to whole blocks, which the length IV makes injective.
///
/// # Parameters
///
/// - `WIDTH` -- total state size (rate + capacity).
/// - `RATE`  -- cells overwritten per block, at the low end of the state.
/// - `OUT`   -- cells squeezed from the rate.
#[derive(Copy, Clone, Debug)]
pub struct RtlPaddingFreeSponge<P, const WIDTH: usize, const RATE: usize, const OUT: usize> {
    /// The cryptographic permutation applied after each absorbed block.
    permutation: P,
}

/// The sponge state after absorbing the zero blocks at the end of an input.
///
/// It records how many leading blocks of the padded input are still to be absorbed.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct RtlMidstate<T, const WIDTH: usize> {
    /// Full sponge state, rate and capacity.
    state: [T; WIDTH],
    /// Number of blocks the reversed stream must still supply.
    remaining_blocks: usize,
}

impl<T, const WIDTH: usize> RtlMidstate<T, WIDTH> {
    /// Number of blocks the reversed stream must still supply.
    pub const fn remaining_blocks(&self) -> usize {
        self.remaining_blocks
    }
}

impl<P, const WIDTH: usize, const RATE: usize, const OUT: usize>
    RtlPaddingFreeSponge<P, WIDTH, RATE, OUT>
{
    /// Wrap a permutation.
    ///
    /// The shape is checked at compile time.
    /// The digest must fit in the rate, and the capacity must be non-empty.
    pub const fn new(permutation: P) -> Self {
        const {
            assert!(RATE > 0);
            assert!(RATE < WIDTH);
            assert!(OUT > 0);
            assert!(OUT <= RATE);
        }
        Self { permutation }
    }

    /// Number of blocks of the zero-padded input of `len` elements.
    ///
    /// An empty input still takes one all-zero block, so the digest always passes through the permutation.
    const fn num_blocks(len: usize) -> usize {
        if len == 0 { 1 } else { len.div_ceil(RATE) }
    }

    /// Absorb the zero blocks at the end of an input of `len` elements.
    ///
    /// The result depends only on `len` and `zero_blocks`.
    /// Every input of that length sharing those trailing zero blocks can resume from it.
    ///
    /// Blocks are counted on the zero-padded input.
    /// Padding cells count as zeros.
    ///
    /// # Panics
    ///
    /// Panics if `zero_blocks` exceeds the number of blocks of the padded input.
    pub fn precompute_zero_suffix_state<T>(
        &self,
        len: usize,
        zero_blocks: usize,
    ) -> RtlMidstate<T, WIDTH>
    where
        T: PrimeCharacteristicRing + Copy,
        P: CryptographicPermutation<[T; WIDTH]>,
    {
        let num_blocks = Self::num_blocks(len);
        assert!(
            zero_blocks <= num_blocks,
            "{zero_blocks} zero blocks exceed the {num_blocks} blocks of a length-{len} input"
        );

        // Length IV in the first capacity cell, zeros everywhere else.
        let mut midstate = RtlMidstate {
            state: [T::ZERO; WIDTH],
            remaining_blocks: num_blocks,
        };
        midstate.state[RATE] = T::from_usize(len);

        // The zero blocks run through the very loop the full hash uses.
        self.absorb(&mut midstate, iter::repeat_n(T::ZERO, zero_blocks * RATE));
        midstate
    }

    /// Finish a hash from a midstate, reading the rest of the padded input last element first.
    ///
    /// The stream is the head of the zero-padded input that the midstate still owes, reversed.
    /// It holds a whole number of blocks.
    /// From a midstate with no zero blocks, it starts with the padding zeros.
    ///
    /// # Panics
    ///
    /// Panics if the stream does not hold exactly the blocks the midstate still owes.
    pub fn hash_rev_iter<T, I>(&self, midstate: &RtlMidstate<T, WIDTH>, rev_input: I) -> [T; OUT]
    where
        T: PrimeCharacteristicRing + Copy,
        P: CryptographicPermutation<[T; WIDTH]>,
        I: IntoIterator<Item = T>,
    {
        let mut midstate = *midstate;
        let mut rev_input = rev_input.into_iter();
        self.absorb(&mut midstate, rev_input.by_ref());

        // Both checks keep every digest equal to the hash of some genuine input of this length.
        assert_eq!(
            midstate.remaining_blocks, 0,
            "reversed input ended {} blocks early",
            midstate.remaining_blocks
        );
        assert!(
            rev_input.next().is_none(),
            "reversed input is longer than the padded input"
        );

        // Squeeze from the rate, the low end of the state.
        midstate.state[..OUT]
            .try_into()
            .expect("OUT <= RATE <= WIDTH by the constructor")
    }

    /// Absorb whole blocks from a reversed stream until it runs dry or no block remains.
    ///
    /// The first element of a block lands at `state[RATE - 1]`, the last at `state[0]`.
    /// So each block sits in the rate in its natural order.
    ///
    /// # Panics
    ///
    /// Panics if the stream ends inside a block.
    fn absorb<T, I>(&self, midstate: &mut RtlMidstate<T, WIDTH>, mut rev_input: I)
    where
        T: Copy,
        P: CryptographicPermutation<[T; WIDTH]>,
        I: Iterator<Item = T>,
    {
        while midstate.remaining_blocks > 0 {
            // A dry stream on a block boundary hands control back to the caller.
            let Some(last) = rev_input.next() else {
                return;
            };
            midstate.state[RATE - 1] = last;

            // Fill the rest of the rate downward.
            for cell in midstate.state[..RATE - 1].iter_mut().rev() {
                *cell = rev_input
                    .next()
                    .expect("reversed input ended inside a block");
            }

            self.permutation.permute_mut(&mut midstate.state);
            midstate.remaining_blocks -= 1;
        }
    }
}

impl<T, P, const WIDTH: usize, const RATE: usize, const OUT: usize> CryptographicHasher<T, [T; OUT]>
    for RtlPaddingFreeSponge<P, WIDTH, RATE, OUT>
where
    T: PrimeCharacteristicRing + Copy,
    P: CryptographicPermutation<[T; WIDTH]>,
{
    fn hash_iter<I>(&self, input: I) -> [T; OUT]
    where
        I: IntoIterator<Item = T>,
    {
        // The last block is absorbed first, so a forward stream has to be buffered.
        let input: Vec<T> = input.into_iter().collect();
        self.hash_slice(&input)
    }

    fn hash_iter_slices<'a, I>(&self, input: I) -> [T; OUT]
    where
        I: IntoIterator<Item = &'a [T]>,
        T: 'a,
    {
        // Only the slice handles are buffered, never the elements.
        let slices: Vec<&[T]> = input.into_iter().collect();
        let len = slices.iter().map(|s| s.len()).sum();
        let padding = Self::num_blocks(len) * RATE - len;

        let midstate = self.precompute_zero_suffix_state(len, 0);
        let rev_input = iter::repeat_n(T::ZERO, padding).chain(
            slices
                .into_iter()
                .rev()
                .flat_map(|s| s.iter().rev().copied()),
        );
        self.hash_rev_iter(&midstate, rev_input)
    }

    fn hash_slice(&self, input: &[T]) -> [T; OUT] {
        // Padding zeros come first in reverse order, then the input from its last element.
        let len = input.len();
        let padding = Self::num_blocks(len) * RATE - len;

        let midstate = self.precompute_zero_suffix_state(len, 0);
        let rev_input = iter::repeat_n(T::ZERO, padding).chain(input.iter().rev().copied());
        self.hash_rev_iter(&midstate, rev_input)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_field::PrimeCharacteristicRing;
    use p3_koala_bear::KoalaBear;
    use proptest::prelude::*;

    use super::*;
    use crate::Permutation;

    type F = KoalaBear;

    // A small but genuine permutation of F^W, sensitive to every cell and its position.
    //
    //   round:  x_i <- x_i^3            (bijective: gcd(3, p - 1) = 1 for KoalaBear)
    //           x_i <- x_i + sum_j x_j  (matrix I + J, invertible since 1 + W != 0 mod p)
    //           x_i <- x_i + (i + 1)    (breaks the symmetry between cells)
    #[derive(Clone)]
    struct ToyPermutation;

    impl<const W: usize> Permutation<[F; W]> for ToyPermutation {
        fn permute_mut(&self, state: &mut [F; W]) {
            for _ in 0..3 {
                state.iter_mut().for_each(|x| *x = x.cube());
                let sum: F = state.iter().copied().sum();
                for (i, x) in state.iter_mut().enumerate() {
                    *x += sum + F::from_usize(i + 1);
                }
            }
        }
    }

    impl<const W: usize> CryptographicPermutation<[F; W]> for ToyPermutation {}

    // A rotation, so a hand trace can follow every cell.
    #[derive(Clone)]
    struct RotateLeft;

    impl<const W: usize> Permutation<[F; W]> for RotateLeft {
        fn permute_mut(&self, state: &mut [F; W]) {
            state.rotate_left(1);
        }
    }

    impl<const W: usize> CryptographicPermutation<[F; W]> for RotateLeft {}

    const WIDTH: usize = 6;
    const RATE: usize = 3;
    const OUT: usize = 2;

    type Sponge = RtlPaddingFreeSponge<ToyPermutation, WIDTH, RATE, OUT>;

    // The textbook forward overwrite sponge, written independently of the code under test.
    //
    //   1. zero-pad to max(1, ceil(L / r)) blocks
    //   2. reverse the block order
    //   3. start from rate = 0, capacity = (L, 0, ..., 0)
    //   4. per block: overwrite the rate, permute
    //   5. squeeze the first OUT rate cells
    fn reference(data: &[F]) -> [F; OUT] {
        let num_blocks = data.len().div_ceil(RATE).max(1);
        let mut padded = data.to_vec();
        padded.resize(num_blocks * RATE, F::ZERO);

        let mut state = [F::ZERO; WIDTH];
        state[RATE] = F::from_usize(data.len());
        for block in padded.as_chunks::<RATE>().0.iter().rev() {
            state[..RATE].copy_from_slice(block);
            ToyPermutation.permute_mut(&mut state);
        }
        state[..OUT].try_into().unwrap()
    }

    fn arb_data(max_len: usize) -> impl Strategy<Value = Vec<F>> {
        proptest::collection::vec(any::<u32>().prop_map(F::new), 0..=max_len)
    }

    #[test]
    fn hand_traced_layout() {
        // b = 4, r = 2, rotation as the permutation, input [1, 2, 3].
        //
        //   L = 3, n = 2, padded = [1, 2, 3, 0]
        //
        //   start                        [0, 0 | 3, 0]
        //   block 1 = [3, 0]             [3, 0 | 3, 0]  rotate -> [0, 3, 0, 3]
        //   block 0 = [1, 2]             [1, 2 | 0, 3]  rotate -> [2, 0, 3, 1]
        //   squeeze the 2 rate cells     [2, 0]
        let sponge = RtlPaddingFreeSponge::<RotateLeft, 4, 2, 2>::new(RotateLeft);
        let digest = sponge.hash_slice(&F::new_array([1, 2, 3]));
        assert_eq!(digest, F::new_array([2, 0]));
    }

    #[test]
    fn empty_input_passes_through_the_permutation() {
        // L = 0: one all-zero block over an all-zero IV, then one permutation.
        let mut state = [F::ZERO; WIDTH];
        ToyPermutation.permute_mut(&mut state);

        let digest = Sponge::new(ToyPermutation).hash_slice(&[]);
        assert_eq!(digest, state[..OUT]);
    }

    #[test]
    fn trailing_zeros_change_the_digest() {
        // [a] and [a, 0] pad to the same block, so only the length IV separates them.
        let sponge = Sponge::new(ToyPermutation);
        let a = F::new(7);
        assert_ne!(sponge.hash_slice(&[a]), sponge.hash_slice(&[a, F::ZERO]));
    }

    #[test]
    fn midstate_counts_padded_blocks() {
        // L = 7, r = 3: three padded blocks, two of them precomputed.
        let midstate: RtlMidstate<F, WIDTH> =
            Sponge::new(ToyPermutation).precompute_zero_suffix_state(7, 2);
        assert_eq!(midstate.remaining_blocks(), 1);
    }

    #[test]
    #[should_panic(expected = "exceed the 3 blocks")]
    fn too_many_zero_blocks_are_refused() {
        // A length-7 input has only three blocks to skip.
        let _: RtlMidstate<F, WIDTH> =
            Sponge::new(ToyPermutation).precompute_zero_suffix_state(7, 4);
    }

    #[test]
    #[should_panic(expected = "ended 1 blocks early")]
    fn short_reversed_stream_is_refused() {
        // Two blocks are owed, only one arrives.
        let sponge = Sponge::new(ToyPermutation);
        let midstate = sponge.precompute_zero_suffix_state(6, 0);
        let _ = sponge.hash_rev_iter(&midstate, [F::ONE; RATE]);
    }

    #[test]
    #[should_panic(expected = "longer than the padded input")]
    fn long_reversed_stream_is_refused() {
        // One block is owed, two arrive.
        let sponge = Sponge::new(ToyPermutation);
        let midstate = sponge.precompute_zero_suffix_state(3, 0);
        let _ = sponge.hash_rev_iter(&midstate, [F::ONE; 2 * RATE]);
    }

    #[test]
    #[should_panic(expected = "ended inside a block")]
    fn partial_block_is_refused() {
        // The stream stops one element into the only block.
        let sponge = Sponge::new(ToyPermutation);
        let midstate = sponge.precompute_zero_suffix_state(3, 0);
        let _ = sponge.hash_rev_iter(&midstate, [F::ONE]);
    }

    proptest! {
        #[test]
        fn hash_slice_matches_the_reference(data in arb_data(4 * RATE + 2)) {
            // Covers the empty input, partial last blocks, and whole blocks.
            let sponge = Sponge::new(ToyPermutation);
            prop_assert_eq!(sponge.hash_slice(&data), reference(&data));
        }

        #[test]
        fn every_entry_point_agrees(data in arb_data(4 * RATE + 2), cut in any::<prop::sample::Index>()) {
            // The forward iterator, a split into two slices, and the slice itself.
            let sponge = Sponge::new(ToyPermutation);
            let expected = sponge.hash_slice(&data);

            prop_assert_eq!(sponge.hash_iter(data.iter().copied()), expected);

            let (left, right) = data.split_at(cut.index(data.len() + 1));
            prop_assert_eq!(sponge.hash_iter_slices([left, right]), expected);
        }

        #[test]
        fn zero_suffix_precomputation_matches_the_reference(
            prefix in arb_data(3 * RATE),
            zeros in 0..=4 * RATE,
            zero_blocks_seed in any::<prop::sample::Index>(),
        ) {
            // Input = prefix ‖ zeros, of length L.
            let mut data = prefix;
            data.extend(iter::repeat_n(F::ZERO, zeros));
            let len = data.len();

            // Padded blocks lying entirely in the zero tail, padding included.
            let num_blocks = len.div_ceil(RATE).max(1);
            let nonzero_blocks = (len - zeros).div_ceil(RATE);
            let zero_blocks = zero_blocks_seed.index(num_blocks - nonzero_blocks + 1);

            // Resume with the leading blocks of the padded input, last element first.
            let sponge = Sponge::new(ToyPermutation);
            let midstate = sponge.precompute_zero_suffix_state(len, zero_blocks);
            let mut padded = data.clone();
            padded.resize(num_blocks * RATE, F::ZERO);
            let head = &padded[..midstate.remaining_blocks() * RATE];

            let digest = sponge.hash_rev_iter(&midstate, head.iter().rev().copied());
            prop_assert_eq!(digest, reference(&data));
        }
    }

    #[test]
    fn precomputation_is_shared_across_inputs() {
        // Two inputs of length 9 ending in two zero blocks share one midstate.
        let sponge = Sponge::new(ToyPermutation);
        let midstate = sponge.precompute_zero_suffix_state(9, 2);

        for head in [F::new_array([1, 2, 3]), F::new_array([4, 5, 6])] {
            let mut data = head.to_vec();
            data.extend(vec![F::ZERO; 2 * RATE]);
            let digest = sponge.hash_rev_iter(&midstate, head.iter().rev().copied());
            assert_eq!(digest, sponge.hash_slice(&data));
        }
    }
}
