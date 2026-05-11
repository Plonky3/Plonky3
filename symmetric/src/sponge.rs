//! Sponge-based hash functions built from cryptographic permutations.
//!
//! # Background
//!
//! A sponge [BDPV07] hashes an input using a fixed-width permutation P.
//! The b-element state has two regions:
//!
//! ```text
//!     +--------------------------------------------+
//!     |   state[0 .. r]   |   state[r .. b]        |
//!     |   rate  (outer)   |   capacity  (inner)    |
//!     +--------------------------------------------+
//! ```
//!
//! - **Rate (r)** -- absorbs input, produces output.
//! - **Capacity (c = b - r)** -- never exposed directly.
//!   Provides collision resistance up to |F|^{c/2} queries [BDPA08].
//!
//! This module uses the **overwrite** variant: each input block
//! overwrites (rather than XORs into) the rate portion.
//! Security carries over from the standard sponge [BDPA08, AMP10].
//!
//! # Variants
//!
//! This module provides two sponge variants for different use cases:
//!
//! - `PaddingFreeSponge` -- for **fixed-length** inputs where the
//!   number of elements to hash is predetermined by the protocol and
//!   not controlled by the attacker. Collision-resistant in this
//!   setting. Not suitable when the attacker controls input length.
//!
//! - `Pad10Sponge` -- for **variable-length** inputs where the number
//!   of elements can be chosen at runtime. Also secure for fixed-
//!   length inputs but slightly slower than the padding-free variant.
//!
//! # Why Padding Matters
//!
//! Without padding, different-length messages can collide trivially.
//!
//! ```text
//!     WIDTH = 8, RATE = 4, capacity = 4
//!
//!     Message A (length 10):
//!
//!            block 1         block 2        partial
//!       +--------------+ +--------------+ +--------+
//!       | h0 h1 h2 h3  | | h4 h5 h6 h7  | | h8 h9  |
//!       +--------------+ +--------------+ +--------+
//!
//!     Step 1 – absorb block 1:
//!       state = [h0, h1, h2, h3 | 0, 0, 0, 0]   -> P
//!       state = [p0, p1, p2, p3 | p4, p5, p6, p7]
//!
//!     Step 2 – absorb block 2:
//!       state = [h4, h5, h6, h7 | p4, p5, p6, p7]   -> P
//!       state = [q0, q1, q2, q3 | q4, q5, q6, q7]
//!
//!     Step 3 – absorb partial (only 2 elements):
//!       overwrite positions 0..2, leave 2..4 untouched:
//!       state = [h8, h9, q2, q3 | q4, q5, q6, q7]   -> P -> digest
//!                        ^^  ^^
//!                        still hold old values from q
//! ```
//!
//! An attacker who knows q2 can forge a collision:
//!
//! ```text
//!     Message B (length 11):
//!
//!            block 1          block 2        partial
//!       +--------------+ +--------------+ +-----------+
//!       | h0 h1 h2 h3  | | h4 h5 h6 h7  | | h8 h9 q2  |
//!       +--------------+ +--------------+ +-----------+
//!
//!     Steps 1-2 are identical. Step 3 now has 3 elements:
//!       state = [h8, h9, q2, q3 | q4, q5, q6, q7]   -> P -> digest
//!                ^^^^^^^^^^^^^^^^
//!                same state as Message A => same digest!
//! ```
//!
//! In XOR-mode sponges this would be called 0-padding. In overwrite
//! mode the leftover positions aren't zeros but old permutation
//! output -- the effect is the same: no injective encoding of length.
//!
//! Note: this is only exploitable when the attacker controls the input
//! length. When the length is fixed by the protocol (e.g. Merkle tree
//! leaves), no collision is possible.
//!
//! The fix is 10-padding -- see `Pad10Sponge` for the full scheme.

use alloc::string::String;
use core::marker::PhantomData;
use core::ops::Add;

use itertools::Itertools;
use p3_field::{Field, PrimeField, PrimeField32, reduce_32};

use crate::Permutation;
use crate::hasher::CryptographicHasher;
use crate::permutation::{CryptographicPermutation, Derangement};

/// A derangement d(x) = x + increment.
///
/// This is the standard padding function for sponge constructions.
/// A derangement has no fixed points (d(x) != x for all x), which
/// holds as long as the stored increment is non-zero.
///
/// ```ignore
/// Increment(BabyBear::ONE)   // d(x) = x + 1  for field elements
/// Increment(1u64)            // d(x) = x + 1  for raw integers
/// ```
#[derive(Copy, Clone, Debug)]
pub struct Increment<T>(pub T);

impl<T: Clone + Sync + Send + Add<Output = T>> Permutation<T> for Increment<T> {
    fn permute(&self, input: T) -> T {
        input + self.0.clone()
    }
}

impl<T: Clone + Sync + Send + Add<Output = T>> Derangement<T> for Increment<T> {}

/// A padding-free, overwrite-mode sponge.
///
/// # Security
///
/// Safe **only** for fixed-length inputs (e.g. Merkle leaves, trace
/// rows). For variable-length inputs, use `Pad10Sponge`.
///
/// **Not** collision-resistant for variable-length inputs.
/// Different-length messages can hash identically:
///
/// ```text
///     RATE = 2
///     [a]    -> [a, 0 | cap...] -> P -> digest
///     [a, 0] -> [a, 0 | cap...] -> P -> digest   <- same!
/// ```
///
/// # Parameters
///
/// - `WIDTH` -- total state size (rate + capacity).
/// - `RATE`  -- positions overwritten per block.
/// - `OUT`   -- elements squeezed from the final state.
#[derive(Copy, Clone, Debug)]
pub struct PaddingFreeSponge<P, const WIDTH: usize, const RATE: usize, const OUT: usize> {
    /// The cryptographic permutation applied after each absorbed block.
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
        // Start from the all-zero state.
        let mut state = [T::default(); WIDTH];
        let mut input = input.into_iter();

        'outer: loop {
            // Absorb one block: overwrite state[0..RATE] with input elements one at a time.
            for i in 0..RATE {
                if let Some(x) = input.next() {
                    // Overwrite the i-th rate position.
                    state[i] = x;
                } else {
                    // Input exhausted mid-block. Permute only if at least
                    // one element was absorbed in this block (i > 0).
                    // If i == 0 the state already reflects the previous
                    // permutation output and needs no extra call.
                    if i != 0 {
                        self.permutation.permute_mut(&mut state);
                    }
                    break 'outer;
                }
            }

            // Full block absorbed. Permute before the next block.
            self.permutation.permute_mut(&mut state);
        }

        // Squeeze: return the first OUT elements of the final state.
        state[..OUT].try_into().unwrap()
    }
}

/// An overwrite-mode sponge with 10-padding.
///
/// Absorbs input into the rate, permutes after each full block, and
/// squeezes `OUT` elements. Two-case padding ensures collision
/// resistance for inputs of **variable** length.
///
/// # Padding Rule
///
/// **Case 1 -- partial block** (input ends at position i < RATE):
///
/// ```text
///     Sentinel at position i, zeros after, then permute.
///
///     [a]    RATE=2:  [a, S, 0, ... | cap...]  -> P
///     [a, 0] RATE=2:  [a, 0, S, ... | cap...]  -> P
///                           ^
///                           different position => no collision
/// ```
///
/// **Case 2 -- full block** (input length is a multiple of RATE):
///
/// ```text
///     Add sentinel to first capacity element, then permute.
///
///     [a, b] RATE=2:  [a, b | cap_0 + S, cap_1, ...]  -> P
/// ```
///
/// Sentinel lands in rate (case 1) vs capacity (case 2), so no
/// length-k input can collide with any length != k.
///
/// # Role of the Derangement
///
/// The padding function is a derangement d: a permutation with no
/// fixed points (d(x) != x for all x). This guarantees:
///
/// - **Rate-domain**: d(0) != 0, so the sentinel is always non-zero.
/// - **Capacity-domain**: d(state\[RATE\]) != state\[RATE\], so the
///   capacity always changes.
///
/// ```text
///     Partial:  state[i]    = d(0)           -- sentinel
///     Full:     state[RATE] = d(state[RATE]) -- domain separator
/// ```
///
/// # Construction
///
/// The padding function is a derangement (permutation with no fixed
/// points). The standard choice is `Increment` which computes d(x) = x + 1:
///
/// ```ignore
/// Pad10Sponge::new(permutation, Increment(BabyBear::ONE))  // field
/// Pad10Sponge::new(permutation, Increment(1u64))           // integer
/// ```
///
/// The derangement **must have no fixed points** (d(x) != x for all x).
///
/// # Parameters
///
/// - `WIDTH` -- total state size (rate + capacity).
/// - `RATE`  -- positions overwritten per block.
/// - `OUT`   -- elements squeezed from the final state.
///
/// # Security
///
/// Indifferentiable from a random oracle up to |F|^{c/2} queries (c = WIDTH - RATE).
///
/// Implies collision resistance, preimage resistance, etc. [BDPA08] + [LBM25, Section 3.1].
#[derive(Debug)]
pub struct Pad10Sponge<T, P, D, const WIDTH: usize, const RATE: usize, const OUT: usize> {
    /// The cryptographic permutation applied after each absorbed block.
    permutation: P,

    /// A derangement (permutation with no fixed points) used for padding.
    ///
    /// - Rate-domain:    state[i]    = d(T::default())
    /// - Capacity-domain: state[RATE] = d(state[RATE])
    padding_derangement: D,

    _phantom: PhantomData<T>,
}

impl<T, P: Clone, D: Clone, const WIDTH: usize, const RATE: usize, const OUT: usize> Clone
    for Pad10Sponge<T, P, D, WIDTH, RATE, OUT>
{
    fn clone(&self) -> Self {
        Self {
            permutation: self.permutation.clone(),
            padding_derangement: self.padding_derangement.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<T, P: Copy, D: Copy, const WIDTH: usize, const RATE: usize, const OUT: usize> Copy
    for Pad10Sponge<T, P, D, WIDTH, RATE, OUT>
{
}

impl<T, P, D, const WIDTH: usize, const RATE: usize, const OUT: usize>
    Pad10Sponge<T, P, D, WIDTH, RATE, OUT>
{
    pub const fn new(permutation: P, padding_derangement: D) -> Self {
        const {
            assert!(RATE > 0);
            assert!(RATE < WIDTH);
            assert!(OUT <= WIDTH);
        }
        Self {
            permutation,
            padding_derangement,
            _phantom: PhantomData,
        }
    }
}

impl<T, P, D, const WIDTH: usize, const RATE: usize, const OUT: usize>
    CryptographicHasher<T, [T; OUT]> for Pad10Sponge<T, P, D, WIDTH, RATE, OUT>
where
    T: Default + Copy,
    P: CryptographicPermutation<[T; WIDTH]>,
    D: Derangement<T>,
{
    fn hash_iter<I>(&self, input: I) -> [T; OUT]
    where
        I: IntoIterator<Item = T>,
    {
        // Start from the all-zero state.
        let mut state = [T::default(); WIDTH];

        // Wrap the iterator in `peekable()`.
        //
        // We can detect when input is exhausted on a block boundary without consuming past it.
        let mut input = input.into_iter().peekable();

        loop {
            // Absorb phase: overwrite state[0..RATE] one element at a time.
            //
            // If the iterator runs dry mid-block we enter partial-block padding immediately.
            for i in 0..RATE {
                if let Some(x) = input.next() {
                    // Overwrite the i-th rate position with the next input element.
                    state[i] = x;
                } else {
                    // Partial block: rate-domain 10*-padding.
                    //   position i      <- d(0)    (the sentinel)
                    //   positions i+1.. <- zero    (the "0*" suffix)
                    //
                    //   [a]    RATE=3 -> [a, d(0), 0 | cap...]
                    //   [a, b] RATE=3 -> [a, b, d(0) | cap...]
                    state[i] = self.padding_derangement.permute(T::default());
                    for s in state.iter_mut().take(RATE).skip(i + 1) {
                        *s = T::default();
                    }

                    // Permute the padded state and squeeze.
                    self.permutation.permute_mut(&mut state);
                    return state[..OUT].try_into().unwrap();
                }
            }

            // Full block absorbed. Check whether more input follows.
            if input.peek().is_none() {
                // Capacity-domain padding: apply derangement to state[RATE].
                //
                // Why derangement (not overwrite)?
                // - Overwriting would leak a relation between sponge(M)
                //   and sponge(M || 0^RATE) via multi-block squeeze.
                // - The derangement preserves accumulated capacity
                //   while injecting the domain separator [LBM25].
                state[RATE] = self.padding_derangement.permute(state[RATE]);

                // Permute the padded state and squeeze.
                self.permutation.permute_mut(&mut state);
                return state[..OUT].try_into().unwrap();
            }

            // More input to come. Permute and continue to the next block.
            self.permutation.permute_mut(&mut state);
        }
    }
}

/// Padding-free sponge over a large prime field, accepting 32-bit field elements as input.
///
/// # Security
///
/// **Not** collision-resistant for variable-length inputs.
///
/// For variable-length inputs, use [`MultiField32Pad10Sponge`].
#[derive(Clone, Debug)]
pub struct MultiField32PaddingFreeSponge<
    F,
    PF,
    P,
    const WIDTH: usize,
    const RATE: usize,
    const OUT: usize,
> {
    /// The cryptographic permutation applied after each absorbed block.
    permutation: P,
    /// How many small-field elements fit inside one large-field element.
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
        const {
            assert!(RATE > 0);
            assert!(RATE < WIDTH);
            assert!(OUT <= WIDTH);
        }
        if F::order() >= PF::order() {
            return Err(String::from("F::order() must be less than PF::order()"));
        }

        // Compute packing ratio: how many 32-bit field elements pack into one native field element.
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

        // Example: RATE = 3, num_f_elms = 2, input = [f0..f7]
        //
        //   block_chunk = [f0, f1, f2, f3, f4, f5]  (RATE * 2 = 6 small elems)
        //     chunk 0: [f0, f1] -> pack into PF -> state[0]
        //     chunk 1: [f2, f3] -> pack into PF -> state[1]
        //     chunk 2: [f4, f5] -> pack into PF -> state[2]
        //   -> permute
        //
        //   block_chunk = [f6, f7]  (partial)
        //     chunk 0: [f6, f7] -> pack into PF -> state[0]
        //   -> permute
        for block_chunk in &input.into_iter().chunks(RATE) {
            for (chunk_id, chunk) in (&block_chunk.chunks(self.num_f_elms))
                .into_iter()
                .enumerate()
            {
                // Pack num_f_elms small-field elements into one
                // large-field element via mixed-radix reduction.
                state[chunk_id] = reduce_32(&chunk.collect_vec());
            }
            state = self.permutation.permute(state);
        }

        state[..OUT].try_into().unwrap()
    }
}

/// 10-padded sponge over a large prime field, accepting 32-bit field elements as input.
///
/// # Data Flow
///
/// ```text
///     Small-field input:   [f0, f1,     f2, f3,    f4, f5, ...]
///                           \___/        \___/     \___/
///                          pack into   pack into  pack into
///                          state[0]    state[1]    state[2]
///                          ---- one large-field block ----  -> P
/// ```
///
/// # Padding
///
/// Same two-case scheme as [`Pad10Sponge`], applied in the large-field
/// domain using the multiplicative identity as sentinel.
///
/// # Security
///
/// Collision-resistant for variable-length inputs.
#[derive(Clone, Debug)]
pub struct MultiField32Pad10Sponge<
    F,
    PF,
    P,
    const WIDTH: usize,
    const RATE: usize,
    const OUT: usize,
> {
    /// The cryptographic permutation applied after each absorbed block.
    permutation: P,
    /// Packing ratio: how many small-field elements fit in one large-field element.
    ///
    /// E.g. 64-bit field / 32-bit field = 2.
    num_f_elms: usize,
    _phantom: PhantomData<(F, PF)>,
}

impl<F, PF, P, const WIDTH: usize, const RATE: usize, const OUT: usize>
    MultiField32Pad10Sponge<F, PF, P, WIDTH, RATE, OUT>
where
    F: PrimeField32,
    PF: Field,
{
    pub fn new(permutation: P) -> Result<Self, String> {
        if F::order() >= PF::order() {
            return Err(String::from("F::order() must be less than PF::order()"));
        }

        // E.g. PF has 64 bits, F has 32 bits -> 2 small elems per large elem.
        let num_f_elms = PF::bits() / F::bits();
        Ok(Self {
            permutation,
            num_f_elms,
            _phantom: PhantomData,
        })
    }
}

impl<F, PF, P, const WIDTH: usize, const RATE: usize, const OUT: usize>
    CryptographicHasher<F, [PF; OUT]> for MultiField32Pad10Sponge<F, PF, P, WIDTH, RATE, OUT>
where
    F: PrimeField32,
    PF: PrimeField + Default + Copy,
    P: CryptographicPermutation<[PF; WIDTH]>,
{
    fn hash_iter<I>(&self, input: I) -> [PF; OUT]
    where
        I: IntoIterator<Item = F>,
    {
        // All-zero initial state in the large-field domain.
        let mut state = [PF::default(); WIDTH];

        // The padding sentinel: multiplicative identity in the large field.
        let sentinel = PF::ONE;

        // Tracks how many large-field rate slots the current block filled.
        //
        //   After the loop:
        //     last_chunk_len = 0  &&  absorbed_any = true  -> full-block case
        //     last_chunk_len = 0  &&  absorbed_any = false -> empty input
        //     last_chunk_len > 0                           -> partial block
        let mut last_chunk_len = 0;
        let mut absorbed_any = false;

        // Outer loop: consume RATE * num_f_elms small-field elements per iteration.
        //
        // That fills exactly RATE large-field rate slots.
        //
        //   Example: RATE = 3, num_f_elms = 2
        //
        //     iter 1: [f0..f5]         -> state = [pack(f0,f1), pack(f2,f3), pack(f4,f5), cap...]
        //     full block (3 = RATE)    -> permute, reset last_chunk_len = 0
        //
        //     iter 2: [f6..f9]     -> state = [pack(f6,f7), pack(f8,f9), old, cap...]
        //     partial (2 < RATE)   -> skip permute, pad below
        for block_chunk in &input.into_iter().chunks(RATE * self.num_f_elms) {
            absorbed_any = true;
            last_chunk_len = 0;

            // Inner loop:
            // - group num_f_elms small-field elements,
            // - pack each group into one large-field element at the next rate slot.
            for (chunk_id, chunk) in (&block_chunk.chunks(self.num_f_elms))
                .into_iter()
                .enumerate()
            {
                // Mixed-radix reduction: num_f_elms small -> 1 large.
                state[chunk_id] = reduce_32(&chunk.collect_vec());

                // Record how far we got (1-indexed).
                last_chunk_len = chunk_id + 1;
            }

            // Only permute when the block is full.
            // Partial blocks fall through to the padding logic below.
            if last_chunk_len == RATE {
                state = self.permutation.permute(state);
                last_chunk_len = 0;
            }
        }

        // Two-case padding in the large-field domain.
        //
        //   last_chunk_len = 0 + absorbed -> capacity pad: state[RATE] += 1
        //   last_chunk_len > 0 or empty   -> rate pad:     state[pos] = 1, zeros after
        if last_chunk_len == 0 && absorbed_any {
            // Full block: add sentinel to first capacity element.
            state[RATE] += sentinel;
        } else {
            // Partial block or empty: sentinel at next open rate slot.
            state[last_chunk_len] = sentinel;

            // Zero-fill remaining rate slots (the "0*" suffix).
            for s in state.iter_mut().take(RATE).skip(last_chunk_len + 1) {
                *s = PF::default();
            }
        }

        // Permute the padded state.
        state = self.permutation.permute(state);

        // Squeeze the first OUT large-field elements.
        state[..OUT].try_into().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use p3_field::PrimeCharacteristicRing;
    use p3_koala_bear::KoalaBear;
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
            // Sum all elements and broadcast.
            let sum: T = input.iter().copied().fold(T::default(), |acc, x| acc + x);
            // Set every element to the sum
            *input = [sum; WIDTH];
        }
    }

    impl<T, const WIDTH: usize> CryptographicPermutation<[T; WIDTH]> for MockPermutation where
        T: Copy + core::ops::Add<Output = T> + Default
    {
    }

    /// Mock: weighted sum. output[i] = sum_j state[j] * (j+1).
    ///
    /// Position-sensitive: sentinel position affects the output.
    #[derive(Clone)]
    struct WeightedSumPermutation;

    impl<const WIDTH: usize> Permutation<[KoalaBear; WIDTH]> for WeightedSumPermutation {
        fn permute_mut(&self, input: &mut [KoalaBear; WIDTH]) {
            // Weighted sum: element j contributes input[j] * (j + 1).
            let weighted_sum: KoalaBear = input
                .iter()
                .enumerate()
                .map(|(j, &x)| x * KoalaBear::new((j + 1) as u32))
                .fold(KoalaBear::ZERO, |a, b| a + b);

            // Broadcast the weighted sum to every position.
            *input = [weighted_sum; WIDTH];
        }
    }

    impl<const WIDTH: usize> CryptographicPermutation<[KoalaBear; WIDTH]> for WeightedSumPermutation {}

    #[test]
    fn test_padding_free_sponge_basic() {
        // Fixture: WIDTH = 4, RATE = 2, OUT = 2, plain-sum mock.
        //
        // Step-by-step absorption of [1, 2, 3, 4, 5]:
        //
        //   Initial state:                  [0, 0, 0, 0]
        //
        //   Block 1: overwrite rate         [1, 2, 0, 0]
        //   Permute (sum = 3):              [3, 3, 3, 3]
        //
        //   Block 2: overwrite rate         [3, 4, 3, 3]
        //   Permute (sum = 13):             [13, 13, 13, 13]
        //
        //   Partial block: overwrite [5]    [5, 13, 13, 13]
        //   Permute (sum = 44):             [44, 44, 44, 44]
        //
        //   Squeeze OUT = 2:                [44, 44]
        const WIDTH: usize = 4;
        const RATE: usize = 2;
        const OUT: usize = 2;

        let sponge = PaddingFreeSponge::<MockPermutation, WIDTH, RATE, OUT>::new(MockPermutation);

        let input = [1, 2, 3, 4, 5];
        let output = sponge.hash_iter(input);

        assert_eq!(output, [44; OUT]);
    }

    #[test]
    fn test_padding_free_sponge_empty_input() {
        // Empty input: no elements absorbed, no permutation called.
        //
        // The initial all-zero state is returned directly.
        const WIDTH: usize = 4;
        const RATE: usize = 2;
        const OUT: usize = 2;

        let sponge = PaddingFreeSponge::<MockPermutation, WIDTH, RATE, OUT>::new(MockPermutation);

        let input: [u64; 0] = [];
        let output = sponge.hash_iter(input);

        // Squeeze from the untouched zero state.
        assert_eq!(output, [0; OUT]);
    }

    #[test]
    fn test_padding_free_sponge_exact_block_size() {
        // Fixture: WIDTH = 6, RATE = 3, OUT = 2, plain-sum mock.
        //
        // Input [10, 20, 30] fills exactly one block:
        //
        //   Block 1: overwrite rate         [10, 20, 30, 0, 0, 0]
        //   Permute (sum = 60):             [60, 60, 60, 60, 60, 60]
        //
        //   Squeeze OUT = 2:               [60, 60]
        const WIDTH: usize = 6;
        const RATE: usize = 3;
        const OUT: usize = 2;

        let sponge = PaddingFreeSponge::<MockPermutation, WIDTH, RATE, OUT>::new(MockPermutation);

        let input = [10, 20, 30];
        let output = sponge.hash_iter(input);

        assert_eq!(output, [60; OUT]);
    }

    #[test]
    fn test_pad10_no_collision_partial_vs_trailing_zero() {
        // Invariant: [a] != [a, 0].
        //
        //   [a]    -> partial pad -> [42, 1, 0, 0]   wt_sum = 44
        //   [a, 0] -> full block  -> [42, 0, 1, 0]   wt_sum = 45
        //                                 ^  sentinel at different pos
        let sponge =
            Pad10Sponge::<KoalaBear, WeightedSumPermutation, Increment<KoalaBear>, 4, 2, 2>::new(
                WeightedSumPermutation,
                Increment(KoalaBear::ONE),
            );

        let a = KoalaBear::new(42);

        let hash_short = sponge.hash_iter([a]);
        let hash_long = sponge.hash_iter([a, KoalaBear::ZERO]);

        assert_ne!(hash_short, hash_long);
    }

    #[test]
    fn test_pad10_no_collision_full_block_vs_extended() {
        // Invariant: [a, b] (1 full block) != [a, b, c] (1.5 blocks).
        //
        //   [a, b]    -> capacity pad -> [1, 2 | 0+1, 0]  -> P
        //   [a, b, c] -> P([1, 2, 0, 0]) -> partial [5, 1, *, *] -> P
        let sponge =
            Pad10Sponge::<KoalaBear, WeightedSumPermutation, Increment<KoalaBear>, 4, 2, 2>::new(
                WeightedSumPermutation,
                Increment(KoalaBear::ONE),
            );

        let a = KoalaBear::new(1);
        let b = KoalaBear::new(2);
        let c = KoalaBear::new(5);

        let hash_full = sponge.hash_iter([a, b]);
        let hash_ext = sponge.hash_iter([a, b, c]);

        assert_ne!(hash_full, hash_ext);
    }

    #[test]
    fn test_pad10_empty_input_is_nontrivial() {
        // Empty input -> partial pad at position 0.
        //
        //   state = [1, 0, 0, 0]  ->  wt_sum = 1
        //
        // Digest must NOT be the all-zero default.
        let sponge =
            Pad10Sponge::<KoalaBear, WeightedSumPermutation, Increment<KoalaBear>, 4, 2, 2>::new(
                WeightedSumPermutation,
                Increment(KoalaBear::ONE),
            );

        let output = sponge.hash_iter(core::iter::empty::<KoalaBear>());

        assert_eq!(output, [KoalaBear::ONE; 2]);
    }

    #[test]
    fn test_pad10_basic_absorption() {
        // Fixture: WIDTH = 4, RATE = 2, OUT = 2, weighted-sum mock.
        //
        // Step-by-step absorption of [1, 2, 3, 4, 5]:
        //
        //   Initial state:                         [0, 0, 0, 0]
        //
        //   Block 1: overwrite rate                [1, 2, 0, 0]
        //   (peek: more input)
        //   Weighted sum = 1*1 + 2*2 + 0 + 0 = 5
        //   Permute:                               [5, 5, 5, 5]
        //
        //   Block 2: overwrite rate                [3, 4, 5, 5]
        //   (peek: more input)
        //   Weighted sum = 3*1 + 4*2 + 5*3 + 5*4 = 46
        //   Permute:                               [46, 46, 46, 46]
        //
        //   Partial block: overwrite position 0    [5, 46, 46, 46]
        //   Rate-domain pad at position 1:         [5, 1, 46, 46]
        //                                           ^  ^ sentinel
        //
        //   Weighted sum = 5*1 + 1*2 + 46*3 + 46*4 = 329
        //
        //   Permute:                               [329, 329, 329, 329]
        //
        //   Squeeze OUT = 2:                       [329, 329]
        let sponge =
            Pad10Sponge::<KoalaBear, WeightedSumPermutation, Increment<KoalaBear>, 4, 2, 2>::new(
                WeightedSumPermutation,
                Increment(KoalaBear::ONE),
            );

        let input = [1u32, 2, 3, 4, 5].map(KoalaBear::new);
        let output = sponge.hash_iter(input);

        assert_eq!(output, [KoalaBear::new(329); 2]);
    }

    #[test]
    fn test_pad10_exact_block_uses_capacity_padding() {
        // Fixture: WIDTH = 6, RATE = 3, OUT = 2, weighted-sum mock.
        //
        // Input [10, 20, 30] fills exactly one block:
        //
        //   Block 1: overwrite rate                [10, 20, 30, 0, 0, 0]
        //   (peek: no more input -> capacity pad)
        //   state[RATE] += 1 -> state[3] += 1:     [10, 20, 30, 1, 0, 0]
        //
        //   Weighted sum = 10*1 + 20*2 + 30*3 + 1*4 + 0 + 0 = 144
        //   Permute:                               [144; 6]
        //
        //   Squeeze OUT = 2:                       [144, 144]
        let sponge =
            Pad10Sponge::<KoalaBear, WeightedSumPermutation, Increment<KoalaBear>, 6, 3, 2>::new(
                WeightedSumPermutation,
                Increment(KoalaBear::ONE),
            );

        let input = [10u32, 20, 30].map(KoalaBear::new);
        let output = sponge.hash_iter(input);

        assert_eq!(output, [KoalaBear::new(144); 2]);
    }

    // Arbitrary field element from any u32.
    fn arb_koala_bear() -> impl Strategy<Value = KoalaBear> {
        any::<u32>().prop_map(KoalaBear::new)
    }

    proptest! {
        #[test]
        fn prop_pad10_different_lengths_never_collide(
            // Generate a random base input of 1..=8 elements.
            base in prop::collection::vec(arb_koala_bear(), 1..=8)
        ) {
            // Invariant: hash(msg) != hash(msg ++ [0]).
            //
            // This is the exact attack that padding prevents.
            let sponge = Pad10Sponge::<KoalaBear, WeightedSumPermutation, Increment<KoalaBear>, 4, 2, 2>::new(
                WeightedSumPermutation,
                Increment(KoalaBear::ONE),
            );

            // Hash the base message.
            let hash_base = sponge.hash_iter(base.iter().copied());

            // Extend by one zero element and hash again.
            let mut extended = base.clone();
            extended.push(KoalaBear::ZERO);
            let hash_extended = sponge.hash_iter(extended.iter().copied());

            // The two digests must differ.
            prop_assert_ne!(
                hash_base,
                hash_extended,
                "base len={}, extended len={}",
                base.len(),
                base.len() + 1
            );
        }

        #[test]
        fn prop_pad10_deterministic(
            // Generate a random input of 0..=12 elements.
            input in prop::collection::vec(arb_koala_bear(), 0..=12)
        ) {
            // Invariant: hash(x) == hash(x). No hidden mutable state.
            let sponge = Pad10Sponge::<KoalaBear, WeightedSumPermutation, Increment<KoalaBear>, 4, 2, 2>::new(
                WeightedSumPermutation,
                Increment(KoalaBear::ONE),
            );

            // Hash the input twice with independent iterator clones.
            let hash_1 = sponge.hash_iter(input.iter().copied());
            let hash_2 = sponge.hash_iter(input.iter().copied());

            prop_assert_eq!(hash_1, hash_2);
        }

        #[test]
        fn prop_pad10_prefix_differs_from_full(
            // Generate a random input of 2..=8 elements.
            input in prop::collection::vec(arb_koala_bear(), 2..=8)
        ) {
            // Invariant: hash(input[..k]) != hash(input) for all k < len.
            //
            // Tests collision resistance across arbitrary length gaps.
            let sponge = Pad10Sponge::<KoalaBear, WeightedSumPermutation, Increment<KoalaBear>, 4, 2, 2>::new(
                WeightedSumPermutation,
                Increment(KoalaBear::ONE),
            );

            // Hash the full input.
            let hash_full = sponge.hash_iter(input.iter().copied());

            // Hash every strict prefix and verify distinctness.
            for k in 1..input.len() {
                let hash_prefix = sponge.hash_iter(input[..k].iter().copied());
                prop_assert_ne!(
                    hash_full,
                    hash_prefix,
                    "prefix len={} collides with full len={}",
                    k,
                    input.len()
                );
            }
        }
    }
}
