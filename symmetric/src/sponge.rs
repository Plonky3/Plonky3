//! Sponge-based hash functions built from cryptographic permutations.
//!
//! # Background
//!
//! A sponge \[BDPV07\] hashes an input using a fixed-width permutation P.
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
//!   Provides collision resistance up to |F|^{c/2} queries \[BDPA08\].
//!
//! This module uses the **overwrite** variant: each input block
//! overwrites (rather than XORs into) the rate portion.
//! Security carries over from the standard sponge \[BDPA08, AMP10\].
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
use p3_field::{
    PrimeField, PrimeField32, absorb_radix_bits, max_shifted_absorb_injective_limbs,
    reduce_packed_shifted,
};

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
        const {
            assert!(RATE > 0);
            assert!(RATE < WIDTH);
            assert!(OUT <= WIDTH);
        }
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

/// Absorb the remainder of an input stream into the rate window, one block at a time.
///
/// # Overview
///
/// - Shared by the full hashing path and the precomputed-state path.
/// - Each iteration consumes `RATE` elements.
/// - Each iteration runs one permutation.
/// - The first element of a block lands at the topmost state index.
/// - The remaining elements fill the rate window downward.
/// - The capacity is left untouched between blocks.
/// - The capacity carries the chaining variable forward.
///
/// # Per-block diagram
///
/// ```text
///     WIDTH=4, RATE=2, input = [a, b, c, d]
///
///         block 1                            block 2
///         -------                            -------
///         state[3] = a                       state[3] = c
///         state[2] = b                       state[2] = d
///         permute(state)                     permute(state)
/// ```
///
/// # Panics
///
/// - The iterator length must be a multiple of `RATE`.
/// - The helper panics if a block starts but cannot complete.
#[inline]
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
    // Shape constraints checked at compile time.
    //
    // - `RATE > 0`     — at least one element per block.
    // - `RATE < WIDTH` — at least one slot reserved for the capacity.
    // - `OUT <= WIDTH` — digest cannot exceed the state size.
    const {
        assert!(RATE > 0);
        assert!(RATE < WIDTH);
        assert!(OUT <= WIDTH);
    }

    // Each outer iteration processes one rate-sized block.
    //
    //     1. write the topmost slot from the starting element
    //     2. fill the remaining rate slots downward
    //     3. permute the full state
    //
    // The loop exits when no more starting element arrives.
    // That means the stream ended cleanly on a block boundary.
    while let Some(elem) = iter.next() {
        // Starting element of a block always lands at the topmost slot.
        state[WIDTH - 1] = elem;

        // Remaining rate slots, descending from `WIDTH - 2` to `WIDTH - RATE`.
        //
        // A drained iterator here means the input length is not a multiple of `RATE` — a programmer error.
        for pos in (WIDTH - RATE..WIDTH - 1).rev() {
            state[pos] = iter
                .next()
                .expect("iterator length must be a multiple of RATE");
        }

        // Diffuse the new rate slots into the capacity for the next block.
        permutation.permute_mut(state);
    }

    // Squeeze: return the first `OUT` positions of the final state.
    //
    // - Positions `0..(WIDTH - RATE)` form the capacity window.
    // - For `OUT <= WIDTH - RATE` the digest sits entirely in the capacity.
    // - For larger `OUT` it crosses into the rate.
    // - Both choices are sound in the random-permutation model.
    // - Any contiguous truncation is indifferentiable from random output.
    state[..OUT]
        .try_into()
        .expect("OUT is at most WIDTH by the const assertion above")
}

/// A padding-free sponge that absorbs input into the state right-to-left.
///
/// # Overview
///
/// - Standard sponge construction with the rate window at the high state indices.
/// - Streams reverse-ordered iterators with no pre-buffering.
/// - Typical use: Merkle tree construction over FRI/WHIR evaluations.
///
/// # Reference
///
/// Bertoni, Daemen, Peeters, Van Assche, *"Sponge Functions"*, ECRYPT Hash Workshop 2007.
///
/// Paper: <https://keccak.team/files/SpongeFunctions.pdf>.
///
/// # State layout
///
/// ```text
///     index:  0                       ← length IV slot
///     index:  1 .. WIDTH-RATE         ← rest of capacity (chaining variable)
///     index:  WIDTH-RATE .. WIDTH     ← rate (absorption window)
/// ```
///
/// - The capacity is **never directly overwritten by input** — it acts as a barrier.
/// - The first capacity slot is initialised with the input length to defeat length-extension attacks.
/// - The rate sits at the high indices because that is the side absorption writes to.
/// - The digest is the first `OUT` positions of the final state.
///
/// # Absorption
///
/// ```text
///     WIDTH=4, RATE=2, length-IV = L, next block = [e, f]
///
///         index:    0    1    2    3
///                 +----+----+----+----+
///                 | L  |  ?  | f  | e  |   ← only the rate window is rewritten
///                 +----+----+----+----+
///                   capacity     rate
/// ```
///
/// `WIDTH` is the sum of the rate and capacity sizes.
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

    /// Slice-based reference form of the RTL hash.
    ///
    /// # Overview
    ///
    /// - Canonical algorithm for the construction.
    /// - The streaming form is the optimisation of this slice form.
    /// - The caller supplies the length-IV; passing `data.len()` (or its
    ///   field-element image) is the conventional choice.
    ///
    /// # Algorithm
    ///
    /// 1. Initialise the state as `[length_iv, 0, …, 0]`.
    /// 2. For each `RATE`-sized chunk of `data`, in reverse order:
    ///    overwrite the rate window with the chunk, then permute.
    /// 3. Squeeze the first `OUT` positions.
    ///
    /// # Panics
    ///
    /// `data.len()` must be a multiple of `RATE`.
    pub fn hash_slice<T>(&self, data: &[T], length_iv: T) -> [T; OUT]
    where
        T: Default + Copy,
        P: CryptographicPermutation<[T; WIDTH]>,
    {
        const {
            assert!(RATE > 0);
            assert!(RATE < WIDTH);
            assert!(OUT <= WIDTH);
        }

        // Every chunk must be `RATE` elements wide.
        assert!(
            data.len().is_multiple_of(RATE),
            "input length must be a multiple of RATE",
        );

        // Initial state: length-IV in the first capacity slot, rest zero.
        let mut state = [T::default(); WIDTH];
        state[0] = length_iv;

        // Walk chunks of `data` backward, rewriting only the rate window each time.
        for chunk in data.chunks_exact(RATE).rev() {
            state[WIDTH - RATE..].copy_from_slice(chunk);
            self.permutation.permute_mut(&mut state);
        }

        state[..OUT]
            .try_into()
            .expect("OUT is at most WIDTH by the const assertion above")
    }

    /// Precompute the state that results from absorbing an all-zero prefix.
    ///
    /// # Overview
    ///
    /// - Midstate caching for inputs whose first chunks are known zero.
    /// - In FRI/WHIR many Merkle leaves are zero.
    /// - Their leading zeros feed identical permutation calls every time.
    /// - Reusing the post-zero state saves work, like SHA-256 midstate caching in Bitcoin mining.
    ///
    /// # Algorithm
    ///
    /// 1. Initialise the state as `[iv_first, 0, …, 0]`.
    /// 2. For each of `n_zero_chunks` iterations: zero the rate window, then permute.
    /// 3. Return the resulting state.
    ///
    /// # Diagram
    ///
    /// ```text
    ///     iv_first = L, n_zero_chunks = 3, WIDTH = 4, RATE = 2
    ///
    ///         start    : state = [ L  0  0  0 ]
    ///         zero rate, permute → s_1
    ///         zero rate, permute → s_2
    ///         zero rate, permute → s_3  ← returned
    /// ```
    pub fn precompute_zero_suffix_state<T>(&self, iv_first: T, n_zero_chunks: usize) -> [T; WIDTH]
    where
        T: Default + Copy,
        P: CryptographicPermutation<[T; WIDTH]>,
    {
        const {
            assert!(RATE > 0);
            assert!(RATE < WIDTH);
            assert!(OUT <= WIDTH);
        }

        // Initial state: caller-supplied IV in the first capacity slot, rest zero.
        let mut state = [T::default(); WIDTH];
        state[0] = iv_first;

        // Each iteration absorbs one all-zero rate block.
        // The rate is re-zeroed before each permutation so capacity carries forward.
        for _ in 0..n_zero_chunks {
            state[WIDTH - RATE..].fill(T::default());
            self.permutation.permute_mut(&mut state);
        }

        state
    }

    /// Continue right-to-left absorption from a precomputed initial state.
    ///
    /// # Overview
    ///
    /// - Companion to the zero-suffix precomputation.
    /// - The caller passes the cached state and the remaining non-zero elements.
    /// - Absorption resumes block-by-block from that state.
    ///
    /// # Returns
    ///
    /// The first `OUT` positions of the final state.
    ///
    /// # Panics
    ///
    /// Panics if the iterator length is not a multiple of `RATE`.
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

        // Copy the precomputed state.
        // The caller's array stays immutable.
        let mut state = *initial_state;

        // Hand the remaining elements to the shared per-block absorber.
        let mut iter = iter.into_iter();
        absorb_rtl_chunks::<T, P, _, WIDTH, RATE, OUT>(&self.permutation, &mut state, &mut iter)
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
/// Implies collision resistance, preimage resistance, etc. \[BDPA08\] + \[LBM25, Section 3.1\].
#[derive(Debug)]
pub struct Pad10Sponge<T, P, D, const WIDTH: usize, const RATE: usize, const OUT: usize> {
    /// The cryptographic permutation applied after each absorbed block.
    permutation: P,

    /// A derangement (permutation with no fixed points) used for padding.
    ///
    /// - Rate-domain:    `state[i]    = d(T::default())`
    /// - Capacity-domain: `state[RATE] = d(state[RATE])`
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
    /// Radix used for shifted packing into the large field.
    radix_bits: u32,
    _phantom: PhantomData<(F, PF)>,
}

impl<F, PF, P, const WIDTH: usize, const RATE: usize, const OUT: usize>
    MultiField32PaddingFreeSponge<F, PF, P, WIDTH, RATE, OUT>
where
    F: PrimeField32,
    PF: PrimeField,
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

        // Use shifted-radix injective packing for robust absorb encoding.
        let num_f_elms = max_shifted_absorb_injective_limbs::<F, PF>();
        let radix_bits = absorb_radix_bits::<F>();
        Ok(Self {
            permutation,
            num_f_elms,
            radix_bits,
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
        for block_chunk in &input.into_iter().chunks(RATE * self.num_f_elms) {
            for (chunk_id, chunk) in (&block_chunk.chunks(self.num_f_elms))
                .into_iter()
                .enumerate()
            {
                // Pack num_f_elms small-field elements into one large-field
                // element via shifted-radix reduction.
                state[chunk_id] = reduce_packed_shifted(&chunk.collect_vec(), self.radix_bits);
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
    /// Radix used for shifted packing into the large field.
    radix_bits: u32,
    _phantom: PhantomData<(F, PF)>,
}

impl<F, PF, P, const WIDTH: usize, const RATE: usize, const OUT: usize>
    MultiField32Pad10Sponge<F, PF, P, WIDTH, RATE, OUT>
where
    F: PrimeField32,
    PF: PrimeField,
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

        // Use shifted-radix injective packing for robust absorb encoding.
        let num_f_elms = max_shifted_absorb_injective_limbs::<F, PF>();
        let radix_bits = absorb_radix_bits::<F>();
        Ok(Self {
            permutation,
            num_f_elms,
            radix_bits,
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
                // Shifted-radix reduction: num_f_elms small -> 1 large.
                state[chunk_id] = reduce_packed_shifted(&chunk.collect_vec(), self.radix_bits);

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
    use alloc::vec::Vec;

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

    // Position-sensitive mock for detecting *order* differences.
    //
    // - The plain mock collapses the state to a single sum.
    // - States sharing the same multiset collapse to the same vector.
    // - This mock replaces slot `i` with the prefix sum over `0..=i`.
    // - Different positions for the same multiset now yield different states.
    #[derive(Clone)]
    struct PrefixSumPermutation;

    impl<T, const WIDTH: usize> Permutation<[T; WIDTH]> for PrefixSumPermutation
    where
        T: Copy + core::ops::Add<Output = T> + Default,
    {
        fn permute_mut(&self, input: &mut [T; WIDTH]) {
            // Walk left-to-right and write back the running prefix sum.
            let mut acc = T::default();
            for slot in input.iter_mut() {
                acc = acc + *slot;
                *slot = acc;
            }
        }
    }

    impl<T, const WIDTH: usize> CryptographicPermutation<[T; WIDTH]> for PrefixSumPermutation where
        T: Copy + core::ops::Add<Output = T> + Default
    {
    }

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

    #[test]
    fn test_rtl_sponge_basic() {
        // Hand-traced state transitions for `data = [1, 2, 3, 4, 5, 6]`, `length_iv = 6`.
        //
        //   Initial state (length-IV in capacity slot 0):  [6, 0, 0, 0]
        //
        //   Chunks of `data` are processed in reverse:
        //
        //   chunk [5, 6]: state[2..4] = [5, 6]     [6, 0, 5, 6]
        //   permute (sum = 17):                    [17, 17, 17, 17]
        //
        //   chunk [3, 4]: state[2..4] = [3, 4]     [17, 17, 3, 4]
        //   permute (sum = 41):                    [41, 41, 41, 41]
        //
        //   chunk [1, 2]: state[2..4] = [1, 2]     [41, 41, 1, 2]
        //   permute (sum = 85):                    [85, 85, 85, 85]
        //
        //   Squeeze OUT = 2:                       [85, 85]
        const WIDTH: usize = 4;
        const RATE: usize = 2;
        const OUT: usize = 2;

        let sponge =
            RtlPaddingFreeSponge::<MockPermutation, WIDTH, RATE, OUT>::new(MockPermutation);
        let data = [1u64, 2, 3, 4, 5, 6];
        let output = sponge.hash_slice(&data, data.len() as u64);

        assert_eq!(output, [85; OUT]);
    }

    #[test]
    fn test_rtl_sponge_single_block() {
        // `data` is exactly one RATE-sized chunk; one permutation runs.
        //
        //   Initial state:                          [2, 0, 0, 0]
        //
        //   chunk [10, 20]: state[2..4] = [10, 20]  [2, 0, 10, 20]
        //   permute (sum = 32):                     [32, 32, 32, 32]
        //
        //   Squeeze OUT = 2:                        [32, 32]
        const WIDTH: usize = 4;
        const RATE: usize = 2;
        const OUT: usize = 2;

        let sponge =
            RtlPaddingFreeSponge::<MockPermutation, WIDTH, RATE, OUT>::new(MockPermutation);
        let data = [10u64, 20];
        let output = sponge.hash_slice(&data, data.len() as u64);

        assert_eq!(output, [32; OUT]);
    }

    #[test]
    fn test_rtl_sponge_empty_input() {
        // Empty input: no chunks, no permutation.
        // State stays `[length_iv, 0, ..., 0]` and the squeeze reads from it directly.
        const WIDTH: usize = 4;
        const RATE: usize = 2;
        const OUT: usize = 2;

        let sponge =
            RtlPaddingFreeSponge::<MockPermutation, WIDTH, RATE, OUT>::new(MockPermutation);
        let output = sponge.hash_slice::<u64>(&[], 0);

        assert_eq!(output, [0; OUT]);
    }

    #[test]
    #[should_panic(expected = "input length must be a multiple of RATE")]
    fn test_rtl_sponge_misaligned_panics() {
        // Slice length must be a multiple of `RATE`. Length 5 with `RATE = 2` is rejected.
        const WIDTH: usize = 4;
        const RATE: usize = 2;
        const OUT: usize = 2;

        let sponge =
            RtlPaddingFreeSponge::<MockPermutation, WIDTH, RATE, OUT>::new(MockPermutation);
        let _ = sponge.hash_slice(&[1u64, 2, 3, 4, 5], 5);
    }

    #[test]
    fn test_rtl_sponge_out_larger_than_capacity() {
        // `OUT > WIDTH - RATE`: the squeeze crosses from capacity into rate.
        //
        //   WIDTH = 4, RATE = 2  →  capacity is positions [0..2]
        //   OUT = 3              →  digest = state[0..3]
        //
        //   data = [1, 2, 3, 4], length_iv = 4. Chunks reversed: [[3,4], [1,2]].
        //
        //   Initial state:                       [4, 0, 0, 0]
        //   chunk [3, 4]:  state[2..4] = [3, 4]  [4, 0, 3, 4]
        //   permute (sum = 11):                  [11; 4]
        //   chunk [1, 2]:  state[2..4] = [1, 2]  [11, 11, 1, 2]
        //   permute (sum = 25):                  [25; 4]
        //
        //   Squeeze OUT = 3:                     [25, 25, 25]
        const WIDTH: usize = 4;
        const RATE: usize = 2;
        const OUT: usize = 3;

        let sponge =
            RtlPaddingFreeSponge::<MockPermutation, WIDTH, RATE, OUT>::new(MockPermutation);
        let data = [1u64, 2, 3, 4];
        let output = sponge.hash_slice(&data, data.len() as u64);

        assert_eq!(output, [25; OUT]);
    }

    #[test]
    fn test_rtl_vs_ltr_different_outputs() {
        // The RTL and LTR sponges are different constructions:
        //
        // - The RTL sponge encodes the input length in the first capacity slot.
        // - The LTR sponge uses an all-zero IV.
        // - Rate sits at opposite ends of the state.
        //
        // For any non-trivial input the digests must differ.
        const WIDTH: usize = 4;
        const RATE: usize = 2;
        const OUT: usize = 2;

        let data = [1u64, 2, 3, 4, 5, 6];

        let ltr_sponge =
            PaddingFreeSponge::<PrefixSumPermutation, WIDTH, RATE, OUT>::new(PrefixSumPermutation);
        let ltr_output = ltr_sponge.hash_iter(data);

        let rtl_sponge = RtlPaddingFreeSponge::<PrefixSumPermutation, WIDTH, RATE, OUT>::new(
            PrefixSumPermutation,
        );
        let rtl_output = rtl_sponge.hash_slice(&data, data.len() as u64);

        assert_ne!(
            ltr_output, rtl_output,
            "RTL and LTR sponges must produce different outputs for the same input"
        );
    }

    #[test]
    fn test_precompute_zero_suffix_all_zero() {
        // `iv_first = 0`, all-zero capacity, sum-broadcast mock: every state stays zero.
        const WIDTH: usize = 4;
        const RATE: usize = 2;
        const OUT: usize = 2;

        let sponge =
            RtlPaddingFreeSponge::<MockPermutation, WIDTH, RATE, OUT>::new(MockPermutation);
        let state: [u64; WIDTH] = sponge.precompute_zero_suffix_state(0u64, 3);

        assert_eq!(state, [0u64; WIDTH]);
    }

    #[test]
    fn test_precompute_zero_suffix_with_iv() {
        // Non-zero IV drives a non-trivial trajectory.
        //
        //   Initial:                       [5, 0, 0, 0]
        //   zero rate, permute (sum=5):    [5, 5, 5, 5]
        //   zero rate, permute (sum=10):   [10, 10, 10, 10]
        //   zero rate, permute (sum=20):   [20, 20, 20, 20]
        const WIDTH: usize = 4;
        const RATE: usize = 2;
        const OUT: usize = 2;

        let sponge =
            RtlPaddingFreeSponge::<MockPermutation, WIDTH, RATE, OUT>::new(MockPermutation);
        let state: [u64; WIDTH] = sponge.precompute_zero_suffix_state(5u64, 3);

        assert_eq!(state, [20u64; WIDTH]);
    }

    #[test]
    fn test_precompute_zero_suffix_no_chunks() {
        // Zero chunks: no permutation runs, state stays at `[iv_first, 0, ..., 0]`.
        const WIDTH: usize = 4;
        const RATE: usize = 2;
        const OUT: usize = 2;

        let sponge =
            RtlPaddingFreeSponge::<MockPermutation, WIDTH, RATE, OUT>::new(MockPermutation);
        let state: [u64; WIDTH] = sponge.precompute_zero_suffix_state(7u64, 0);

        assert_eq!(state, [7, 0, 0, 0]);
    }

    #[test]
    fn test_hash_with_initial_state_matches_full_hash() {
        // Invariant: hashing `[non_zero_prefix ‖ zero_suffix]` in one shot via the
        // slice form must equal precomputing the zero suffix and then continuing
        // from that midstate with the reverse-iterator form on the non-zero prefix.
        //
        //   data = [5, 6, 0, 0, 0, 0]   (one non-zero block followed by two zero blocks)
        //   length_iv = 6
        //
        //   Path A: hash_slice(data, 6).
        //   Path B: precompute(6, 2) → midstate; hash_with_initial_state(midstate, [6, 5]).
        const WIDTH: usize = 4;
        const RATE: usize = 2;
        const OUT: usize = 2;

        let sponge =
            RtlPaddingFreeSponge::<MockPermutation, WIDTH, RATE, OUT>::new(MockPermutation);

        let data = [5u64, 6, 0, 0, 0, 0];
        let length_iv = data.len() as u64;

        let full_output = sponge.hash_slice(&data, length_iv);

        let initial_state: [u64; WIDTH] = sponge.precompute_zero_suffix_state(length_iv, 2);
        let partial_output = sponge.hash_with_initial_state(&initial_state, [6u64, 5]);

        assert_eq!(full_output, partial_output);
    }

    #[test]
    fn test_hash_with_initial_state_empty_remaining() {
        // Empty iterator: no permutation runs, the squeeze reads from `initial_state[..OUT]`.
        const WIDTH: usize = 4;
        const RATE: usize = 2;
        const OUT: usize = 2;

        let sponge =
            RtlPaddingFreeSponge::<MockPermutation, WIDTH, RATE, OUT>::new(MockPermutation);

        let initial_state = [10u64, 20, 30, 40];
        let output = sponge.hash_with_initial_state(&initial_state, core::iter::empty::<u64>());

        assert_eq!(output, [10, 20]);
    }

    #[test]
    #[should_panic(expected = "iterator length must be a multiple of RATE")]
    fn test_hash_with_initial_state_non_multiple_of_rate_panics() {
        // Iterator length must be a multiple of `RATE`.
        // Supplying a single element with `RATE = 2` starts a block that cannot complete.
        const WIDTH: usize = 4;
        const RATE: usize = 2;
        const OUT: usize = 2;

        let sponge =
            RtlPaddingFreeSponge::<MockPermutation, WIDTH, RATE, OUT>::new(MockPermutation);

        let initial_state = [0u64; WIDTH];
        let _ = sponge.hash_with_initial_state(&initial_state, [5u64]);
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

        #[test]
        fn proptest_rtl_precompute_equivalence(
            n_zero_blocks in 0..=8usize,
            n_nonzero_blocks in 0..=4usize,
            nonzero_vals in proptest::collection::vec(0..1000u64, 0..=8),
        ) {
            // Invariant: hash_slice(nonzero ‖ zeros, len) equals
            // precompute(len, n_zero_blocks) followed by
            // hash_with_initial_state on the reversed non-zero prefix.
            const WIDTH: usize = 4;
            const RATE: usize = 2;
            const OUT: usize = 2;

            // Non-zero prefix, exact multiple of RATE.
            let nonzero: Vec<u64> = nonzero_vals
                .into_iter()
                .chain(core::iter::repeat(0))
                .take(n_nonzero_blocks * RATE)
                .collect();

            // Full data layout: `[nonzero ‖ zeros]`.
            let mut data: Vec<u64> = nonzero.clone();
            data.extend(core::iter::repeat_n(0u64, n_zero_blocks * RATE));
            let length_iv = data.len() as u64;

            let sponge =
                RtlPaddingFreeSponge::<MockPermutation, WIDTH, RATE, OUT>::new(MockPermutation);

            // Path A: one-shot slice hash.
            let full_output = sponge.hash_slice(&data, length_iv);

            // Path B: precompute the zero suffix midstate, then continue on the reversed prefix.
            let initial_state: [u64; WIDTH] =
                sponge.precompute_zero_suffix_state(length_iv, n_zero_blocks);
            let nonzero_rev: Vec<u64> = nonzero.iter().rev().copied().collect();
            let partial_output = sponge.hash_with_initial_state(&initial_state, nonzero_rev);

            prop_assert_eq!(full_output, partial_output);
        }

        #[test]
        fn proptest_rtl_determinism(
            // Number of `RATE`-sized blocks in the input.
            n_blocks in 0..=6usize,
            vals in proptest::collection::vec(0..1000u64, 12..=12),
        ) {
            const WIDTH: usize = 4;
            const RATE: usize = 2;
            const OUT: usize = 2;

            let data: Vec<u64> = vals.into_iter().take(n_blocks * RATE).collect();
            let length_iv = data.len() as u64;

            let sponge =
                RtlPaddingFreeSponge::<MockPermutation, WIDTH, RATE, OUT>::new(MockPermutation);

            // Same input twice must produce the same output.
            let output_1 = sponge.hash_slice(&data, length_iv);
            let output_2 = sponge.hash_slice(&data, length_iv);
            prop_assert_eq!(output_1, output_2);
        }

        #[test]
        fn proptest_rtl_slice_iter_equivalence(
            n_blocks in 0..=6usize,
            vals in proptest::collection::vec(0..1000u64, 16..=16),
        ) {
            // Invariant: the slice form and the streaming form on the reversed input compute the same digest.
            //
            // - Reduces collision resistance of the streaming form to that of the slice form.
            // - Uses the position-aware mock so equality reflects algorithmic agreement.
            // - A symmetric permutation would let the equality hold by coincidence.
            const WIDTH: usize = 4;
            const RATE: usize = 2;
            const OUT: usize = 2;

            let data: Vec<u64> = vals
                .into_iter()
                .take(n_blocks * RATE)
                .collect();
            let length_iv = data.len() as u64;

            let sponge = RtlPaddingFreeSponge::<PrefixSumPermutation, WIDTH, RATE, OUT>::new(
                PrefixSumPermutation,
            );

            // Both paths use the same initial state `[length_iv, 0, ..., 0]`.
            let mut initial_state = [0u64; WIDTH];
            initial_state[0] = length_iv;

            let slice_digest = sponge.hash_slice(&data, length_iv);
            let iter_digest =
                sponge.hash_with_initial_state(&initial_state, data.iter().rev().copied());

            prop_assert_eq!(slice_digest, iter_digest);
        }
    }
}
