//! The batched sponge: one message per lane, on the widest backend the CPU has.
//!
//! x86-64 compiles AVX-512 and AVX2 even when the build does not enable them.
//!
//! It then picks the widest backend the running CPU has, falling back to SSE2.
//!
//! Other targets pick their backend at build time.

use core::fmt;

/// Byte rate of both 256-bit sponges.
///
/// Capacity is twice the 256-bit digest, taken out of the 1600-bit permutation.
/// That leaves 1088 bits, exactly 136 bytes, absorbed per permutation.
pub(crate) const RATE: usize = (1600 - 2 * 256) / 8;

/// Digest length of both 256-bit sponges in bytes.
pub(crate) const DIGEST_BYTES: usize = 32;

/// Index of the state word holding the last byte of the rate.
pub(crate) const LAST_RATE_WORD: usize = (RATE - 1) / 8;

/// Closing padding mark, placed at the last byte of the rate.
const CLOSING_MARK: u64 = 0x80u64 << (8 * ((RATE - 1) % 8));

/// First padding byte of Keccak-256.
///
/// The original submission appends the `pad10*1` rule directly after the message.
/// Its first one bit is the low bit of the byte.
pub(crate) const KECCAK_DOMAIN: u8 = 0x01;

/// First padding byte of SHA3-256.
///
/// FIPS 202 appends the two domain bits `01` before the `pad10*1` rule.
/// Bits enter the byte from the low end, so `0 1 1` reads as `0x06`.
pub(crate) const SHA3_DOMAIN: u8 = 0x06;

/// Every backend this build compiles, widest first.
///
/// The last one runs on every CPU of the target.
const KERNELS: &[Kernel] = &[
    #[cfg(target_arch = "x86_64")]
    crate::avx512::KERNEL,
    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
    crate::avx2::KERNEL,
    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
    crate::sse2::KERNEL,
    #[cfg(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_feature = "sha3"
    ))]
    crate::neon_sha3::KERNEL,
    #[cfg(all(
        target_arch = "aarch64",
        target_feature = "neon",
        not(target_feature = "sha3")
    ))]
    crate::neon::KERNEL,
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    crate::wasm32_simd128::KERNEL,
    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "neon"),
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    crate::fallback::KERNEL,
];

/// Messages the widest compiled backend hashes in one permutation.
///
/// Callers group messages by this count.
///
/// A narrower backend picked at run time only splits a group into several permutations.
pub(crate) const LANES: usize = KERNELS[0].lanes;

/// The batched driver of one backend, picked at run time.
#[derive(Clone, Copy)]
pub(crate) struct Kernel {
    /// The backend's name, for diagnostics.
    name: &'static str,
    /// Messages one permutation advances at once.
    pub(crate) lanes: usize,
    /// Whether the running CPU has the backend's target features.
    supported: fn() -> bool,
    /// The driver compiled for the backend, sound to call only on a CPU that has its features.
    run: unsafe fn(u8, &[u8], usize, &mut [[u8; DIGEST_BYTES]]),
}

impl Kernel {
    /// Describe one backend.
    ///
    /// # Arguments
    ///
    /// - `name`: the backend's name, for diagnostics.
    /// - `lanes`: messages one permutation advances at once.
    /// - `supported`: whether the running CPU has the backend's target features.
    /// - `run`: the driver, sound to call only on such a CPU.
    pub(crate) const fn new(
        name: &'static str,
        lanes: usize,
        supported: fn() -> bool,
        run: unsafe fn(u8, &[u8], usize, &mut [[u8; DIGEST_BYTES]]),
    ) -> Self {
        Self {
            name,
            lanes,
            supported,
            run,
        }
    }

    /// Hash equal-length messages laid end to end, one digest each.
    ///
    /// The caller guarantees that the input holds exactly one message per digest.
    #[inline]
    pub(crate) fn hash_many(
        self,
        domain: u8,
        input: &[u8],
        len: usize,
        out: &mut [[u8; DIGEST_BYTES]],
    ) {
        debug_assert_eq!(input.len(), len * out.len());

        // SAFETY: kernels only leave this module through the filter that checks the CPU.
        unsafe { (self.run)(domain, input, len, out) }
    }
}

impl fmt::Debug for Kernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name)
    }
}

/// Every backend the running CPU supports, widest first.
pub(crate) fn supported() -> impl Iterator<Item = Kernel> {
    KERNELS
        .iter()
        .copied()
        .filter(|kernel| (kernel.supported)())
}

/// The widest backend the running CPU supports.
#[inline]
pub(crate) fn detect() -> Kernel {
    supported()
        .next()
        .expect("the last backend runs on every CPU of the target")
}

/// Hash a batch of equal-length messages on the widest backend the CPU has.
///
/// The domain byte is the first padding byte, which alone tells the two 256-bit hashes apart.
///
/// # Panics
///
/// Panics if the input length is not a whole multiple of the digest count.
#[inline]
pub(crate) fn hash_many(domain: u8, input: &[u8], out: &mut [[u8; DIGEST_BYTES]]) {
    // No digests requested means there is nothing to read from the input.
    if out.is_empty() {
        return;
    }

    // Every message has the same length, so the split is exact by contract.
    assert!(
        input.len().is_multiple_of(out.len()),
        "input length ({}) must be a whole multiple of the digest count ({})",
        input.len(),
        out.len()
    );
    let len = input.len() / out.len();

    detect().hash_many(domain, input, len, out);
}

/// Define the batched driver of a backend, with an optional target feature.
///
/// With a target feature, the driver compiles for it, so every step inlines into one loop.
macro_rules! kernel {
    ($name:literal, $backend:ty, $width:expr $(, $feature:literal)?) => {
        /// Hash equal-length messages, one per lane of this backend.
        ///
        /// # Safety
        ///
        /// The running CPU has this backend's target features.
        $(#[target_feature(enable = $feature)])?
        unsafe fn hash_many(
            domain: u8,
            input: &[u8],
            len: usize,
            out: &mut [[u8; $crate::batch::DIGEST_BYTES]],
        ) {
            // SAFETY: the caller runs this on a CPU with the backend's features.
            unsafe {
                $crate::batch::hash_many_lanes::<$backend, { $width }>(domain, input, len, out)
            };
        }

        /// The batched driver on this backend.
        pub(crate) const KERNEL: $crate::batch::Kernel =
            $crate::batch::Kernel::new($name, $width, supported, hash_many);
    };
}

pub(crate) use kernel;

/// Interleaved Keccak states: word `i` of the state in lane `l` is at `[i][l]`.
///
/// ```text
///     state[word][lane]      word = 0..25, lane = 0..W
///
///     state[0]  [ s_0^(0)  s_0^(1)  ...  s_0^(W-1) ]
///     state[1]  [ s_1^(0)  s_1^(1)  ...  s_1^(W-1) ]
///     ...
/// ```
pub(crate) type State<const W: usize> = [[u64; W]; 25];

/// The steps of the batched sponge that a backend supplies.
///
/// Only the permutation is required.
///
/// A backend may also move messages in and digests out with its own register transposes.
pub(crate) trait Lanes<const W: usize> {
    /// Permute the state of every lane.
    ///
    /// # Safety
    ///
    /// The running CPU has the backend's target features.
    unsafe fn permute(state: &mut State<W>);

    /// Exclusive-or whole message words into the leading state words, one message per lane.
    ///
    /// The slice length is the number of words each lane gives, read from the byte offset on.
    ///
    /// # Safety
    ///
    /// The running CPU has the backend's target features.
    #[inline(always)]
    unsafe fn absorb_words(state: &mut [[u64; W]], lanes: &[&[u8]; W], offset: usize) {
        absorb_words(state, lanes, offset);
    }

    /// Write the digest of every lane.
    ///
    /// # Safety
    ///
    /// The running CPU has the backend's target features.
    #[inline(always)]
    unsafe fn squeeze(state: &State<W>, digests: &mut [[u8; DIGEST_BYTES]; W]) {
        for (lane, digest) in digests.iter_mut().enumerate() {
            *digest = squeeze_lane(state, lane);
        }
    }
}

/// A value stored at the start of a cache line.
///
/// A 64-byte vector load of a state word then never straddles two lines.
#[repr(C, align(64))]
struct CacheAligned<T>(T);

// The kernel helpers below use plain loops, never closures.
//
// A closure inherits the target features of the driver it sits in.
// A library helper, such as the standard array constructor, does not.
// So it cannot inline the closure.
// The call would then stay out of line, returning its vectors through memory.

/// Exclusive-or one whole state word, every lane at once.
///
/// The permutation loads a state word as a single vector, and a partially written word cannot
/// be store-to-load forwarded into that load, so every write here covers the word in full.
#[inline(always)]
fn xor_state_word<const W: usize>(word: &mut [u64; W], value: [u64; W]) {
    for (lane, value) in word.iter_mut().zip(value) {
        *lane ^= value;
    }
}

/// Read the eight message bytes at `offset` in every lane into one state word value.
///
/// Bytes enter the state little-endian, eight to a state word, matching the Keccak convention.
#[inline(always)]
fn gather_word<const W: usize>(lanes: &[&[u8]; W], offset: usize) -> [u64; W] {
    let mut word = [0u64; W];
    for (value, lane) in word.iter_mut().zip(lanes) {
        *value = u64::from_le_bytes(lane[offset..][..8].try_into().unwrap());
    }
    word
}

/// Exclusive-or whole message words into the leading state words, one word at a time.
///
/// Each row is assembled across all lanes first and then stored once, so no state word is
/// built out of partial writes. The messages themselves are never copied or reordered.
#[inline(always)]
pub(crate) fn absorb_words<const W: usize>(
    state: &mut [[u64; W]],
    lanes: &[&[u8]; W],
    offset: usize,
) {
    debug_assert!(state.len() <= RATE / 8);

    for (word_index, word) in state.iter_mut().enumerate() {
        xor_state_word(word, gather_word(lanes, offset + 8 * word_index));
    }
}

/// Absorb the final, shorter block of every lane and close it with the padding.
///
/// The first padding byte `d` carries the domain bits of the hash, then the rule's first one bit:
///
/// ```text
///     [ message bytes | d | 0x00 ... 0x00 | 0x80 ]
///                       ^                   ^
///                  block_len            rate - 1
///
///     Keccak-256:  d = 0x01
///     SHA3-256:    d = 0x06
/// ```
///
/// A block ending one byte short of the rate puts both marks in the same byte.
/// Exclusive-or makes that byte `d | 0x80`, exactly what the rule requires.
///
/// The leftover bytes and the marks meet in the word value before it reaches the state,
/// so the closing word is stored once like every other one.
///
/// # Safety
///
/// The running CPU has the backend's target features.
#[inline(always)]
unsafe fn absorb_final_block<L: Lanes<W>, const W: usize>(
    state: &mut State<W>,
    lanes: &[&[u8]; W],
    domain: u8,
    offset: usize,
    block_len: usize,
) {
    debug_assert!(block_len < RATE);

    let words = block_len / 8;
    let tail = block_len % 8;

    // SAFETY: the caller runs this on a CPU with the backend's features.
    unsafe { L::absorb_words(&mut state[..words], lanes, offset) };

    // The first mark sits immediately after the last message byte, in the same word as any
    // leftover bytes. The second mark joins it when that word is already the closing word of
    // the rate.
    let mut marks = u64::from(domain) << (8 * tail);
    if words == LAST_RATE_WORD {
        marks ^= CLOSING_MARK;
    }
    // At most seven leftover bytes, folded in little-endian order.
    // A byte loop keeps the fold inline, where a slice copy would call out to a library copy.
    let mut value = [marks; W];
    for (value, lane) in value.iter_mut().zip(lanes) {
        for (i, &byte) in lane[offset + 8 * words..][..tail].iter().enumerate() {
            *value ^= u64::from(byte) << (8 * i);
        }
    }
    xor_state_word(&mut state[words], value);

    // The closing word is otherwise untouched by the message and the first mark, so it takes
    // the second mark alone.
    if words != LAST_RATE_WORD {
        xor_state_word(&mut state[LAST_RATE_WORD], [CLOSING_MARK; W]);
    }
}

/// Read the digest of one lane out of a permuted state.
#[inline(always)]
fn squeeze_lane<const W: usize>(state: &State<W>, lane: usize) -> [u8; DIGEST_BYTES] {
    let mut digest = [0u8; DIGEST_BYTES];

    // The digest is the leading bytes of the rate portion, little-endian per state word.
    for (word_index, word) in digest.as_chunks_mut::<8>().0.iter_mut().enumerate() {
        *word = state[word_index][lane].to_le_bytes();
    }

    digest
}

/// Hash equal-length messages on one backend, one message per lane.
///
/// Every step inlines into the backend's driver, so the whole loop compiles for its features.
///
/// The caller guarantees that the input holds exactly one message per digest.
///
/// # Safety
///
/// The running CPU has the backend's target features.
#[inline(always)]
pub(crate) unsafe fn hash_many_lanes<L: Lanes<W>, const W: usize>(
    domain: u8,
    input: &[u8],
    len: usize,
    out: &mut [[u8; DIGEST_BYTES]],
) {
    // SAFETY (every block below): the caller runs this on a CPU with the backend's features.

    // Empty messages all share one digest.
    // One padding-only block in a single state gives it, with no message to split.
    if len == 0 {
        let mut state = CacheAligned([[0u64; W]; 25]);
        unsafe {
            absorb_final_block::<L, W>(&mut state.0, &[&[]; W], domain, 0, 0);
            L::permute(&mut state.0);
        }
        out.fill(squeeze_lane(&state.0, 0));
        return;
    }

    // Whole rate blocks are absorbed in lockstep, then one shorter final block.
    // That final block carries the padding and is empty when the length divides the rate.
    let full_blocks = len / RATE;
    let final_block = len % RATE;

    // One message per lane per permutation.
    for (messages, digests) in input.chunks(len * W).zip(out.chunks_mut(W)) {
        let mut state = CacheAligned([[0u64; W]; 25]);

        // A short last group repeats its first message in the spare lanes, which keeps the
        // lane count fixed at compile time. Those lanes are hashed alongside the requested
        // ones and never squeezed.
        let present = digests.len();
        let mut lanes: [&[u8]; W] = [&[]; W];
        for (lane, slot) in lanes.iter_mut().enumerate() {
            let index = if lane < present { lane } else { 0 };
            *slot = &messages[index * len..][..len];
        }

        // Every lane contributes its block, then one permutation advances all the sponges.
        for block in 0..full_blocks {
            unsafe {
                L::absorb_words(&mut state.0[..RATE / 8], &lanes, block * RATE);
                L::permute(&mut state.0);
            }
        }

        // The final partial block carries the padding and permutes once more.
        unsafe {
            absorb_final_block::<L, W>(
                &mut state.0,
                &lanes,
                domain,
                full_blocks * RATE,
                final_block,
            );
            L::permute(&mut state.0);
        }

        // A full group squeezes every lane at once, a short one only the requested lanes.
        if let Ok(group) = <&mut [[u8; DIGEST_BYTES]; W]>::try_from(&mut *digests) {
            unsafe { L::squeeze(&state.0, group) };
        } else {
            for (lane, digest) in digests.iter_mut().enumerate() {
                *digest = squeeze_lane(&state.0, lane);
            }
        }
    }
}
