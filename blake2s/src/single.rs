//! Hashing one message on the general-purpose registers.
//!
//! One message is a single chain: every block needs the previous one's output.
//!
//! So its speed is set by the latency of that chain, not by how many operations issue at once.
//!
//! The four G functions of each step are independent, which gives the scalar units four chains to overlap.
//!
//! Left alone, the compiler packs those four G into one vector register on some targets.
//!
//! That puts lane shuffles and extracts on the chain, so on x86-64 this code keeps the state scalar.
//!
//! AArch64 keeps the state scalar without help.

use crate::DIGEST_BYTES;
use crate::params::{IV, PARAM_BLOCK_0, SIGMA};

/// Bytes in one compression block.
const BLOCK_BYTES: usize = 64;

/// Chaining value of one message.
type State = [u32; 8];

/// The chaining value every message starts from: the IV with the parameter block folded in.
const INITIAL_STATE: State = {
    let mut h = IV;
    h[0] ^= PARAM_BLOCK_0;
    h
};

/// Hash one contiguous message.
#[inline]
pub(crate) fn hash(message: &[u8]) -> [u8; DIGEST_BYTES] {
    // Every block but the last is compressed plainly.
    //
    // The last one holds 1 to 64 bytes, or none at all for the empty message, zero padded.
    //
    //     130 bytes:  [ 64 | 64 | 2 ]    two plain blocks, then a final block of 2 bytes
    //     128 bytes:  [ 64 | 64 ]        one plain block, then a final block of 64 bytes
    let plain = message.len().saturating_sub(1) / BLOCK_BYTES;
    let (head, tail) = message.split_at(plain * BLOCK_BYTES);
    let mut last = [0u8; BLOCK_BYTES];
    last[..tail.len()].copy_from_slice(tail);

    let mut h = INITIAL_STATE;
    blocks(&mut h, head.as_chunks::<BLOCK_BYTES>().0, 0);
    compress(&mut h, &last, message.len() as u64, true);
    digest(&h)
}

/// A message fed a piece at a time.
///
/// It holds back one block, since the final block is compressed with its own flag.
#[derive(Clone)]
pub(crate) struct Hasher {
    /// The chaining value so far.
    h: State,
    /// Bytes compressed so far.
    t: u64,
    /// Bytes not compressed yet, at most one block.
    buf: [u8; BLOCK_BYTES],
    /// How many bytes of the buffer are filled.
    len: usize,
}

impl Hasher {
    /// An empty message.
    #[inline]
    pub(crate) const fn new() -> Self {
        Self {
            h: INITIAL_STATE,
            t: 0,
            buf: [0; BLOCK_BYTES],
            len: 0,
        }
    }

    /// Append bytes to the message.
    pub(crate) fn update(&mut self, mut input: &[u8]) {
        // A partial buffer is topped up first.
        //
        // It is compressed only once more input follows, since it may be the final block.
        if self.len > 0 {
            let take = (BLOCK_BYTES - self.len).min(input.len());
            self.buf[self.len..][..take].copy_from_slice(&input[..take]);
            self.len += take;
            input = &input[take..];
            if input.is_empty() {
                return;
            }
            self.t += BLOCK_BYTES as u64;
            compress(&mut self.h, &self.buf, self.t, false);
            self.len = 0;
        }
        if input.is_empty() {
            return;
        }

        // Whole blocks straight from the input, keeping back the one that may end the message.
        let plain = (input.len() - 1) / BLOCK_BYTES;
        let (head, tail) = input.split_at(plain * BLOCK_BYTES);
        blocks(&mut self.h, head.as_chunks::<BLOCK_BYTES>().0, self.t);
        self.t += head.len() as u64;
        self.buf[..tail.len()].copy_from_slice(tail);
        self.len = tail.len();
    }

    /// The digest of the whole message.
    pub(crate) fn finalize(mut self) -> [u8; DIGEST_BYTES] {
        // The held-back bytes, zero padded, are the final block.
        self.buf[self.len..].fill(0);
        compress(&mut self.h, &self.buf, self.t + self.len as u64, true);
        digest(&self.h)
    }
}

/// Compress whole blocks in order, where `t` bytes came before the first.
#[inline]
fn blocks(h: &mut State, blocks: &[[u8; BLOCK_BYTES]], mut t: u64) {
    // The counter is the byte count up to and including each block.
    for block in blocks {
        t += BLOCK_BYTES as u64;
        compress(h, block, t, false);
    }
}

/// Advance the chaining value by one block.
///
/// `t` is the byte counter, and `last` sets the final-block flag.
#[inline]
fn compress(h: &mut State, block: &[u8; BLOCK_BYTES], t: u64, last: bool) {
    let (words, _) = block.as_chunks::<4>();
    let m: [u32; 16] = core::array::from_fn(|w| u32::from_le_bytes(words[w]));

    // The chaining value, then the IV with the counter and the final-block flag folded in.
    let mut v = [0u32; 16];
    v[..8].copy_from_slice(h);
    v[8..].copy_from_slice(&IV);
    v[12] ^= t as u32;
    v[13] ^= (t >> 32) as u32;
    if last {
        v[14] = !v[14];
    }

    // Ten literal rounds, so every schedule index is a constant.
    round::<0>(&mut v, &m);
    round::<1>(&mut v, &m);
    round::<2>(&mut v, &m);
    round::<3>(&mut v, &m);
    round::<4>(&mut v, &m);
    round::<5>(&mut v, &m);
    round::<6>(&mut v, &m);
    round::<7>(&mut v, &m);
    round::<8>(&mut v, &m);
    round::<9>(&mut v, &m);

    // Feed-forward: h'[i] = h[i] ^ v[i] ^ v[i + 8].
    for i in 0..8 {
        h[i] ^= v[i] ^ v[i + 8];
    }
}

/// One round: four column mixes, then four diagonal mixes.
#[inline(always)]
fn round<const R: usize>(v: &mut [u32; 16], m: &[u32; 16]) {
    let s = const { SIGMA[R] };
    g(v, [0, 4, 8, 12], m[s[0]], m[s[1]]);
    g(v, [1, 5, 9, 13], m[s[2]], m[s[3]]);
    g(v, [2, 6, 10, 14], m[s[4]], m[s[5]]);
    g(v, [3, 7, 11, 15], m[s[6]], m[s[7]]);
    g(v, [0, 5, 10, 15], m[s[8]], m[s[9]]);
    g(v, [1, 6, 11, 12], m[s[10]], m[s[11]]);
    g(v, [2, 7, 8, 13], m[s[12]], m[s[13]]);
    g(v, [3, 4, 9, 14], m[s[14]], m[s[15]]);
}

/// The mixing function G on one column or diagonal, with the rotation distances 16, 12, 8 and 7.
#[inline(always)]
fn g(v: &mut [u32; 16], [a, b, c, d]: [usize; 4], x: u32, y: u32) {
    // The message word is added first, so that addition sits off the dependency chain.
    v[a] = v[a].wrapping_add(x).wrapping_add(v[b]);
    v[d] = rotr::<16>(v[d] ^ v[a]);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = rotr::<12>(v[b] ^ v[c]);
    v[a] = v[a].wrapping_add(y).wrapping_add(v[b]);
    v[d] = rotr::<8>(v[d] ^ v[a]);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = rotr::<7>(v[b] ^ v[c]);

    // The four G of a step are alike, and a vectoriser would pack them into one register.
    //
    // Keeping each G's words in scalar registers makes that packing cost more than it saves.
    pin(v, [a, b, c, d]);
}

/// Rotate a word right by `N` bits.
///
/// Some x86-64 tunings lower a rotate to a double shift, which is several times slower on current cores.
///
/// Without BMI2, one `ror` is written out; with it, the compiler already emits the flag-free `rorx`.
#[inline(always)]
fn rotr<const N: u32>(x: u32) -> u32 {
    #[cfg(all(target_arch = "x86_64", not(target_feature = "bmi2"), not(miri)))]
    {
        let mut x = x;
        // SAFETY: one register rotate, which touches no memory and no stack.
        unsafe {
            core::arch::asm!("ror {0:e}, {n}", inout(reg) x, n = const N, options(pure, nomem, nostack));
        }
        x
    }
    #[cfg(not(all(target_arch = "x86_64", not(target_feature = "bmi2"), not(miri))))]
    x.rotate_right(N)
}

/// Keep four state words in general-purpose registers at this point.
///
/// The empty assembly emits no instruction.
///
/// It only stops the vectoriser from moving the state into vector registers.
///
/// Only x86-64 needs it, since no AArch64 core measured gains from it.
///
/// Miri cannot run assembly, so it takes the plain path.
#[inline(always)]
#[cfg_attr(
    not(all(target_arch = "x86_64", not(miri))),
    allow(clippy::missing_const_for_fn)
)]
fn pin(v: &mut [u32; 16], [a, b, c, d]: [usize; 4]) {
    // SAFETY: the assembly is empty and names only these four registers.
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    unsafe {
        core::arch::asm!(
            "/* {0:e} {1:e} {2:e} {3:e} */",
            inout(reg) v[a],
            inout(reg) v[b],
            inout(reg) v[c],
            inout(reg) v[d],
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    #[cfg(not(all(target_arch = "x86_64", not(miri))))]
    let _ = (v, a, b, c, d);
}

/// The 32 little-endian bytes of a chaining value.
#[inline]
fn digest(h: &State) -> [u8; DIGEST_BYTES] {
    let mut out = [0u8; DIGEST_BYTES];
    for (bytes, word) in out.as_chunks_mut::<4>().0.iter_mut().zip(h) {
        *bytes = word.to_le_bytes();
    }
    out
}
