//! The BLAKE3 hash of a byte string of fixed length, on wires (BLAKE3 specification, Sections 2.4 to 2.6).
//!
//! A chunk of 1024 bytes chains its 64-byte blocks; chunks merge pairwise into parents, the left subtree the largest power of two of chunks.

use alloc::vec::Vec;

use p3_blake3::Blake3;
use p3_symmetric::CryptographicHasher;

use super::gates::Byte;
use crate::circuit::{Builder, Wire, Word};
use crate::gadgets::blake3::compress;

/// The BLAKE3 IV, the first eight SHA-256 constants (BLAKE3 specification, Section 2.2).
const IV: [u32; 8] = [
    0x6A09_E667,
    0xBB67_AE85,
    0x3C6E_F372,
    0xA54F_F53A,
    0x510E_527F,
    0x9B05_688C,
    0x1F83_D9AB,
    0x5BE0_CD19,
];

/// Bytes of a chunk.
const CHUNK_LEN: usize = 1024;

/// Bytes of a block.
const BLOCK_LEN: usize = 64;

/// Flag of a chunk's first block.
const CHUNK_START: u32 = 1;

/// Flag of a chunk's last block.
const CHUNK_END: u32 = 2;

/// Flag of a parent node.
const PARENT: u32 = 4;

/// Flag of the root node.
const ROOT: u32 = 8;

/// The digest of `input`, its wires and native value.
pub(crate) fn hash(c: &mut Builder, input: &[Byte]) -> [Byte; 32] {
    let native: Vec<u8> = input.iter().map(|b| b.value).collect();
    let digest = Blake3.hash_slice(&native);
    let chunks: Vec<&[Byte]> = if input.is_empty() {
        alloc::vec![input]
    } else {
        input.chunks(CHUNK_LEN).collect()
    };
    let cv = subtree(c, &chunks, 0, true);
    let wires: Vec<Wire> = cv.iter().flat_map(|w| w.iter().copied()).collect();
    core::array::from_fn(|i| Byte {
        wires: core::array::from_fn(|b| wires[8 * i + b]),
        value: digest[i],
    })
}

/// The chaining value of chunks `first..first + chunks.len()`, the root's when `root` holds.
fn subtree(c: &mut Builder, chunks: &[&[Byte]], first: u64, root: bool) -> [Word; 8] {
    if let [chunk] = chunks {
        return chunk_cv(c, chunk, first, root);
    }
    let left = 1usize << (usize::BITS - 1 - (chunks.len() - 1).leading_zeros());
    let l = subtree(c, &chunks[..left], first, false);
    let r = subtree(c, &chunks[left..], first + left as u64, false);
    let block: [Word; 16] = core::array::from_fn(|i| {
        if i < 8 {
            l[i].clone()
        } else {
            r[i - 8].clone()
        }
    });
    let flags = PARENT | if root { ROOT } else { 0 };
    let iv = iv(c);
    let tail = tail(c, 0, BLOCK_LEN as u32, flags);
    let out = compress(c, &iv, &block, &tail);
    core::array::from_fn(|i| out[i].clone())
}

/// The chaining value of one chunk at index `counter`.
fn chunk_cv(c: &mut Builder, chunk: &[Byte], counter: u64, root: bool) -> [Word; 8] {
    let blocks: Vec<&[Byte]> = if chunk.is_empty() {
        alloc::vec![chunk]
    } else {
        chunk.chunks(BLOCK_LEN).collect()
    };
    let mut cv = iv(c);
    for (j, block) in blocks.iter().enumerate() {
        let mut flags = 0;
        if j == 0 {
            flags |= CHUNK_START;
        }
        if j + 1 == blocks.len() {
            flags |= CHUNK_END | if root { ROOT } else { 0 };
        }
        let words: [Word; 16] = core::array::from_fn(|w| {
            (0..32)
                .map(|bit| {
                    block
                        .get(4 * w + bit / 8)
                        .map_or(Wire::ZERO, |byte| byte.wires[bit % 8])
                })
                .collect()
        });
        let tail = tail(c, counter, block.len() as u32, flags);
        let out = compress(c, &cv, &words, &tail);
        cv = core::array::from_fn(|i| out[i].clone());
    }
    cv
}

/// The IV as constant words.
fn iv(c: &Builder) -> [Word; 8] {
    core::array::from_fn(|i| c.literal(u64::from(IV[i]), 32))
}

/// The counter's halves, the block length and the flags, as constant words.
fn tail(c: &Builder, counter: u64, block_len: u32, flags: u32) -> [Word; 4] {
    [
        c.literal(counter & 0xFFFF_FFFF, 32),
        c.literal(counter >> 32, 32),
        c.literal(u64::from(block_len), 32),
        c.literal(u64::from(flags), 32),
    ]
}
