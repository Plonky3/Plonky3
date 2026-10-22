//! A binary-friendly tweakable hash: keyed BLAKE3 truncated to 128-bit digests.
//!
//! Every call is one BLAKE3 keyed hash whose key is the 16-byte parameter, then a 16-byte tweak.
//!
//! A tweak is four little-endian words: a domain tag, then the epoch or tree level, then positions.
//!
//! Key derivation keys BLAKE3 with the seed instead of the parameter.
//!
//! Inactive subtrees are the zero digest at every height and are never hashed, so node tweaks can name their position.

use rand::{CryptoRng, RngExt};

use crate::hash::TweakableHash;

/// Bytes of a digest, a parameter and a seed.
pub(crate) const DIGEST_BYTES: usize = 16;

/// Bytes of a signature's encoding randomness.
pub(crate) const RANDOMNESS_BYTES: usize = 16;

/// Bytes of a message.
pub(crate) const MESSAGE_BYTES: usize = 32;

/// Domain tag of chain steps.
pub(crate) const CHAIN: u32 = 1;

/// Domain tag of the message hash.
pub(crate) const MESSAGE: u32 = 2;

/// Domain tag of the leaf hash.
pub(crate) const LEAF: u32 = 3;

/// Domain tag of chain-start derivation.
const CHAIN_START: u32 = 4;

/// Domain tag of randomness derivation.
const RANDOMNESS: u32 = 5;

/// Domain tag of tree nodes.
pub(crate) const NODE: u32 = 6;

/// A 128-bit digest.
type Digest = [u8; DIGEST_BYTES];

/// The tweakable hash of keyed BLAKE3, with 128-bit digests.
///
/// Every hash targets 128 bits of classical security, and Grover halves that.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Blake3Hash;

/// The first 16 bytes of `BLAKE3(key = prefix || tweak, input)`.
fn keyed(prefix: &[u8; 16], tweak: [u32; 4], input: &[u8]) -> Digest {
    let mut key = [0; 32];
    key[..16].copy_from_slice(prefix);
    for (bytes, word) in key[16..].as_chunks_mut::<4>().0.iter_mut().zip(tweak) {
        bytes.copy_from_slice(&word.to_le_bytes());
    }
    let hash = blake3::keyed_hash(&key, input);
    core::array::from_fn(|i| hash.as_bytes()[i])
}

impl TweakableHash for Blake3Hash {
    type Digest = Digest;
    type Parameter = [u8; DIGEST_BYTES];
    type Randomness = [u8; RANDOMNESS_BYTES];
    type Message = [u8; MESSAGE_BYTES];

    const DIGIT_BITS: usize = 8 * DIGEST_BYTES;

    fn sample<R: CryptoRng + ?Sized>(&self, rng: &mut R) -> (Self::Parameter, Digest) {
        (rng.random(), rng.random())
    }

    fn chain_start(&self, seed: &Digest, epoch: u32, chain: usize) -> Digest {
        keyed(seed, [CHAIN_START, epoch, chain as u32, 0], &[])
    }

    fn randomness(
        &self,
        seed: &Digest,
        epoch: u32,
        attempt: u32,
        message: &[u8; MESSAGE_BYTES],
    ) -> [u8; RANDOMNESS_BYTES] {
        keyed(seed, [RANDOMNESS, epoch, attempt, 0], message)
    }

    fn chain_step(
        &self,
        parameter: &[u8; DIGEST_BYTES],
        epoch: u32,
        chain: usize,
        position: usize,
        value: Digest,
    ) -> Digest {
        keyed(
            parameter,
            [CHAIN, epoch, chain as u32, position as u32],
            &value,
        )
    }

    /// The hash of `randomness || message`.
    fn message(
        &self,
        parameter: &[u8; DIGEST_BYTES],
        epoch: u32,
        randomness: &[u8; RANDOMNESS_BYTES],
        message: &[u8; MESSAGE_BYTES],
    ) -> Digest {
        let mut input = [0; RANDOMNESS_BYTES + MESSAGE_BYTES];
        input[..RANDOMNESS_BYTES].copy_from_slice(randomness);
        input[RANDOMNESS_BYTES..].copy_from_slice(message);
        keyed(parameter, [MESSAGE, epoch, 0, 0], &input)
    }

    /// The digit from bits `chunk_bits * index` on of the little-endian digest.
    fn digit(digest: &Digest, chunk_bits: usize, index: usize) -> usize {
        ((u128::from_le_bytes(*digest) >> (chunk_bits * index)) & ((1 << chunk_bits) - 1)) as usize
    }

    fn leaf(&self, parameter: &[u8; DIGEST_BYTES], epoch: u32, ends: &[Digest]) -> Digest {
        keyed(parameter, [LEAF, epoch, 0, 0], ends.as_flattened())
    }

    /// The hash of `left || right`.
    fn node(
        &self,
        parameter: &[u8; DIGEST_BYTES],
        level: usize,
        index: usize,
        left: Digest,
        right: Digest,
    ) -> Digest {
        keyed(
            parameter,
            [NODE, level as u32, index as u32, 0],
            [left, right].as_flattened(),
        )
    }

    fn inactive_root(&self, _parameter: &[u8; DIGEST_BYTES], _level: usize) -> Digest {
        [0; DIGEST_BYTES]
    }
}
