//! The tweakable hash family the scheme is generic over.

use core::fmt::Debug;
use core::hash::Hash;

use rand::CryptoRng;

/// A tweakable hash family: every call is keyed by a public parameter and a tweak naming its role and position (ePrint 2025/055, Section 2).
pub trait TweakableHash: Clone + Debug + Send + Sync {
    /// A hash output: chain values, leaves, tree nodes, message digests and secret seeds.
    type Digest: Copy + Debug + Eq + Hash + Send + Sync;

    /// The public parameter that separates one key's hashes from every other key's.
    type Parameter: Copy + Debug + Eq + Hash + Send + Sync;

    /// The encoding randomness a signature carries.
    type Randomness: Copy + Debug + Eq + Hash + Send + Sync;

    /// A message to sign.
    type Message: Copy + Debug + Eq + Hash + Send + Sync;

    /// Bits of a message digest the digits can read.
    const DIGIT_BITS: usize;

    /// A fresh public parameter and secret seed.
    fn sample<R: CryptoRng + ?Sized>(&self, rng: &mut R) -> (Self::Parameter, Self::Digest);

    /// The secret start of chain `chain` at `epoch`, derived from `seed`.
    fn chain_start(&self, seed: &Self::Digest, epoch: u32, chain: usize) -> Self::Digest;

    /// The encoding randomness of signing attempt `attempt`, derived from `seed`.
    fn randomness(
        &self,
        seed: &Self::Digest,
        epoch: u32,
        attempt: u32,
        message: &Self::Message,
    ) -> Self::Randomness;

    /// The chain step ending at `position` of chain `chain`.
    fn chain_step(
        &self,
        parameter: &Self::Parameter,
        epoch: u32,
        chain: usize,
        position: usize,
        value: Self::Digest,
    ) -> Self::Digest;

    /// The digest whose digits encode `message`.
    fn message(
        &self,
        parameter: &Self::Parameter,
        epoch: u32,
        randomness: &Self::Randomness,
        message: &Self::Message,
    ) -> Self::Digest;

    /// Digit `index` of `chunk_bits` bits of a message digest.
    fn digit(digest: &Self::Digest, chunk_bits: usize, index: usize) -> usize;

    /// The leaf of an epoch, from the ends of its chains.
    fn leaf(&self, parameter: &Self::Parameter, epoch: u32, ends: &[Self::Digest]) -> Self::Digest;

    /// The parent at height `level` and position `index` of two tree nodes.
    fn node(
        &self,
        parameter: &Self::Parameter,
        level: usize,
        index: usize,
        left: Self::Digest,
        right: Self::Digest,
    ) -> Self::Digest;

    /// The root of a subtree of height `level` whose leaves are all inactive.
    fn inactive_root(&self, parameter: &Self::Parameter, level: usize) -> Self::Digest;
}
