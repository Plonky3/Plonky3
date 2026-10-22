//! The tweakable hashes the scheme is generic over, and their Poseidon2 instance over KoalaBear.
//!
//! The Poseidon2 instance builds every hash on one permutation of width 16.
//!
//! A chain step compresses `value || parameter, epoch, tweak, CHAIN` with a truncated permutation, so `CHAIN` sits in state element 15.
//!
//! The tweak of the step ending at `position` of chain `i` is `256 * i + position`.
//!
//! The message, leaf and key-derivation hashes are padding-free sponges of rate 8 whose first input is a domain tag.
//!
//! A sponge's first call has element 15 zero, so it never meets a chain step.
//!
//! Tree nodes compress `left || right` untweaked, the layout a circuit Merkle step proves.

use core::fmt::Debug;
use core::hash::Hash;

use p3_field::{PrimeCharacteristicRing, PrimeField32};
use p3_koala_bear::default_koalabear_poseidon2_16;
use p3_symmetric::{
    CryptographicHasher, PaddingFreeSponge, PseudoCompressionFunction, TruncatedPermutation,
};
use rand::{CryptoRng, RngExt};

use crate::params::BITS_PER_ELEMENT;
use crate::{Digest, F, Message, Perm};

/// Field elements per digest, about 248 bits.
pub const DIGEST_LEN: usize = 8;

/// Field elements of a public key's hash parameter.
pub const PARAMETER_LEN: usize = 5;

/// Field elements of a signature's encoding randomness.
pub const RANDOMNESS_LEN: usize = 7;

/// Field elements per message.
pub const MESSAGE_LEN: usize = 8;

/// The permutation's state width.
pub(crate) const WIDTH: usize = 16;

/// The sponges' rate and output length.
pub(crate) const RATE: usize = 8;

/// Domain tag of chain steps.
pub(crate) const CHAIN: u32 = 1;

/// Domain tag of the message hash.
pub(crate) const MESSAGE: u32 = 2;

/// Domain tag of the leaf hash.
pub(crate) const LEAF: u32 = 3;

/// Domain tag of chain-start derivation.
pub(crate) const CHAIN_START: u32 = 4;

/// Domain tag of randomness derivation.
pub(crate) const RANDOMNESS: u32 = 5;

/// The tweak of the chain step that ends at `position` of chain `chain`.
pub(crate) fn chain_tweak(chain: usize, position: usize) -> F {
    F::from_usize((chain << 8) | position)
}

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

/// The tweakable hashes over the standard KoalaBear Poseidon2 permutation.
#[derive(Clone, Debug)]
pub struct Poseidon2Hash {
    sponge: PaddingFreeSponge<Perm, WIDTH, RATE, RATE>,
    compress: TruncatedPermutation<Perm, 2, DIGEST_LEN, WIDTH>,
}

impl Poseidon2Hash {
    /// The hashes over the standard KoalaBear Poseidon2 permutation.
    pub fn new() -> Self {
        let perm = default_koalabear_poseidon2_16();
        Self {
            sponge: PaddingFreeSponge::new(perm.clone()),
            compress: TruncatedPermutation::new(perm),
        }
    }

    /// The pseudorandom output of `seed` on a domain tag and its inputs.
    fn derive(&self, seed: &Digest, tag: u32, inputs: impl IntoIterator<Item = F>) -> Digest {
        self.sponge.hash_iter(
            core::iter::once(F::from_u32(tag))
                .chain(*seed)
                .chain(inputs),
        )
    }
}

impl Default for Poseidon2Hash {
    fn default() -> Self {
        Self::new()
    }
}

impl TweakableHash for Poseidon2Hash {
    type Digest = Digest;
    type Parameter = [F; PARAMETER_LEN];
    type Randomness = [F; RANDOMNESS_LEN];
    type Message = Message;

    const DIGIT_BITS: usize = DIGEST_LEN * BITS_PER_ELEMENT;

    fn sample<R: CryptoRng + ?Sized>(&self, rng: &mut R) -> (Self::Parameter, Digest) {
        (rng.random(), rng.random())
    }

    fn chain_start(&self, seed: &Digest, epoch: u32, chain: usize) -> Digest {
        self.derive(
            seed,
            CHAIN_START,
            [F::from_u32(epoch), F::from_usize(chain)],
        )
    }

    fn randomness(
        &self,
        seed: &Digest,
        epoch: u32,
        attempt: u32,
        message: &Message,
    ) -> [F; RANDOMNESS_LEN] {
        let inputs = [F::from_u32(epoch), F::from_u32(attempt)];
        let derived = self.derive(seed, RANDOMNESS, inputs.into_iter().chain(*message));
        core::array::from_fn(|i| derived[i])
    }

    fn chain_step(
        &self,
        parameter: &[F; PARAMETER_LEN],
        epoch: u32,
        chain: usize,
        position: usize,
        value: Digest,
    ) -> Digest {
        let mut tweak = [F::ZERO; DIGEST_LEN];
        tweak[..PARAMETER_LEN].copy_from_slice(parameter);
        tweak[PARAMETER_LEN..].copy_from_slice(&[
            F::from_u32(epoch),
            chain_tweak(chain, position),
            F::from_u32(CHAIN),
        ]);
        self.compress.compress([value, tweak])
    }

    fn message(
        &self,
        parameter: &[F; PARAMETER_LEN],
        epoch: u32,
        randomness: &[F; RANDOMNESS_LEN],
        message: &Message,
    ) -> Digest {
        let prefix = [F::from_u32(MESSAGE), F::from_u32(epoch)];
        self.sponge.hash_iter(
            prefix
                .into_iter()
                .chain(*parameter)
                .chain(*randomness)
                .chain(*message),
        )
    }

    /// The digit from the low bits of its digest element, least significant first.
    fn digit(digest: &Digest, chunk_bits: usize, index: usize) -> usize {
        let per_element = BITS_PER_ELEMENT / chunk_bits;
        let element = digest[index / per_element].as_canonical_u32();
        ((element >> ((index % per_element) * chunk_bits)) & ((1 << chunk_bits) - 1)) as usize
    }

    fn leaf(&self, parameter: &[F; PARAMETER_LEN], epoch: u32, ends: &[Digest]) -> Digest {
        let prefix = [F::from_u32(LEAF), F::from_u32(epoch)];
        self.sponge.hash_iter(
            prefix
                .into_iter()
                .chain(*parameter)
                .chain(ends.iter().flatten().copied()),
        )
    }

    /// The untweaked compression of `left || right`.
    fn node(
        &self,
        _parameter: &[F; PARAMETER_LEN],
        _level: usize,
        _index: usize,
        left: Digest,
        right: Digest,
    ) -> Digest {
        self.compress.compress([left, right])
    }

    /// The zero digest hashed up `level` times.
    fn inactive_root(&self, parameter: &[F; PARAMETER_LEN], level: usize) -> Digest {
        (0..level).fold([F::ZERO; DIGEST_LEN], |node, l| {
            self.node(parameter, l + 1, 0, node, node)
        })
    }
}
