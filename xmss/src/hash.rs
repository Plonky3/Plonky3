//! The tweakable hashes of the scheme, all built on one Poseidon2 permutation of width 16.
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

use p3_field::PrimeCharacteristicRing;
use p3_koala_bear::default_koalabear_poseidon2_16;
use p3_symmetric::{
    CryptographicHasher, PaddingFreeSponge, PseudoCompressionFunction, TruncatedPermutation,
};

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

/// The scheme's hashes, natively.
#[derive(Clone, Debug)]
pub(crate) struct Hasher {
    sponge: PaddingFreeSponge<Perm, WIDTH, RATE, RATE>,
    compress: TruncatedPermutation<Perm, 2, DIGEST_LEN, WIDTH>,
}

impl Hasher {
    /// The hashes over the standard KoalaBear Poseidon2 permutation.
    pub(crate) fn new() -> Self {
        let perm = default_koalabear_poseidon2_16();
        Self {
            sponge: PaddingFreeSponge::new(perm.clone()),
            compress: TruncatedPermutation::new(perm),
        }
    }

    /// The chain step ending at `position` of chain `chain`.
    pub(crate) fn chain_step(
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

    /// The digest whose digits encode `message`.
    pub(crate) fn message(
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

    /// The leaf of an epoch, from the ends of its chains.
    pub(crate) fn leaf(
        &self,
        parameter: &[F; PARAMETER_LEN],
        epoch: u32,
        ends: &[Digest],
    ) -> Digest {
        let prefix = [F::from_u32(LEAF), F::from_u32(epoch)];
        self.sponge.hash_iter(
            prefix
                .into_iter()
                .chain(*parameter)
                .chain(ends.iter().flatten().copied()),
        )
    }

    /// The pseudorandom output of `seed` on a domain tag and its inputs.
    pub(crate) fn derive(
        &self,
        seed: &Digest,
        tag: u32,
        inputs: impl IntoIterator<Item = F>,
    ) -> Digest {
        self.sponge.hash_iter(
            core::iter::once(F::from_u32(tag))
                .chain(*seed)
                .chain(inputs),
        )
    }

    /// The parent of two tree nodes.
    pub(crate) fn node(&self, left: Digest, right: Digest) -> Digest {
        self.compress.compress([left, right])
    }
}
