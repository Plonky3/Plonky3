//! Key generation, signing and verification.

use alloc::vec::Vec;

use p3_field::PrimeCharacteristicRing;
use p3_maybe_rayon::prelude::*;
use rand::{CryptoRng, RngExt};

use crate::hash::{CHAIN_START, Hasher, PARAMETER_LEN, RANDOMNESS, RANDOMNESS_LEN};
use crate::keys::Tree;
use crate::{Digest, F, Message, Params, PublicKey, SecretKey, Signature};

/// Signing attempts before the encoding gives up, far above the expected `1 / P(target)`.
const MAX_ATTEMPTS: u32 = 1 << 16;

/// Why a message could not be signed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum SignError {
    /// The epoch is outside the key's active epochs.
    #[error("epoch {0} is not active for this key")]
    InactiveEpoch(u32),
    /// No randomness within the attempt budget made the digits reach the target sum.
    #[error("no encoding reached the target sum")]
    EncodingExhausted,
}

/// A signature that does not verify.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("invalid signature")]
pub struct InvalidSignature;

/// The scheme for one set of parameters.
#[derive(Clone, Debug)]
pub struct Xmss {
    params: Params,
    hasher: Hasher,
}

impl Xmss {
    /// The scheme with `params`, hashing with the standard KoalaBear Poseidon2 permutation.
    pub fn new(params: Params) -> Self {
        Self {
            params,
            hasher: Hasher::new(),
        }
    }

    /// The parameters.
    pub const fn params(&self) -> &Params {
        &self.params
    }

    /// A key pair whose first `active_epochs` epochs can sign.
    ///
    /// The other leaves are inactive, so a long lifetime costs only its active part.
    ///
    /// # Panics
    ///
    /// When `active_epochs` is zero or exceeds the lifetime.
    pub fn generate<R: CryptoRng + ?Sized>(
        &self,
        rng: &mut R,
        active_epochs: usize,
    ) -> (PublicKey, SecretKey) {
        let parameter: [F; PARAMETER_LEN] = rng.random();
        let seed: Digest = rng.random();
        let leaves = (0..active_epochs as u32)
            .into_par_iter()
            .map(|epoch| {
                let ends: Vec<Digest> = (0..self.params.num_chains())
                    .map(|chain| {
                        let start = self.chain_start(&seed, epoch, chain);
                        self.walk(&parameter, epoch, chain, 0, self.params.chain_len(), start)
                    })
                    .collect();
                self.hasher.leaf(&parameter, epoch, &ends)
            })
            .collect();
        let tree = Tree::new(&self.hasher, leaves, self.params.log_lifetime());
        let public = PublicKey {
            root: tree.root(),
            parameter,
        };
        let secret = SecretKey {
            params: self.params,
            parameter,
            seed,
            tree,
        };
        (public, secret)
    }

    /// The signature of `message` at `epoch`.
    ///
    /// Each epoch must sign at most one message, since a second one reveals more of every chain.
    ///
    /// Signing is deterministic: the randomness derives from the seed, the epoch and the message.
    ///
    /// # Errors
    ///
    /// When the epoch is not active, or no randomness reaches the target sum.
    ///
    /// # Panics
    ///
    /// When the key was generated for other parameters.
    pub fn sign(
        &self,
        secret: &SecretKey,
        epoch: u32,
        message: &Message,
    ) -> Result<Signature, SignError> {
        assert_eq!(secret.params, self.params, "the key has these parameters");
        if epoch as usize >= secret.active_epochs() {
            return Err(SignError::InactiveEpoch(epoch));
        }
        let (randomness, digits) = (0..MAX_ATTEMPTS)
            .find_map(|attempt| {
                let inputs = [F::from_u32(epoch), F::from_u32(attempt)];
                let derived = self.hasher.derive(
                    &secret.seed,
                    RANDOMNESS,
                    inputs.into_iter().chain(*message),
                );
                let randomness: [F; RANDOMNESS_LEN] = core::array::from_fn(|i| derived[i]);
                self.encode(&secret.parameter, epoch, &randomness, message)
                    .map(|digits| (randomness, digits))
            })
            .ok_or(SignError::EncodingExhausted)?;
        let chains = digits
            .into_iter()
            .enumerate()
            .map(|(chain, digit)| {
                let start = self.chain_start(&secret.seed, epoch, chain);
                self.walk(&secret.parameter, epoch, chain, 0, digit, start)
            })
            .collect();
        Ok(Signature {
            randomness,
            chains,
            path: secret.tree.path(epoch as usize),
        })
    }

    /// Check `signature` on `message` at `epoch` under `public`.
    ///
    /// # Errors
    ///
    /// When the signature does not verify.
    pub fn verify(
        &self,
        public: &PublicKey,
        epoch: u32,
        message: &Message,
        signature: &Signature,
    ) -> Result<(), InvalidSignature> {
        let params = &self.params;
        if epoch >> params.log_lifetime() != 0
            || signature.chains.len() != params.num_chains()
            || signature.path.len() != params.log_lifetime()
        {
            return Err(InvalidSignature);
        }
        let digits = self
            .encode(&public.parameter, epoch, &signature.randomness, message)
            .ok_or(InvalidSignature)?;
        let ends: Vec<Digest> = digits
            .into_iter()
            .zip(&signature.chains)
            .enumerate()
            .map(|(chain, (digit, &value))| {
                self.walk(
                    &public.parameter,
                    epoch,
                    chain,
                    digit,
                    params.chain_len(),
                    value,
                )
            })
            .collect();
        let leaf = self.hasher.leaf(&public.parameter, epoch, &ends);
        let root = signature
            .path
            .iter()
            .enumerate()
            .fold(leaf, |node, (level, &sibling)| {
                if (epoch >> level) & 1 == 1 {
                    self.hasher.node(sibling, node)
                } else {
                    self.hasher.node(node, sibling)
                }
            });
        if root == public.root {
            Ok(())
        } else {
            Err(InvalidSignature)
        }
    }

    /// The digits of `message` under `randomness`, if they reach the target sum.
    fn encode(
        &self,
        parameter: &[F; PARAMETER_LEN],
        epoch: u32,
        randomness: &[F; RANDOMNESS_LEN],
        message: &Message,
    ) -> Option<Vec<usize>> {
        let digest = self.hasher.message(parameter, epoch, randomness, message);
        let digits: Vec<usize> = self.params.digits(&digest).collect();
        (digits.iter().sum::<usize>() == self.params.target_sum()).then_some(digits)
    }

    /// The secret start of a chain.
    fn chain_start(&self, seed: &Digest, epoch: u32, chain: usize) -> Digest {
        let inputs = [F::from_u32(epoch), F::from_usize(chain)];
        self.hasher.derive(seed, CHAIN_START, inputs)
    }

    /// The chain value at position `to`, from its value at position `from`.
    fn walk(
        &self,
        parameter: &[F; PARAMETER_LEN],
        epoch: u32,
        chain: usize,
        from: usize,
        to: usize,
        value: Digest,
    ) -> Digest {
        (from + 1..=to).fold(value, |value, position| {
            self.hasher
                .chain_step(parameter, epoch, chain, position, value)
        })
    }
}
