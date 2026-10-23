//! Signatures with the key, epoch and message they verify on.

use crate::hash::TweakableHash;
use crate::{PublicKey, Signature};

/// A signature with the key, epoch and message it verifies on.
#[derive(Clone, Debug)]
pub struct Claim<H: TweakableHash> {
    /// The signer's key.
    pub public_key: PublicKey<H::Digest, H::Parameter>,
    /// The epoch signed at.
    pub epoch: u32,
    /// The signed message.
    pub message: H::Message,
    /// The signature.
    pub signature: Signature<H::Digest, H::Randomness>,
}

/// What a claim states without its signature: the key, the epoch and the message.
#[derive(Clone, Debug)]
pub struct Statement<H: TweakableHash> {
    /// The signer's key.
    pub public_key: PublicKey<H::Digest, H::Parameter>,
    /// The epoch signed at.
    pub epoch: u32,
    /// The signed message.
    pub message: H::Message,
}

impl<H: TweakableHash> Claim<H> {
    /// The claim without its signature.
    pub const fn statement(&self) -> Statement<H> {
        Statement {
            public_key: self.public_key,
            epoch: self.epoch,
            message: self.message,
        }
    }
}
