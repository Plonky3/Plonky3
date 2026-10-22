//! Leaf circuits that verify many signatures, for [`p3_circuit::recursion::aggregate`] to fold.

use alloc::vec::Vec;

use p3_circuit::{Circuit, CircuitBuilder, PermutationChip, Unsatisfied};
use p3_field::{ExtensionField, PrimeCharacteristicRing};

use crate::hash::{Poseidon2Hash, TweakableHash};
use crate::{F, Params, PublicKey, PublicKeyTarget, Signature, verify_in_circuit};

/// A signature with the key, epoch and message it verifies on.
#[derive(Clone, Debug)]
pub struct Claim<H: TweakableHash = Poseidon2Hash> {
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
pub struct Statement<H: TweakableHash = Poseidon2Hash> {
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

impl Claim {
    /// The public values a leaf circuit publishes for this claim: root, parameter, epoch, message.
    pub fn public_values(&self) -> Vec<F> {
        let key = &self.public_key;
        key.root
            .into_iter()
            .chain(key.parameter)
            .chain([F::from_u32(self.epoch)])
            .chain(self.message)
            .collect()
    }
}

/// The circuit verifying every claim, publishing each claim's [`Claim::public_values`] in order.
///
/// Its shape, hence its verifying key, depends only on `params` and the number of claims.
///
/// # Errors
///
/// When a signature does not verify.
pub fn leaf_circuit<EF, P>(
    chip: P,
    params: &Params,
    claims: &[Claim],
) -> Result<Circuit<F, EF, P>, Unsatisfied>
where
    EF: ExtensionField<F>,
    P: PermutationChip<F>,
{
    let mut builder = CircuitBuilder::new(chip);
    for claim in claims {
        let key = PublicKeyTarget::public(&mut builder, &claim.public_key);
        let epoch = builder.public(F::from_u32(claim.epoch));
        let message = claim.message.map(|x| builder.public(x));
        verify_in_circuit(
            &mut builder,
            params,
            &key,
            &epoch,
            &message,
            &claim.signature,
        )?;
    }
    Ok(builder.build())
}
