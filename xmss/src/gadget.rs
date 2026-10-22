//! Signature verification inside a circuit, with a shape fixed by the parameters alone.
//!
//! Every chain takes all `w - 1` steps, and digit-driven selects keep or drop each one.
//!
//! A step's tweak depends only on its slot, so the selects never touch the hash inputs.

use alloc::vec::Vec;

use p3_circuit::{Base, CircuitBuilder, PermutationChip, Unsatisfied, sponge};
use p3_field::{ExtensionField, PrimeCharacteristicRing};

use crate::hash::{
    CHAIN, DIGEST_LEN, LEAF, MESSAGE, MESSAGE_LEN, PARAMETER_LEN, RATE, WIDTH, chain_tweak,
};
use crate::params::BITS_PER_ELEMENT;
use crate::{F, Params, PublicKey, Signature};

/// A public key whose values are circuit values.
#[derive(Clone, Debug)]
pub struct PublicKeyTarget {
    /// The Merkle root.
    pub root: [Base<F>; DIGEST_LEN],
    /// The hash parameter.
    pub parameter: [Base<F>; PARAMETER_LEN],
}

impl PublicKeyTarget {
    /// The key as public inputs of the circuit, root first.
    pub fn public<EF, P>(builder: &mut CircuitBuilder<F, EF, P>, key: &PublicKey) -> Self
    where
        EF: ExtensionField<F>,
        P: PermutationChip<F>,
    {
        Self {
            root: key.root.map(|x| builder.public(x)),
            parameter: key.parameter.map(|x| builder.public(x)),
        }
    }
}

/// Constrain `signature` to verify on `message` at `epoch` under `public_key`.
///
/// The signature enters as prover-chosen values, and the circuit's shape depends only on `params`.
///
/// # Errors
///
/// When the signature does not verify, after every row is emitted.
///
/// # Panics
///
/// When the chip's width is not 16, or the signature does not have the shape `params` gives.
pub fn verify_in_circuit<EF, P>(
    builder: &mut CircuitBuilder<F, EF, P>,
    params: &Params,
    public_key: &PublicKeyTarget,
    epoch: &Base<F>,
    message: &[Base<F>; MESSAGE_LEN],
    signature: &Signature,
) -> Result<(), Unsatisfied>
where
    EF: ExtensionField<F>,
    P: PermutationChip<F>,
{
    assert_eq!(P::WIDTH, WIDTH, "the chip has width 16");
    assert_eq!(
        signature.chains.len(),
        params.num_chains(),
        "one value per chain"
    );
    assert_eq!(
        signature.path.len(),
        params.log_lifetime(),
        "one sibling per level"
    );
    let parameter = &public_key.parameter;

    let randomness: Vec<Base<F>> = signature
        .randomness
        .iter()
        .map(|&x| builder.witness(x))
        .collect();
    let digest = message_digest(builder, parameter, epoch, &randomness, message);
    let digits = digit_bits(builder, params, &digest);

    // The digits, each the weighted sum of its bits, must reach the target.
    let sum: Base<F> = digits
        .iter()
        .flat_map(|bits| {
            bits.iter()
                .enumerate()
                .map(|(i, bit)| bit.clone() * F::from_u32(1 << i))
        })
        .sum();
    let encoded = builder.assert_zero(&(sum - F::from_usize(params.target_sum())));

    let mut ends = Vec::with_capacity(params.num_chains() * DIGEST_LEN);
    for (chain, (bits, value)) in digits.iter().zip(&signature.chains).enumerate() {
        let mut state: Vec<Base<F>> = value.iter().map(|&x| builder.witness(x)).collect();
        // `reached` is one once the slot's start position is at least the digit.
        let mut reached = Base::ZERO;
        for slot in 0..params.chain_len() {
            reached += bits
                .iter()
                .enumerate()
                .map(|(i, bit)| {
                    if (slot >> i) & 1 == 1 {
                        bit.clone()
                    } else {
                        Base::ONE - bit.clone()
                    }
                })
                .product::<Base<F>>();
            let step = chain_step(builder, parameter, epoch, chain, slot + 1, &state);
            state = state
                .into_iter()
                .zip(step)
                .map(|(kept, stepped)| kept.clone() + reached.clone() * (stepped - kept))
                .collect();
        }
        ends.extend(state);
    }

    let prefix = [Base::from(F::from_u32(LEAF)), epoch.clone()];
    let leaf_input: Vec<Base<F>> = prefix
        .into_iter()
        .chain(parameter.iter().cloned())
        .chain(ends)
        .collect();
    let leaf = sponge(&leaf_input, WIDTH, RATE, RATE, |state| {
        builder.permute(state)
    });

    // The epoch's low bits steer the path, and its high bits must be zero.
    let epoch_bits = builder.canonical_bits(epoch);
    let (low, high) = epoch_bits.split_at(params.log_lifetime());
    let in_range = builder.assert_zero(&high.iter().cloned().sum());
    let root = signature
        .path
        .iter()
        .zip(low)
        .fold(leaf, |node, (sibling, bit)| {
            builder.merkle_step(&node, sibling, bit)
        });
    let roots_match: Vec<_> = root
        .iter()
        .zip(&public_key.root)
        .map(|(computed, expected)| builder.assert_zero(&(computed.clone() - expected.clone())))
        .collect();

    encoded
        .and(in_range)
        .and_then(|()| roots_match.into_iter().collect())
}

/// The chain step ending at `position` of `chain`, as [`Hasher::chain_step`](crate::hash::Hasher::chain_step) lays it out.
fn chain_step<EF, P>(
    builder: &mut CircuitBuilder<F, EF, P>,
    parameter: &[Base<F>; PARAMETER_LEN],
    epoch: &Base<F>,
    chain: usize,
    position: usize,
    value: &[Base<F>],
) -> Vec<Base<F>>
where
    EF: ExtensionField<F>,
    P: PermutationChip<F>,
{
    let tweak = [
        epoch.clone(),
        Base::from(chain_tweak(chain, position)),
        Base::from(F::from_u32(CHAIN)),
    ];
    let state: Vec<Base<F>> = value
        .iter()
        .cloned()
        .chain(parameter.iter().cloned())
        .chain(tweak)
        .collect();
    let mut output = builder.permute(&state);
    output.truncate(DIGEST_LEN);
    output
}

/// The message digest, as [`Hasher::message`](crate::hash::Hasher::message) computes it.
fn message_digest<EF, P>(
    builder: &mut CircuitBuilder<F, EF, P>,
    parameter: &[Base<F>; PARAMETER_LEN],
    epoch: &Base<F>,
    randomness: &[Base<F>],
    message: &[Base<F>; MESSAGE_LEN],
) -> Vec<Base<F>>
where
    EF: ExtensionField<F>,
    P: PermutationChip<F>,
{
    let input: Vec<Base<F>> = [Base::from(F::from_u32(MESSAGE)), epoch.clone()]
        .into_iter()
        .chain(parameter.iter().cloned())
        .chain(randomness.iter().cloned())
        .chain(message.iter().cloned())
        .collect();
    sponge(&input, WIDTH, RATE, RATE, |state| builder.permute(state))
}

/// The bits of every digit of a digest, least significant first, as [`Params::digits`] reads them.
fn digit_bits<EF, P>(
    builder: &mut CircuitBuilder<F, EF, P>,
    params: &Params,
    digest: &[Base<F>],
) -> Vec<Vec<Base<F>>>
where
    EF: ExtensionField<F>,
    P: PermutationChip<F>,
{
    let mut digits = Vec::with_capacity(params.num_chains());
    for element in &digest[..params.digit_elements()] {
        let bits = builder.canonical_bits(element);
        digits.extend(
            bits[..BITS_PER_ELEMENT]
                .chunks(params.chunk_bits())
                .map(<[_]>::to_vec),
        );
    }
    digits.truncate(params.num_chains());
    digits
}

#[cfg(test)]
mod tests {
    use p3_field::PrimeField32;
    use p3_field::extension::BinomialExtensionField;
    use proptest::prelude::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::*;
    use crate::hash::Hasher;
    use crate::{Chip, Message, Xmss, chip};

    type Builder = CircuitBuilder<F, BinomialExtensionField<F, 4>, Chip>;

    fn values(builder: &Builder, values: &[Base<F>]) -> Vec<F> {
        values.iter().map(|x| builder.value(x)).collect()
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(8))]

        #[test]
        fn native_and_circuit_verification_agree(
            seed in any::<u64>(),
            epoch in 0u32..4,
            message in prop::array::uniform8(0u32..F::ORDER_U32),
        ) {
            let params = Params::with_lifetime(2);
            let xmss = Xmss::new(params);
            let message: Message = message.map(F::from_u32);
            let (public, secret) = xmss.generate(&mut StdRng::seed_from_u64(seed), 4);
            let signature = xmss.sign(&secret, epoch, &message).unwrap();
            prop_assert_eq!(xmss.verify(&public, epoch, &message, &signature), Ok(()));

            let mut builder = Builder::new(chip());
            let key = PublicKeyTarget::public(&mut builder, &public);
            let epoch_value = builder.public(F::from_u32(epoch));
            let message_values = message.map(|x| builder.public(x));

            // The circuit hashes match the native ones.
            let hasher = Hasher::new();
            let randomness: Vec<Base<F>> =
                signature.randomness.iter().map(|&x| builder.witness(x)).collect();
            let digest = message_digest(&mut builder, &key.parameter, &epoch_value, &randomness, &message_values);
            prop_assert_eq!(
                values(&builder, &digest),
                hasher.message(&public.parameter, epoch, &signature.randomness, &message).to_vec()
            );
            let value: Vec<Base<F>> = signature.chains[3].iter().map(|&x| builder.witness(x)).collect();
            let step = chain_step(&mut builder, &key.parameter, &epoch_value, 3, 2, &value);
            prop_assert_eq!(
                values(&builder, &step),
                hasher.chain_step(&public.parameter, epoch, 3, 2, signature.chains[3]).to_vec()
            );

            prop_assert_eq!(
                verify_in_circuit(&mut builder, &params, &key, &epoch_value, &message_values, &signature),
                Ok(())
            );

            // Another message fails in the circuit as it does natively.
            let mut other = message;
            other[0] += F::ONE;
            let mut builder = Builder::new(chip());
            let key = PublicKeyTarget::public(&mut builder, &public);
            let epoch_value = builder.public(F::from_u32(epoch));
            let other_values = other.map(|x| builder.public(x));
            prop_assert!(xmss.verify(&public, epoch, &other, &signature).is_err());
            prop_assert_eq!(
                verify_in_circuit(&mut builder, &params, &key, &epoch_value, &other_values, &signature),
                Err(Unsatisfied)
            );
        }
    }
}
