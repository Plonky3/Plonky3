//! XMSS-style hash-based signatures over KoalaBear with Poseidon2, and their aggregation.
//!
//! The scheme is the synchronized signature of "Hash-Based Multi-Signatures for Post-Quantum Ethereum" (ePrint 2025/055).
//!
//! Messages are encoded with the target-sum Winternitz encoding of "At the top of the hypercube" (ePrint 2025/889).
//!
//! The same verification runs natively and inside a [`p3_circuit`] circuit, whose proofs aggregate recursively.

#![no_std]

extern crate alloc;

mod aggregate;
mod gadget;
mod hash;
mod keys;
mod params;
mod scheme;
mod signature;

pub use aggregate::{Claim, leaf_circuit};
pub use gadget::{PublicKeyTarget, verify_in_circuit};
pub use hash::{DIGEST_LEN, MESSAGE_LEN, PARAMETER_LEN, RANDOMNESS_LEN};
pub use keys::{PublicKey, SecretKey};
use p3_circuit::Poseidon2Chip;
use p3_koala_bear::{
    GenericPoseidon2LinearLayersKoalaBear, KOALABEAR_POSEIDON2_RC_16_EXTERNAL_FINAL,
    KOALABEAR_POSEIDON2_RC_16_EXTERNAL_INITIAL, KOALABEAR_POSEIDON2_RC_16_INTERNAL, KoalaBear,
    Poseidon2KoalaBear, default_koalabear_poseidon2_16,
};
use p3_poseidon2_air::RoundConstants;
pub use params::Params;
pub use scheme::{InvalidSignature, SignError, Xmss};
pub use signature::Signature;

/// The field every value of the scheme lives in.
type F = KoalaBear;

/// The Poseidon2 permutation of width 16 behind every hash.
type Perm = Poseidon2KoalaBear<16>;

/// A hash output.
pub type Digest = [KoalaBear; DIGEST_LEN];

/// A message to sign.
pub type Message = [KoalaBear; MESSAGE_LEN];

/// The circuit chip proving the permutation the scheme hashes with.
pub type Chip = Poseidon2Chip<F, GenericPoseidon2LinearLayersKoalaBear, Perm, 16, 3, 0, 4, 20>;

/// The chip for the scheme's permutation.
pub fn chip() -> Chip {
    Chip::new(
        default_koalabear_poseidon2_16(),
        RoundConstants::new(
            KOALABEAR_POSEIDON2_RC_16_EXTERNAL_INITIAL,
            KOALABEAR_POSEIDON2_RC_16_INTERNAL,
            KOALABEAR_POSEIDON2_RC_16_EXTERNAL_FINAL,
        ),
    )
}
