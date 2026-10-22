//! Public keys and signatures, and their verification (FIPS 205, Sections 9 and 10).

use alloc::boxed::Box;

use crate::params::{N, PUBLIC_KEY_BYTES, SIGNATURE_BYTES};
use crate::verify::root_from_signature;

/// Why a signature is rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum SignatureError {
    /// The signature does not have the parameter set's length.
    #[error("a signature has {SIGNATURE_BYTES} bytes, not {0}")]
    Length(usize),
    /// The context string is longer than 255 bytes.
    #[error("a context has at most 255 bytes, not {0}")]
    Context(usize),
    /// The signature does not verify.
    #[error("invalid signature")]
    Invalid,
}

/// A public key `PK.seed || PK.root` (FIPS 205, Section 9.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PublicKey {
    /// The public seed, which keys every hash.
    pub seed: [u8; N],
    /// The root of the top XMSS tree.
    pub root: [u8; N],
}

/// A signature `R || SIG_FORS || SIG_HT` (FIPS 205, Section 9.2).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Signature(Box<[u8; SIGNATURE_BYTES]>);

impl Signature {
    /// The signature's bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8; SIGNATURE_BYTES] {
        &self.0
    }
}

impl TryFrom<&[u8]> for Signature {
    type Error = SignatureError;

    fn try_from(bytes: &[u8]) -> Result<Self, SignatureError> {
        let bytes: [u8; SIGNATURE_BYTES] = bytes
            .try_into()
            .map_err(|_| SignatureError::Length(bytes.len()))?;
        Ok(Self(Box::new(bytes)))
    }
}

impl PublicKey {
    /// The key encoded as `PK.seed || PK.root`.
    #[must_use]
    pub fn from_bytes(bytes: &[u8; PUBLIC_KEY_BYTES]) -> Self {
        let (seed, root) = bytes.split_at(N);
        Self {
            seed: seed.try_into().expect("n bytes"),
            root: root.try_into().expect("n bytes"),
        }
    }

    /// Check a signature of `message` under `context` (FIPS 205, Algorithm 24).
    ///
    /// # Errors
    ///
    /// When the context is longer than 255 bytes or the signature does not verify.
    pub fn verify(
        &self,
        message: &[u8],
        context: &[u8],
        signature: &Signature,
    ) -> Result<(), SignatureError> {
        let length =
            u8::try_from(context.len()).map_err(|_| SignatureError::Context(context.len()))?;
        self.verify_parts(&[&[0, length], context, message], signature)
    }

    /// Check a signature of the raw message `message` (FIPS 205, Algorithm 20).
    ///
    /// # Errors
    ///
    /// When the signature does not verify.
    pub fn verify_internal(
        &self,
        message: &[u8],
        signature: &Signature,
    ) -> Result<(), SignatureError> {
        self.verify_parts(&[message], signature)
    }

    /// Check a signature of the concatenated parts.
    fn verify_parts(&self, message: &[&[u8]], signature: &Signature) -> Result<(), SignatureError> {
        let root = root_from_signature(&self.seed, &self.root, message, signature.as_bytes());
        if root == self.root {
            Ok(())
        } else {
            Err(SignatureError::Invalid)
        }
    }
}
