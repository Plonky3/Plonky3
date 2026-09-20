//! What a commitment promises about the data behind it.
//!
//! Every scheme in this repository binds without hiding, so its proofs are not zero-knowledge.
//!
//! The binary path is one of them: its final codeword travels in the clear.

use serde::{Deserialize, Serialize};

/// The promise a commitment makes about the data behind it.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum Secrecy {
    /// Opening reveals part of the committed data.
    BindingOnly,
    /// Opening reveals nothing beyond the claim.
    Hiding,
}

impl Secrecy {
    /// The byte this promise contributes to a statement digest.
    #[must_use]
    pub const fn tag(self) -> u8 {
        match self {
            Self::BindingOnly => 0,
            Self::Hiding => 1,
        }
    }
}

mod sealed {
    pub trait Sealed {}

    impl Sealed for super::BindingOnly {}
    impl Sealed for super::Hiding {}
}

/// A commitment promise fixed at compile time, sealed to the two markers below.
pub trait SecrecyLevel: sealed::Sealed {
    /// The promise this marker stands for.
    const SECRECY: Secrecy;
}

/// Binds its data without hiding it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BindingOnly;

/// Binds its data and hides it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Hiding;

impl SecrecyLevel for BindingOnly {
    const SECRECY: Secrecy = Secrecy::BindingOnly;
}

impl SecrecyLevel for Hiding {
    const SECRECY: Secrecy = Secrecy::Hiding;
}
