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

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_keccak::Keccak256Hash;

    use super::*;
    use crate::contract::error::DeclarationError;
    use crate::contract::machine::MachineDeclaration;
    use crate::contract::table::{ColumnCounts, HeightRange, LocalConstraints, TableDeclaration};

    fn table() -> TableDeclaration {
        TableDeclaration::new(
            ColumnCounts {
                committed: 4,
                preprocessed: 0,
                public: 1,
            },
            LocalConstraints {
                count: 3,
                degree: 2,
            },
            HeightRange::new(2, 16),
        )
    }

    #[test]
    fn the_two_promises_are_distinguishable() {
        assert_eq!(BindingOnly::SECRECY, Secrecy::BindingOnly);
        assert_eq!(Hiding::SECRECY, Secrecy::Hiding);
        assert_ne!(BindingOnly::SECRECY.tag(), Hiding::SECRECY.tag());
    }

    #[test]
    fn the_promise_is_bound_into_the_fingerprint() {
        // The same tables under a different promise are a different statement.
        let binding =
            MachineDeclaration::<BindingOnly, _>::new(Keccak256Hash, vec![table()], 1024).unwrap();
        let hiding =
            MachineDeclaration::<Hiding, _>::new(Keccak256Hash, vec![table()], 1024).unwrap();

        let run = binding.run(&[8], 0).unwrap();
        assert!(binding.run_digest(&run).is_ok());
        assert_eq!(
            hiding.run_digest(&run).unwrap_err(),
            DeclarationError::ForeignRun
        );
    }
}
