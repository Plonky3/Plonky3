//! The choices one proof makes within a statement it cannot change.

use alloc::vec::Vec;

/// What one proof picks, inside what its statement already allows.
///
/// The only way to build one is to pick heights a statement already allows.
///
/// A height outside the declared range therefore never reaches the prover or the verifier.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Run {
    statement: [u8; 32],
    log_heights: Vec<u32>,
    pow_bits: u32,
}

impl Run {
    /// Record the choices a statement has already accepted.
    pub(super) const fn new(statement: [u8; 32], log_heights: Vec<u32>, pow_bits: u32) -> Self {
        Self {
            statement,
            log_heights,
            pow_bits,
        }
    }

    /// Fingerprint of the statement these choices were accepted by.
    pub(super) const fn statement(&self) -> &[u8; 32] {
        &self.statement
    }

    /// Base-two logarithm of each table's height, in declaration order.
    #[must_use]
    pub fn log_heights(&self) -> &[u32] {
        &self.log_heights
    }

    /// Grinding difficulty every delegated round runs at.
    #[must_use]
    pub const fn pow_bits(&self) -> usize {
        self.pow_bits as usize
    }

    /// Grinding difficulty as the width the fingerprint absorbs it at.
    pub(super) const fn pow_bits_raw(&self) -> u32 {
        self.pow_bits
    }
}
