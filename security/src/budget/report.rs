//! Labeled soundness breakdown produced by [`super::security_report`].
//!
//! Fiat-Shamir security of the compiled argument is the minimum over its rounds: an adversary
//! needs only one round to break. Keeping the rounds as named terms rather than collapsing them
//! into a single number means the *binding* round stays visible, which is what tells a reader
//! whether a parameter change helped the round that actually mattered.

/// Label for the lookup / permutation-argument bus-challenge round.
pub const LOOKUP_LABEL: &str = "lookup-challenge";
/// Label for the constraint-batching round.
pub const COMPOSITION_LABEL: &str = "constraint-composition";
/// Label for the out-of-domain evaluation round.
pub const OUT_OF_DOMAIN_LABEL: &str = "out-of-domain";
/// Label for the DEEP-quotient batching round.
pub const DEEP_COMPOSITION_LABEL: &str = "deep-composition";
/// Label for the FRI commit-phase folding rounds.
pub const FOLDING_LABEL: &str = "fri-folding";
/// Label for the FRI query round.
pub const QUERY_LABEL: &str = "fri-query";
/// Label for the commitment-collision cap.
pub const COLLISION_LABEL: &str = "commitment-collision";

/// Number of rounds a [`SecurityReport`] carries.
pub const NUM_TERMS: usize = 7;

/// One round's contribution, as `-log2(error)` in fixed point.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SecurityTerm {
    /// Which round this bounds.
    pub label: &'static str,
    /// Attained bits for the round, grinding included, in fixed point.
    pub bits: u64,
}

impl SecurityTerm {
    /// Builds a term from a label and its fixed-point bit count.
    pub const fn new(label: &'static str, bits: u64) -> Self {
        Self { label, bits }
    }
}

/// The per-round soundness breakdown of one proof configuration.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SecurityReport {
    terms: [SecurityTerm; NUM_TERMS],
}

impl SecurityReport {
    /// Builds a report from its rounds.
    pub const fn new(terms: [SecurityTerm; NUM_TERMS]) -> Self {
        Self { terms }
    }

    /// Every round.
    pub const fn terms(&self) -> &[SecurityTerm; NUM_TERMS] {
        &self.terms
    }

    /// The round that binds — the one attaining the minimum.
    ///
    /// Ties resolve to the earliest such round, so the reported bottleneck is the one an
    /// adversary reaches first.
    pub const fn binding_term(&self) -> SecurityTerm {
        let mut binding = self.terms[0];
        let mut index = 1;
        while index < NUM_TERMS {
            if self.terms[index].bits < binding.bits {
                binding = self.terms[index];
            }
            index += 1;
        }
        binding
    }

    /// Attained conjectured security, in fixed point.
    pub const fn attained(&self) -> u64 {
        self.binding_term().bits
    }

    /// Attained conjectured security in whole bits, rounded down.
    pub const fn security_level(&self) -> u32 {
        crate::fixed::to_bits(self.attained())
    }
}
