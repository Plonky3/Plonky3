//! Private current-row AIR and a polynomial-size reference PCS.
//!
//! Always available, including without default features, independently of `BinaryPcs` and its non-hiding multilinear paths.
//! [`ReferenceZkAir::compile`] admits `Air<CurrentRowAirBuilder>`: generic `AirBuilder<F = Val>` AIRs work, while concrete `SymbolicAirBuilder` implementations must adapt.
//! Explicitly declare no main next-row columns.
//! Only current-row main variables, public values, constants and arithmetic are supported.
//! Successor access, preprocessing, periodic columns, row selectors, public boundaries, extension constraints, permutation arguments, lookups and buses are unsupported.
//! Boolean column metadata adds explicit Booleanity constraints or rejects if inconsistent.
//!
//! The verifier owns the public fixed shape and canonical symbolic constraints.
//! On the Cantor trace subspace H, interpolate each column T0 and mask it as T=T0+rW with independent uniform r and W=product_{a in H}(X-a).
//! Commit T and each separate unsplit quotient Q_i=C_i(T,p)/W before sampling one z outside H in the interactive model.
//! At any fixed such z the trace claims T_j(z) are jointly uniform; quotient claims are determined by these claims and the public statement.
//! See [`ReferenceZkAir`] for the additional committed-degree algebraic bound.
//!
//! The native `zk_air` example uses an OS-seeded private CSPRNG and verifies a serialized proof round-trip.
//! This reference sends full coefficient blends and uses sparse basis conversion plus additive NTT encoding; optimized hiding BaseFold is future work.
//! The existing clear BaseFold reductions are not used here.
//!
//! For message length M and q queries, append kappa=q+1 independent uniform monomial coefficients.
//! Thus h=f+X^M R has L=M+kappa coefficients.
//! A scratch copy is converted to the Cantor novel basis and evaluated by additive NTT on a shifted Cantor domain of n=next_power_of_two(4L) nonzero points.
//! Keccak256 salted Merkle commitments use two independent full-field salts per leaf.
//! Proof coefficients remain in monomial order; this does not reuse the clear BaseFold PCS.
//!
//! Before gamma is sampled, commit an independent uniform L-coefficient u and disclose sigma=lambda(u_low).
//! Disclose all g=gamma*h+u before sampling q indices uniformly with replacement.
//! Check lambda(g_low)=gamma*v+sigma and authenticate h/u at each index.
//! Each column has independent tails and u; only gamma and query positions are shared.
//! Duplicate queries repeat the same authenticated row and contribute no additional information or rank.
//! Configuration, commitment, point and claims are bound before the u/sigma commitment.
//! The q+1 tail masks account for observations jointly conditioned on the full blend and sigma.
//! Private masks never come from the separate public Keccak challenger.
//!
//! [`ZkAirConfig::interactive`] budgets only interactive algebra/query error.
//! [`ZkAirConfig::classical_rom`] additionally checks explicit adversary RO-call and GLOBAL lifetime salted-leaf budgets.
//! [`ReferenceZkAir::security_report`] exposes the checked public geometry and component bounds.
//! See [`security`] for the full derived classical-ROM RBR/knowledge, graph and joint privacy arguments, exact canonical bytes and modeled RO/RNG assumptions.
//! Concrete Keccak/CSPRNG security is assumed; real randomness adds `Delta_rng`.
//!
//! Commitments permit one opening only, enforced by consumed non-Clone private state.
//! Fresh private RNG state is required across proofs: never clone, reseed, reset or replay it, including rollback.
//! Single-use state does not enforce caller RNG freshness.
//! Reusing h and u under distinct gamma reveals every low message coefficient.
//! This reference protocol is not a security certification; its modeled assumptions are documented in [`security`].
//!
//! Protocol v2 separates leaf/node/Fiat–Shamir Keccak domains and retains the complete framed public prefix.
//! Each z/gamma uses one fixed 16-candidate draw (typed abort on exhaustion); all q indices use one geometry-bound draw.
//! Canonical transport starts with P3ZKAIR2 and rejects the earlier protocol.

use p3_binary_field::BinaryField128;

mod pcs;
pub use pcs::*;
mod air;
mod builder;
mod polynomial;
pub mod security;
pub use security::{ClassicalRomBudget, ClassicalRomReport, SecurityReport};
mod wire;
pub use air::*;
pub use builder::CurrentRowAirBuilder;

/// Fixed witness and challenge field.
pub type Val = BinaryField128;
mod transcript;
pub use transcript::{Challenger, challenger};

/// Errors from the reference protocol.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ZkError {
    #[error("invalid configuration: {0}")]
    InvalidConfig(&'static str),
    #[error("invalid shape: {0}")]
    InvalidShape(&'static str),
    #[error("unsupported AIR: {0}")]
    UnsupportedAir(&'static str),
    #[error("invalid witness: {0}")]
    InvalidWitness(&'static str),
    #[error("invalid opening: {0}")]
    InvalidOpening(&'static str),
    #[error("all 16 public challenge candidates rejected: {0}")]
    ChallengeExhausted(&'static str),
    #[error("serialization failed")]
    Serialization,
}

pub(super) fn observe_usize(challenger: &mut Challenger, value: usize) {
    challenger.observe_usize(value);
}

pub(super) fn observe_bytes(challenger: &mut Challenger, bytes: &[u8]) {
    challenger.observe_bytes(bytes);
}

#[cfg(test)]
mod v2_tests {
    use p3_binary_field::TowerLevel;
    use p3_challenger::{CanObserve, CanSample};
    use p3_keccak::Keccak256Hash;
    use p3_symmetric::CryptographicHasher;

    use super::*;

    #[test]
    fn transcript_matches_independently_framed_v2_draw() {
        let mut transcript = challenger();
        transcript.observe(Val::from_repr(7));
        let mut prefix = alloc::vec![1]; // field observation record
        prefix.extend_from_slice(&7u128.to_le_bytes());
        let label = b"generic-field";
        let mut input = b"p3-binary-zk-v2/fiat-shamir\0".to_vec();
        input.extend_from_slice(&(label.len() as u64).to_le_bytes());
        input.extend_from_slice(label);
        input.extend_from_slice(&(prefix.len() as u64).to_le_bytes());
        input.extend_from_slice(&prefix);
        input.extend_from_slice(&0u64.to_le_bytes());
        input.extend_from_slice(&16u64.to_le_bytes());
        input.extend_from_slice(&0u32.to_le_bytes());
        let expected = Keccak256Hash.hash_slice(&input);
        assert_eq!(
            CanSample::<Val>::sample(&mut transcript)
                .to_repr()
                .to_le_bytes(),
            expected[..16]
        );
    }
}
