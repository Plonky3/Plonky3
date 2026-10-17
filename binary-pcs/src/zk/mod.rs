//! Experimental private current-row AIR and a polynomial-size reference PCS.
//!
//! Enabled only by `experimental-zk`, independently of `BinaryPcs` and its
//! non-hiding multilinear paths. [`ReferenceZkAir::compile`] admits
//! `Air<CurrentRowAirBuilder>`: generic `AirBuilder<F = Val>` AIRs work, while
//! concrete `SymbolicAirBuilder` implementations must adapt. Explicitly declare
//! no main next-row columns. Only current-row main variables, public values,
//! constants and arithmetic are supported. Successor access, preprocessing,
//! periodic columns, row selectors, public boundaries, extension constraints,
//! permutation arguments, lookups and buses are unsupported. Boolean column
//! metadata adds explicit Booleanity constraints or rejects if inconsistent.
//!
//! The verifier owns the public fixed shape and canonical symbolic constraints.
//! On the Cantor trace subspace H, interpolate each column T0 and mask it as
//! T=T0+rW with independent uniform r and W=product_{a in H}(X-a). Commit T
//! and each separate unsplit quotient Q_i=C_i(T,p)/W before sampling one z
//! outside H in the interactive model. At any fixed such z the trace claims
//! T_j(z) are jointly uniform; quotient claims are determined by these claims
//! and the public statement. See
//! [`ReferenceZkAir`] for the additional committed-degree algebraic bound.
//!
//! The native `zk_air` example uses an OS-seeded private CSPRNG and verifies
//! a serialized proof round-trip. This reference sends full coefficient blends
//! and uses O(n*L) Horner encoding per polynomial; optimized hiding BaseFold is
//! future work. The existing clear BaseFold reductions are not used here.
//!
//! For message length M and q queries, append kappa=q+1 independent uniform
//! monomial coefficients. Thus h=f+X^M R has L=M+kappa coefficients. Horner
//! evaluation encodes h on a shifted Cantor domain of n=next_power_of_two(4L)
//! nonzero points. Keccak256 salted Merkle commitments use one independent
//! full-field salt per leaf. This is not the novel-basis additive RS encoder.
//!
//! Before gamma is sampled, commit an independent uniform L-coefficient u and
//! disclose sigma=lambda(u_low). Disclose all g=gamma*h+u before sampling q
//! indices uniformly with replacement. Check lambda(g_low)=gamma*v+sigma and
//! authenticate h/u at each index. Each column has independent tails and u;
//! only gamma and query positions are shared. Duplicate queries repeat the same
//! authenticated row and contribute no additional information or rank.
//! Configuration, commitment, point and claims are bound before the u/sigma
//! commitment. The q+1 tail masks account for observations jointly conditioned
//! on the full blend and sigma. Private masks never come from the separate
//! public Keccak challenger.
//!
//! In the interactive oracle model, put e=floor((n-L)/4). The error is at most
//! 1/(2^128-1) + (1-(e+1)/n)^q: outside the unique exceptional gamma, any g
//! satisfying a false low-coefficient claim differs from gamma*h+u at >e
//! rows. Treating columns as a vector-valued RS code gives the same row minimum
//! distance n-L+1. Two distinct gamma with at most e bad rows each would decode
//! the same h matrix and all its low-coefficient claims, so the exceptional-gamma
//! term does not require a union bound over columns when every column is checked.
//! In particular q=4*(bits+2) suffices for a query contribution
//! below 2^-(bits+2). This excludes hash security and Fiat–Shamir compilation
//! losses. No noninteractive random-oracle simulation/compilation theorem is
//! claimed; salt/hash simulator programming and grinding losses are unanalysed.
//! This is an experimental reference protocol, not a production security
//! certification. Commitments must only be opened once, as enforced by the
//! consumed, non-Clone private state. Seed the caller's CSPRNG unpredictably.

use alloc::vec::Vec;

use p3_binary_field::{BinaryChallenger, BinaryField128, TowerLevel};
use p3_challenger::{CanObserve, HashChallenger};
use p3_keccak::Keccak256Hash;

mod pcs;
pub use pcs::*;
mod air;
mod builder;
mod polynomial;
mod wire;
pub use air::*;
pub use builder::CurrentRowAirBuilder;

/// Fixed witness and challenge field.
pub type Val = BinaryField128;
/// Independent Keccak transcript for this experimental protocol.
pub type Challenger = BinaryChallenger<Val, HashChallenger<u8, Keccak256Hash, 32>>;

/// Start a fresh experimental transcript.
pub const fn challenger() -> Challenger {
    Challenger::from_hasher(Vec::new(), Keccak256Hash)
}

/// Errors from the experimental reference protocol.
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
    #[error("serialization failed")]
    Serialization,
}

pub(super) fn observe_usize(challenger: &mut Challenger, value: usize) {
    challenger.observe(Val::from_repr(value as u128));
}

pub(super) fn observe_bytes(challenger: &mut Challenger, bytes: &[u8]) {
    observe_usize(challenger, bytes.len());
    for chunk in bytes.chunks(16) {
        let mut repr = [0; 16];
        repr[..chunk.len()].copy_from_slice(chunk);
        challenger.observe(Val::from_repr(u128::from_le_bytes(repr)));
    }
}
