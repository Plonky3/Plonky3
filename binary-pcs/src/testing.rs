//! Shared transcript checks for the Boolean commitment backends.
//!
//! Same obligation as the multilinear commit contract in the commitment crate.
//!
//! The witness is packed bits rather than field elements.

use core::array;
use core::fmt::Debug;

use p3_binary_field::{PackedGf2, Underlier};
use p3_challenger::CanSample;

use crate::BooleanMultilinearPcs;

/// How many elements are drawn to compare two transcript states.
///
/// One element already separates two sponges that differ.
///
/// Four keeps a single-element coincidence from reading as agreement.
const FINGERPRINT_LEN: usize = 4;

/// The next few challenges a transcript yields, which stand for its state.
///
/// Two transcripts in the same state answer the same way.
///
/// Two that drifted do not.
fn fingerprint<Val, Challenger: Clone + CanSample<Val>>(
    challenger: &Challenger,
) -> [Val; FINGERPRINT_LEN] {
    let mut challenger = challenger.clone();
    array::from_fn(|_| challenger.sample())
}

/// Check that a Boolean backend's commit phase binds exactly what its verifier binds.
///
/// The prover binds the commitment while committing.
///
/// The verifier binds it by observing the commitment.
///
/// Both must leave the sponge in the same state, and the type system does not enforce it.
///
/// Checks:
/// - Committing changes the transcript.
/// - Committing and observing the commitment reach the same state.
/// - Committing different bits reaches a different state.
///
/// Agreement is pinned, not a count: two sides that both bind twice still agree.
///
/// The two witnesses must differ in at least one bit.
///
/// # Panics
///
/// Panics if the commit phase rejects a witness, or if any check fails.
pub fn assert_boolean_multilinear_commit_contract<P, U, EF, Challenger>(
    pcs: &P,
    challenger: &Challenger,
    bits: &[PackedGf2<U>],
    other_bits: &[PackedGf2<U>],
) where
    P: BooleanMultilinearPcs<EF, Challenger>,
    U: Underlier,
    EF: Debug + PartialEq,
    Challenger: Clone + CanSample<EF>,
{
    // Transcript state before anything is bound.
    let before = fingerprint(challenger);

    // Prover side: commit the bits on a fresh copy of the transcript.
    let mut prover = challenger.clone();
    let (commitment, _prover_data) = pcs
        .commit_bits(bits, &mut prover)
        .expect("fixture must cover exactly the committed hypercube");
    let after_commit = fingerprint(&prover);

    assert_ne!(
        after_commit, before,
        "commit_bits must bind its commitment, but it left the transcript untouched"
    );

    // Verifier side: replay only the commitment on another fresh copy.
    let mut verifier = challenger.clone();
    pcs.observe_commitment(&commitment, &mut verifier);
    assert_eq!(
        fingerprint(&verifier),
        after_commit,
        "commit_bits and observe_commitment must leave the same transcript state, \
         or a prover and a verifier that agree on every value still part ways"
    );

    // A different witness must lead to a different commitment, hence a different state.
    let mut other = challenger.clone();
    let (_other_commitment, _other_data) = pcs
        .commit_bits(other_bits, &mut other)
        .expect("fixture must cover exactly the committed hypercube");
    assert_ne!(
        fingerprint(&other),
        after_commit,
        "committing different bits must reach a different transcript state; \
         either the binding ignores what was committed, or the fixture's two \
         witnesses hold the same bits"
    );
}
