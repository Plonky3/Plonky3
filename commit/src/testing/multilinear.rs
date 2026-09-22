//! Shared transcript checks for multilinear commitment backends.

use p3_challenger::CanSample;
use p3_field::ExtensionField;

use crate::MultilinearPcs;

/// How many elements are drawn to compare two transcript states.
///
/// One would already part two sponges that differ. Four keeps a coincidence in a
/// single element from reading as agreement.
const FINGERPRINT_LEN: usize = 4;

/// The next few challenges a transcript yields, which stand for its state.
///
/// Two transcripts in the same state answer the same way; two that drifted do not.
fn fingerprint<Val, Challenger: Clone + CanSample<Val>>(
    challenger: &Challenger,
) -> [Val; FINGERPRINT_LEN] {
    let mut challenger = challenger.clone();
    core::array::from_fn(|_| challenger.sample())
}

/// Check that a backend's commit phase binds exactly what its verifier binds.
///
/// [`MultilinearPcs::commit`] owes the transcript one binding, of the commitment it
/// returns, and the verifier reaches that binding through
/// [`MultilinearPcs::observe_commitment`]. The two sides are interchangeable only if
/// they leave the sponge in the same state, and nothing in the type system says they
/// do. This checks it, on the state itself rather than on the calls made to reach it:
///
/// - committing and observing the returned commitment agree, so neither side absorbs
///   a value the other does not, in an encoding the other does not use, or draws a
///   challenge the other never draws;
/// - observing the commitment twice disagrees with committing once, so a commit phase
///   that binds twice, or one that binds nothing at all, is caught;
/// - committing a different witness disagrees with committing this one, so a binding
///   that does not depend on what was committed is caught.
///
/// The two witnesses are the fixture's job: they must be different enough to commit to
/// different values. Anything a backend does beyond the binding, such as its opening
/// proof and its rejections, belongs in that backend's own tests.
///
/// # Panics
///
/// Panics if the commit phase rejects the fixture, or if any of the checks above fails.
pub fn assert_multilinear_commit_contract<P, Challenge, Challenger>(
    pcs: &P,
    challenger: &Challenger,
    witness: P::Witness,
    other_witness: P::Witness,
) where
    P: MultilinearPcs<Challenge, Challenger>,
    Challenge: ExtensionField<P::Val>,
    Challenger: Clone + CanSample<P::Val>,
{
    let before = fingerprint(challenger);

    let mut prover = challenger.clone();
    let (commitment, _prover_data) = pcs
        .commit(witness, &mut prover)
        .expect("fixture must be within the commit phase's budget");
    let after_commit = fingerprint(&prover);

    assert_ne!(
        after_commit, before,
        "commit must bind its commitment, but it left the transcript untouched"
    );

    let mut verifier = challenger.clone();
    pcs.observe_commitment(&commitment, &mut verifier);
    assert_eq!(
        fingerprint(&verifier),
        after_commit,
        "commit and observe_commitment must leave the same transcript state, \
         or a prover and a verifier that agree on every value still part ways"
    );

    let mut twice = challenger.clone();
    pcs.observe_commitment(&commitment, &mut twice);
    pcs.observe_commitment(&commitment, &mut twice);
    assert_ne!(
        fingerprint(&twice),
        after_commit,
        "commit must bind its commitment exactly once, but binding it twice \
         reaches the same transcript state"
    );

    let mut other = challenger.clone();
    let (_other_commitment, _other_data) = pcs
        .commit(other_witness, &mut other)
        .expect("fixture must be within the commit phase's budget");
    assert_ne!(
        fingerprint(&other),
        after_commit,
        "committing a different witness must reach a different transcript state; \
         either the binding ignores what was committed, or the fixture's two \
         witnesses commit to the same value"
    );
}
