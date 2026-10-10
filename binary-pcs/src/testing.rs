//! Shared transcript checks for the Boolean commitment backends.
//!
//! `p3-commit`'s `assert_multilinear_commit_contract` covers `MultilinearPcs`. The Boolean
//! trait is a sibling rather than a special case of it: it commits bits rather than field
//! elements, so the fixture is packed bits and the call is [`BooleanMultilinearPcs::commit_bits`].
//! The obligation it checks is the same one, which is why this reads like that helper.

use p3_binary_field::{PackedGf2, Underlier};
use p3_challenger::CanSample;

use crate::BooleanMultilinearPcs;

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

/// Check that a Boolean backend's commit phase binds exactly what its verifier binds.
///
/// [`BooleanMultilinearPcs::commit_bits`] owes the transcript one binding, of the commitment
/// it returns, and the verifier reaches that binding through
/// [`BooleanMultilinearPcs::observe_commitment`]. The two sides are interchangeable only if
/// they leave the sponge in the same state, and nothing in the type system says they do.
///
/// - committing binds something, so a commit phase that leaves the transcript untouched is
///   caught;
/// - committing and observing the returned commitment agree, so neither side absorbs a value
///   the other does not, in an encoding the other does not use, or draws a challenge the other
///   never draws. It is agreement that is pinned, not a count: a commit phase and an
///   `observe_commitment` that both bind twice still agree, and this helper is not the place
///   that would object;
/// - committing different bits disagrees with committing these, so a binding that does not
///   depend on what was committed is caught.
///
/// The two witnesses are the fixture's job: they must differ in at least one bit of the
/// committed hypercube. Anything a backend does beyond the binding belongs in that backend's
/// own tests.
///
/// # Panics
///
/// Panics if the commit phase rejects the fixture, or if any of the checks above fails.
pub fn assert_boolean_multilinear_commit_contract<P, U, EF, Challenger>(
    pcs: &P,
    challenger: &Challenger,
    bits: &[PackedGf2<U>],
    other_bits: &[PackedGf2<U>],
) where
    P: BooleanMultilinearPcs<EF, Challenger>,
    U: Underlier,
    EF: core::fmt::Debug + PartialEq,
    Challenger: Clone + CanSample<EF>,
{
    let before = fingerprint(challenger);

    let mut prover = challenger.clone();
    let (commitment, _prover_data) = pcs
        .commit_bits(bits, &mut prover)
        .expect("fixture must cover exactly the committed hypercube");
    let after_commit = fingerprint(&prover);

    assert_ne!(
        after_commit, before,
        "commit_bits must bind its commitment, but it left the transcript untouched"
    );

    let mut verifier = challenger.clone();
    pcs.observe_commitment(&commitment, &mut verifier);
    assert_eq!(
        fingerprint(&verifier),
        after_commit,
        "commit_bits and observe_commitment must leave the same transcript state, \
         or a prover and a verifier that agree on every value still part ways"
    );

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
