use alloc::vec::Vec;

use p3_challenger::FieldChallenger;
use p3_challenger::fs::TranscriptField;
use p3_field::{Dup, ExtensionField, PrimeCharacteristicRing};
use p3_multilinear_util::point::Point;
use p3_sumcheck::generic_degree::RoundPolyInterpolator;
use p3_verifier_arith::{Lift, Native};
use thiserror::Error;

use super::transcript::{FractionGkrShape, FractionGkrVerifierTranscript, ROUND_POLY_LEN};
use super::{FractionGkrOutput, FractionGkrProof, SplitFraction};

/// Malformed fractional-GKR proofs rejected by the verifier.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum FractionGkrError {
    /// The reduction was asked to run over a table with no variables at all.
    #[error("fraction GKR requires at least one input variable")]
    InvalidVariableCount,
    /// The proof carries a layer count the variable count never describes.
    #[error("fraction GKR proof has {actual} layers, expected {expected}")]
    InvalidLayerCount {
        /// Layer count the run was described with.
        expected: usize,
        /// Layer count the proof carries.
        actual: usize,
    },
    /// One layer carries a round count its depth in the tree never describes.
    #[error("fraction GKR layer {layer} has {actual} sumcheck rounds, expected {expected}")]
    InvalidRoundCount {
        /// Position of the layer, counting from the root.
        layer: usize,
        /// Round count that layer was described with.
        expected: usize,
        /// Round count the proof carries.
        actual: usize,
    },
    /// The root fraction has a zero denominator, so it denotes nothing.
    #[error("fraction GKR root denominator is zero")]
    ZeroRootDenominator,
    /// One layer's sumcheck did not close on the two child fractions it sent.
    #[error("fraction GKR layer {layer} failed its gate consistency check")]
    LayerConsistency {
        /// Position of the layer, counting from the root.
        layer: usize,
    },
}

/// Verify the internal consistency of a zero-sum fractional-GKR reduction.
///
/// The returned numerator and denominator are claims, not authenticated values.
/// Nothing here ties them to the committed input polynomials.
///
/// Completing the zero-sum statement is the caller's job.
/// It must check that the two values really are the openings of the input tables at the point.
///
/// Every count the transcript needs comes from the variable count, never from the proof.
/// The proof's own counts are checked against it before the transcript exists.
///
/// # Arguments
///
/// - `proof`: the reduction messages, layer by layer.
/// - `num_variables`: variable count of the padded fraction tables.
/// - `challenger`: sponge of the surrounding protocol, borrowed for the run.
///
/// # Errors
///
/// - The variable count is zero, so there is nothing to reduce.
/// - The proof's layer count or a layer's round count disagrees with the described shape.
/// - The root denominator is zero.
/// - A layer's sumcheck does not close on the children that layer sent.
pub fn verify_fractional_gkr<F, EF, Challenger>(
    proof: &FractionGkrProof<EF>,
    num_variables: usize,
    challenger: &mut Challenger,
) -> Result<FractionGkrOutput<EF>, FractionGkrError>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F>,
{
    verify_fractional_gkr_in(&mut Native::new(), proof, num_variables, challenger)
}

/// As [`verify_fractional_gkr`], within an arithmetic.
///
/// # Errors
///
/// As [`verify_fractional_gkr`].
pub fn verify_fractional_gkr_in<A, Challenger>(
    arith: &mut A,
    proof: &FractionGkrProof<A::Challenge>,
    num_variables: usize,
    challenger: &mut Challenger,
) -> Result<FractionGkrOutput<A::EF>, FractionGkrError>
where
    A: Lift<Val: TranscriptField>,
    Challenger: FieldChallenger<A::Val>,
{
    if num_variables < 1 {
        return Err(FractionGkrError::InvalidVariableCount);
    }

    // Every length the description depends on is settled before the driver exists.
    //
    // The driver panics on a step it was not described with, and a verifier must not panic.
    // So the proof is measured against the shape first, and only then replayed.
    if proof.layers.len() != num_variables {
        return Err(FractionGkrError::InvalidLayerCount {
            expected: num_variables,
            actual: proof.layers.len(),
        });
    }
    for (layer, messages) in proof.layers.iter().enumerate() {
        if messages.round_polys.len() != layer {
            return Err(FractionGkrError::InvalidRoundCount {
                layer,
                expected: layer,
                actual: messages.round_polys.len(),
            });
        }
    }

    // A zero root denominator has no inverse, which the arithmetic witnesses.
    let root_denominator = arith.load_ext(proof.root_denominator);
    arith.batch_inverse(core::slice::from_ref(&root_denominator), |_| {
        FractionGkrError::ZeroRootDenominator
    })?;

    let mut transcript = arith.seeding(|| {
        FractionGkrVerifierTranscript::<Challenger, A::Val, A::Challenge>::new(
            challenger,
            FractionGkrShape { num_variables },
        )
    });
    arith.will_observe_ext(&root_denominator);
    transcript.root_denominator(&proof.root_denominator);

    let mut point = Point::<A::EF>::new(Vec::new());
    let mut numerator = A::EF::ZERO;
    let mut denominator = root_denominator;
    let interpolator = RoundPolyInterpolator::<A::Challenge>::new(ROUND_POLY_LEN);

    // Position of the first layer whose sumcheck did not close, if any.
    //
    // A failing layer does not stop the replay.
    // Every remaining value comes from the proof, so the walk stays well defined,
    // and the driver reaches finalisation on every path out of this function.
    let mut inconsistent_layer = None;

    for (layer_index, layer) in proof.layers.iter().enumerate() {
        let lambda = arith.sampled_ext(transcript.begin_layer());
        let mut running_sum = numerator + lambda.dup() * denominator;
        let mut round_point = Vec::with_capacity(layer_index + 1);
        for round_poly in &layer.round_polys {
            let lifted = arith.load_ext_slice(round_poly);
            for eval in lifted.iter() {
                arith.will_observe_ext(eval);
            }
            let r = arith.sampled_ext(transcript.round(round_poly));
            running_sum = interpolator.eval_in(&lifted, running_sum, &r);
            round_point.push(r);
        }

        let claims = &layer.claims;
        let [n0, d0, n1, d1] =
            [claims.n0, claims.d0, claims.n1, claims.d1].map(|value| arith.load_ext(value));
        for value in [&n0, &d0, &n1, &d1] {
            arith.will_observe_ext(value);
        }
        let lifted = SplitFraction { n0, d0, n1, d1 };

        // The first inconsistent layer is reported once the whole transcript is replayed.
        let expected = Point::eval_eq(point.as_slice(), &round_point) * lifted.gate(lambda);
        if let Err(error) = arith.assert_zero(running_sum - expected, || {
            FractionGkrError::LayerConsistency { layer: layer_index }
        }) {
            inconsistent_layer.get_or_insert(error);
        }

        let branch = arith.sampled_ext(transcript.end_layer(claims));
        let SplitFraction { n0, d0, n1, d1 } = lifted;
        numerator = n0.dup() + branch.dup() * (n1 - n0);
        denominator = d0.dup() + branch.dup() * (d1 - d0);
        round_point.insert(0, branch);
        point = Point::new(round_point);
    }

    transcript.finish();

    if let Some(error) = inconsistent_layer {
        return Err(error);
    }

    Ok(FractionGkrOutput {
        point,
        numerator,
        denominator,
    })
}
