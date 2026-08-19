use alloc::vec::Vec;

use p3_challenger::FieldChallenger;
use p3_field::{ExtensionField, Field};
use p3_multilinear_util::point::Point;
use p3_sumcheck::generic_degree::RoundPolyInterpolator;
use thiserror::Error;

use super::{FractionGkrOutput, FractionGkrProof};

/// Malformed fractional-GKR proofs rejected by the verifier.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum FractionGkrError {
    #[error("fraction GKR requires at least one input variable")]
    InvalidVariableCount,
    #[error("fraction GKR proof has {actual} layers, expected {expected}")]
    InvalidLayerCount { expected: usize, actual: usize },
    #[error("fraction GKR layer {layer} has {actual} sumcheck rounds, expected {expected}")]
    InvalidRoundCount {
        layer: usize,
        expected: usize,
        actual: usize,
    },
    #[error("fraction GKR root denominator is zero")]
    ZeroRootDenominator,
    #[error("fraction GKR layer {layer} failed its gate consistency check")]
    LayerConsistency { layer: usize },
}

/// Verify the internal consistency of a zero-sum fractional-GKR reduction and
/// return its input-table opening claim.
///
/// This function does not authenticate the returned numerator and denominator
/// against committed input polynomials. To complete verification of the
/// zero-sum statement, the caller must verify that the returned values are the
/// openings of the corresponding input tables at the returned point.
pub fn verify_fractional_gkr<F, EF, Challenger>(
    proof: &FractionGkrProof<EF>,
    num_variables: usize,
    challenger: &mut Challenger,
) -> Result<FractionGkrOutput<EF>, FractionGkrError>
where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F>,
{
    if num_variables < 1 {
        return Err(FractionGkrError::InvalidVariableCount);
    }

    let expected_layers = num_variables;
    if proof.layers.len() != expected_layers {
        return Err(FractionGkrError::InvalidLayerCount {
            expected: expected_layers,
            actual: proof.layers.len(),
        });
    }

    if proof.root_denominator == EF::ZERO {
        return Err(FractionGkrError::ZeroRootDenominator);
    }

    challenger.observe_algebra_element(proof.root_denominator);
    let mut point = Point::<EF>::new(Vec::new());
    let mut numerator = EF::ZERO;
    let mut denominator = proof.root_denominator;
    let interpolator = RoundPolyInterpolator::new(3);

    for (layer_index, layer) in proof.layers.iter().enumerate() {
        let expected_rounds = layer_index;
        if layer.round_polys.len() != expected_rounds {
            return Err(FractionGkrError::InvalidRoundCount {
                layer: layer_index,
                expected: expected_rounds,
                actual: layer.round_polys.len(),
            });
        }

        let lambda: EF = challenger.sample_algebra_element();
        let mut running_sum = numerator + lambda * denominator;
        let mut round_point = Vec::with_capacity(expected_rounds + 1);
        for round_poly in &layer.round_polys {
            challenger.observe_algebra_slice(round_poly);
            let r: EF = challenger.sample_algebra_element();
            running_sum = interpolator.eval(round_poly, running_sum, r);
            round_point.push(r);
        }

        let expected = Point::eval_eq(point.as_slice(), &round_point) * layer.claims.gate(lambda);
        if running_sum != expected {
            return Err(FractionGkrError::LayerConsistency { layer: layer_index });
        }

        challenger.observe_algebra_slice(&[
            layer.claims.n0,
            layer.claims.d0,
            layer.claims.n1,
            layer.claims.d1,
        ]);
        let branch: EF = challenger.sample_algebra_element();
        numerator = layer.claims.n0 + branch * (layer.claims.n1 - layer.claims.n0);
        denominator = layer.claims.d0 + branch * (layer.claims.d1 - layer.claims.d0);
        round_point.insert(0, branch);
        point = Point::new(round_point);
    }

    Ok(FractionGkrOutput {
        point,
        numerator,
        denominator,
    })
}
