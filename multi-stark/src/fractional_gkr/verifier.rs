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

/// Verify a fractional-GKR proof and recover its input-table opening claims.
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
    challenger.observe_algebra_element(proof.root_numerator);
    let mut point = Point::<EF>::new(Vec::new());
    let mut numerator = proof.root_numerator;
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

#[cfg(test)]
mod tests {
    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_multilinear_util::poly::{Poly, PolyMaybePacked};
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::fractional_gkr::Fraction;
    use crate::fractional_gkr::prover::prove_fractional_gkr;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type Challenger = DuplexChallenger<F, Perm, 16, 8>;

    fn fresh_challenger() -> Challenger {
        let mut rng = SmallRng::seed_from_u64(0xFACC_7100);
        Challenger::new(Perm::new_from_rng_128(&mut rng))
    }

    fn fraction(seed: u64) -> (Poly<F>, Poly<EF>) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let numer = Poly::new((0..1 << 6).map(|_| rng.random()).collect());
        let denom = Poly::new((0..1 << 6).map(|i| EF::from_u64((i + 1) as u64)).collect());
        (numer, denom)
    }

    #[test]
    fn accepts_honest_proofs() {
        let (numer, denom) = fraction(3);
        let mut prover_challenger = fresh_challenger();
        let (proof, prover_output) = prove_fractional_gkr(
            &Fraction {
                n: numer.clone(),
                d: PolyMaybePacked::Scalar(denom),
            },
            &mut prover_challenger,
        );

        let mut verifier_challenger = fresh_challenger();
        let verifier_output = verify_fractional_gkr::<F, EF, _>(
            &proof,
            numer.num_variables(),
            &mut verifier_challenger,
        )
        .unwrap();

        assert_eq!(verifier_output, prover_output);
    }

    #[test]
    fn rejects_a_tampered_round_polynomial() {
        let (numer, denom) = fraction(4);
        let mut prover_challenger = fresh_challenger();
        let (mut proof, _) = prove_fractional_gkr(
            &Fraction {
                n: numer,
                d: PolyMaybePacked::Scalar(denom),
            },
            &mut prover_challenger,
        );
        proof.layers[1].round_polys[0][0] += EF::ONE;

        let mut verifier_challenger = fresh_challenger();
        assert!(matches!(
            verify_fractional_gkr::<F, EF, _>(&proof, 6, &mut verifier_challenger),
            Err(FractionGkrError::LayerConsistency { .. })
        ));
    }

    #[test]
    fn rejects_a_tampered_claim() {
        let (numer, denom) = fraction(5);
        let mut prover_challenger = fresh_challenger();
        let (mut proof, _) = prove_fractional_gkr(
            &Fraction {
                n: numer,
                d: PolyMaybePacked::Scalar(denom),
            },
            &mut prover_challenger,
        );
        proof.layers[0].claims.n0 += EF::ONE;

        let mut verifier_challenger = fresh_challenger();
        assert_eq!(
            verify_fractional_gkr::<F, EF, _>(&proof, 6, &mut verifier_challenger),
            Err(FractionGkrError::LayerConsistency { layer: 0 })
        );
    }

    #[test]
    fn rejects_a_tampered_root() {
        let (numer, denom) = fraction(7);
        let mut prover_challenger = fresh_challenger();
        let (mut proof, _) = prove_fractional_gkr(
            &Fraction {
                n: numer,
                d: PolyMaybePacked::Scalar(denom),
            },
            &mut prover_challenger,
        );
        proof.root_numerator += EF::ONE;

        let mut verifier_challenger = fresh_challenger();
        assert_eq!(
            verify_fractional_gkr::<F, EF, _>(&proof, 6, &mut verifier_challenger),
            Err(FractionGkrError::LayerConsistency { layer: 0 })
        );
    }

    #[test]
    fn rejects_the_wrong_layer_shape() {
        let (numer, denom) = fraction(6);
        let mut prover_challenger = fresh_challenger();
        let (mut proof, _) = prove_fractional_gkr(
            &Fraction {
                n: numer,
                d: PolyMaybePacked::Scalar(denom),
            },
            &mut prover_challenger,
        );
        proof.layers.pop();

        let mut verifier_challenger = fresh_challenger();
        assert_eq!(
            verify_fractional_gkr::<F, EF, _>(&proof, 6, &mut verifier_challenger),
            Err(FractionGkrError::InvalidLayerCount {
                expected: 6,
                actual: 5,
            })
        );
    }
}
