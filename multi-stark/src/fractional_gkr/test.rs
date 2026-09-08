use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::{DuplexChallenger, FieldChallenger};
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PackedValue, PrimeCharacteristicRing};
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::{Poly, PolyMaybePacked};
use p3_util::log2_strict_usize;
use rand::SeedableRng;
use rand::rngs::SmallRng;

use super::{
    Fraction, FractionGkrError, SplitFraction, prove_fractional_gkr, verify_fractional_gkr,
};

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;
type Perm = Poseidon2BabyBear<16>;
type Challenger = DuplexChallenger<F, Perm, 16, 8>;

fn fresh_challenger() -> Challenger {
    let mut rng = SmallRng::seed_from_u64(0xFACC_7100);
    Challenger::new(Perm::new_from_rng_128(&mut rng))
}

fn zero_sum_fraction(rng: &mut SmallRng, num_variables: usize) -> (Poly<F>, Poly<EF>) {
    loop {
        let mut numer = Poly::<F>::rand(rng, num_variables);
        let mut denom = Poly::<EF>::rand(rng, num_variables);
        let last = numer.num_evals() - 1;

        if denom.as_slice()[..last].contains(&EF::ZERO) {
            continue;
        }

        let partial_sum = numer.as_slice()[..last]
            .iter()
            .zip(&denom.as_slice()[..last])
            .map(|(&numer, &denom)| denom.inverse() * numer)
            .sum::<EF>();
        if partial_sum == EF::ZERO {
            continue;
        }

        numer.as_mut_slice()[last] = F::ONE;
        denom.as_mut_slice()[last] = -partial_sum.inverse();
        return (numer, denom);
    }
}

#[test]
fn accepts_honest_proofs() {
    let (numer, denom) = zero_sum_fraction(&mut SmallRng::seed_from_u64(3), 6);
    let mut prover_challenger = fresh_challenger();
    let (proof, prover_output) = prove_fractional_gkr(
        &Fraction {
            n: numer.clone(),
            d: PolyMaybePacked::Scalar(denom),
        },
        &mut prover_challenger,
    );

    let mut verifier_challenger = fresh_challenger();
    let verifier_output =
        verify_fractional_gkr::<F, EF, _>(&proof, numer.num_variables(), &mut verifier_challenger)
            .unwrap();

    assert_eq!(verifier_output, prover_output);
}

#[test]
fn rejects_a_tampered_round_polynomial() {
    let (numer, denom) = zero_sum_fraction(&mut SmallRng::seed_from_u64(4), 6);
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
    let (numer, denom) = zero_sum_fraction(&mut SmallRng::seed_from_u64(5), 6);
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
fn rejects_a_tampered_root_denominator() {
    let (numer, denom) = zero_sum_fraction(&mut SmallRng::seed_from_u64(7), 6);
    let mut prover_challenger = fresh_challenger();
    let (mut proof, _) = prove_fractional_gkr(
        &Fraction {
            n: numer,
            d: PolyMaybePacked::Scalar(denom),
        },
        &mut prover_challenger,
    );
    proof.root_denominator *= EF::TWO;

    let mut verifier_challenger = fresh_challenger();
    assert_eq!(
        verify_fractional_gkr::<F, EF, _>(&proof, 6, &mut verifier_challenger),
        Err(FractionGkrError::LayerConsistency { layer: 0 })
    );
}

#[test]
fn rejects_the_wrong_layer_shape() {
    let (numer, denom) = zero_sum_fraction(&mut SmallRng::seed_from_u64(6), 6);
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

#[test]
fn rejects_a_layer_carrying_the_wrong_round_count() {
    // Layer `i` proves a sum over `i` variables, so layer 3 owes exactly 3 round polynomials.
    //
    //     described:   3
    //     proof holds: 2   -> rejected before the transcript exists
    //
    // The check has to happen first. The driver panics on a step it was never
    // described with, and a panic there would land on top of the drop-time check.
    let (numer, denom) = zero_sum_fraction(&mut SmallRng::seed_from_u64(11), 6);
    let mut prover_challenger = fresh_challenger();
    let (mut proof, _) = prove_fractional_gkr(
        &Fraction {
            n: numer,
            d: PolyMaybePacked::Scalar(denom),
        },
        &mut prover_challenger,
    );
    proof.layers[3].round_polys.pop();

    let mut verifier_challenger = fresh_challenger();
    assert_eq!(
        verify_fractional_gkr::<F, EF, _>(&proof, 6, &mut verifier_challenger),
        Err(FractionGkrError::InvalidRoundCount {
            layer: 3,
            expected: 3,
            actual: 2,
        })
    );
}

#[test]
fn rejects_a_variable_count_the_prover_never_ran() {
    // Both sides derive the variable count from their own plan, never from the proof.
    //
    // A verifier holding a different one seeds a different sponge, so the two
    // never share a challenge. The layer count catches this first.
    let (numer, denom) = zero_sum_fraction(&mut SmallRng::seed_from_u64(12), 6);
    let mut prover_challenger = fresh_challenger();
    let (proof, _) = prove_fractional_gkr(
        &Fraction {
            n: numer,
            d: PolyMaybePacked::Scalar(denom),
        },
        &mut prover_challenger,
    );

    let mut verifier_challenger = fresh_challenger();
    assert_eq!(
        verify_fractional_gkr::<F, EF, _>(&proof, 5, &mut verifier_challenger),
        Err(FractionGkrError::InvalidLayerCount {
            expected: 5,
            actual: 6,
        })
    );
}

#[test]
fn the_reduction_leaves_both_sponges_in_the_same_state() {
    // Completeness at the seam: whatever runs after the reduction sees one shared state.
    //
    // The driver seeds on construction and absorbs on every step, so a prover
    // and a verifier that agree step for step must agree here too.
    let (numer, denom) = zero_sum_fraction(&mut SmallRng::seed_from_u64(13), 5);
    let mut prover_challenger = fresh_challenger();
    let (proof, _) = prove_fractional_gkr(
        &Fraction {
            n: numer,
            d: PolyMaybePacked::Scalar(denom),
        },
        &mut prover_challenger,
    );

    let mut verifier_challenger = fresh_challenger();
    verify_fractional_gkr::<F, EF, _>(&proof, 5, &mut verifier_challenger)
        .expect("an honest reduction verifies");

    let prover_next: EF = prover_challenger.sample_algebra_element();
    let verifier_next: EF = verifier_challenger.sample_algebra_element();
    assert_eq!(prover_next, verifier_next);
}

#[test]
fn test_gkr_identities() {
    let mut rng = SmallRng::seed_from_u64(1);

    for num_variables in 1..=10 {
        let (numer, denom) = zero_sum_fraction(&mut rng, num_variables);

        let mut prover_challenger = fresh_challenger();
        let (proof, prover_output) = prove_fractional_gkr(
            &Fraction {
                n: numer.clone(),
                d: PolyMaybePacked::Scalar(denom.clone()),
            },
            &mut prover_challenger,
        );

        assert_eq!(proof.layers.len(), num_variables);
        assert!(
            proof
                .layers
                .iter()
                .enumerate()
                .all(|(layer, proof)| proof.round_polys.len() == layer)
        );
        assert_eq!(
            prover_output.numerator,
            numer.eval_base::<EF>(&prover_output.point),
        );
        assert_eq!(
            prover_output.denominator,
            denom.eval_ext::<F>(&prover_output.point),
        );

        let mut verifier_challenger = fresh_challenger();
        let verifier_output =
            verify_fractional_gkr::<F, EF, _>(&proof, num_variables, &mut verifier_challenger)
                .unwrap();
        assert_eq!(verifier_output, prover_output);

        let prover_final_challenge: EF = prover_challenger.sample_algebra_element();
        let verifier_final_challenge: EF = verifier_challenger.sample_algebra_element();
        assert_eq!(prover_final_challenge, verifier_final_challenge);
    }
}

#[test]
fn packed_denominator_preserves_the_gkr_transcript() {
    let mut rng = SmallRng::seed_from_u64(0xFACC_7103);
    let packing_variables = log2_strict_usize(<F as Field>::Packing::WIDTH);

    for num_variables in packing_variables.max(1)..=10 {
        let (numer, denom) = zero_sum_fraction(&mut rng, num_variables);
        let packed_denom = PolyMaybePacked::Packed(denom.pack::<F, EF>());
        let mut scalar_challenger = fresh_challenger();
        let (scalar_proof, scalar_output) = prove_fractional_gkr(
            &Fraction {
                n: numer.clone(),
                d: PolyMaybePacked::Scalar(denom),
            },
            &mut scalar_challenger,
        );
        let mut packed_challenger = fresh_challenger();
        let (packed_proof, packed_output) = prove_fractional_gkr(
            &Fraction {
                n: numer,
                d: packed_denom,
            },
            &mut packed_challenger,
        );

        assert_eq!(packed_proof, scalar_proof);
        assert_eq!(packed_output, scalar_output);
        assert_eq!(
            packed_challenger.sample_algebra_element::<EF>(),
            scalar_challenger.sample_algebra_element::<EF>()
        );
    }
}

#[test]
fn split_fraction_matches_unsplit_sum_and_evaluation() {
    let mut rng = SmallRng::seed_from_u64(0xFACC_7102);

    for num_variables in 2..=10 {
        let numer = Poly::<EF>::rand(&mut rng, num_variables);
        let denom = Poly::<EF>::rand(&mut rng, num_variables);
        let point = Point::<EF>::rand(&mut rng, num_variables);
        let half = numer.num_evals() / 2;
        let fraction = SplitFraction {
            n0: Poly::new(numer.as_slice()[..half].to_vec()),
            d0: Poly::new(denom.as_slice()[..half].to_vec()),
            n1: Poly::new(numer.as_slice()[half..].to_vec()),
            d1: Poly::new(denom.as_slice()[half..].to_vec()),
        };

        let direct_sum = numer
            .iter()
            .zip(denom.iter())
            .map(|(&numer, &denom)| numer * denom.inverse())
            .sum::<EF>();

        assert_eq!(fraction.sum(), direct_sum);
        assert_eq!(
            fraction.eval(&point),
            (numer.eval_ext::<F>(&point), denom.eval_ext::<F>(&point)),
        );
    }
}
