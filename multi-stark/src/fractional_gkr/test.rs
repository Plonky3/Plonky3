use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_binary_field::{BinaryChallenger, BinaryField32, BinaryField128, TowerLevel};
use p3_challenger::fs::TranscriptField;
use p3_challenger::{DuplexChallenger, FieldChallenger, HashChallenger};
use p3_field::extension::BinomialExtensionField;
use p3_field::{ExtensionField, Field, PackedValue, PrimeCharacteristicRing};
use p3_keccak::Keccak256Hash;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::{Poly, PolyMaybePacked};
use p3_util::log2_strict_usize;
use rand::SeedableRng;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;

use super::{
    Fraction, FractionGkrError, FractionGkrOutput, FractionGkrProof, LeafNumerator, SplitFraction,
    prove_fractional_gkr, verify_fractional_gkr,
};

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;
type Perm = Poseidon2BabyBear<16>;
type Challenger = DuplexChallenger<F, Perm, 16, 8>;

fn fresh_challenger() -> Challenger {
    let mut rng = SmallRng::seed_from_u64(0xFACC_7100);
    Challenger::new(Perm::new_from_rng_128(&mut rng))
}

/// Run the reduction over a base-field leaf numerator.
///
/// Every test here exercises this shape, apart from the extension-field ones.
fn prove_base_leaf(
    numer: &Poly<F>,
    denom: &PolyMaybePacked<F, EF>,
    challenger: &mut Challenger,
) -> (FractionGkrProof<EF>, FractionGkrOutput<EF>) {
    prove_fractional_gkr(
        Fraction {
            n: LeafNumerator::Base(numer),
            d: denom,
        },
        challenger,
    )
}

/// Draw a random fraction table whose fractions sum to zero.
///
/// The last leaf is solved for, so every other leaf is unconstrained.
fn zero_sum<N, A>(rng: &mut SmallRng, num_variables: usize) -> (Poly<N>, Poly<A>)
where
    N: Field,
    A: ExtensionField<N>,
    StandardUniform: Distribution<N> + Distribution<A>,
{
    loop {
        let mut numer = Poly::<N>::rand(rng, num_variables);
        let mut denom = Poly::<A>::rand(rng, num_variables);
        let last = numer.num_evals() - 1;

        // A zero denominator anywhere would leave the table meaningless.
        if denom.as_slice()[..last].contains(&A::ZERO) {
            continue;
        }

        // Everything but the last leaf sums to this, which the last leaf has to cancel.
        let partial_sum = numer.as_slice()[..last]
            .iter()
            .zip(&denom.as_slice()[..last])
            .map(|(&numer, &denom)| denom.inverse() * numer)
            .sum::<A>();
        if partial_sum == A::ZERO {
            continue;
        }

        numer.as_mut_slice()[last] = N::ONE;
        denom.as_mut_slice()[last] = -partial_sum.inverse();
        return (numer, denom);
    }
}

#[test]
fn accepts_honest_proofs() {
    let (numer, denom) = zero_sum::<F, EF>(&mut SmallRng::seed_from_u64(3), 6);
    let mut prover_challenger = fresh_challenger();
    let (proof, prover_output) = prove_base_leaf(
        &numer,
        &PolyMaybePacked::Scalar(denom),
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
    let (numer, denom) = zero_sum::<F, EF>(&mut SmallRng::seed_from_u64(4), 6);
    let mut prover_challenger = fresh_challenger();
    let (mut proof, _) = prove_base_leaf(
        &numer,
        &PolyMaybePacked::Scalar(denom),
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
    let (numer, denom) = zero_sum::<F, EF>(&mut SmallRng::seed_from_u64(5), 6);
    let mut prover_challenger = fresh_challenger();
    let (mut proof, _) = prove_base_leaf(
        &numer,
        &PolyMaybePacked::Scalar(denom),
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
    let (numer, denom) = zero_sum::<F, EF>(&mut SmallRng::seed_from_u64(7), 6);
    let mut prover_challenger = fresh_challenger();
    let (mut proof, _) = prove_base_leaf(
        &numer,
        &PolyMaybePacked::Scalar(denom),
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
    let (numer, denom) = zero_sum::<F, EF>(&mut SmallRng::seed_from_u64(6), 6);
    let mut prover_challenger = fresh_challenger();
    let (mut proof, _) = prove_base_leaf(
        &numer,
        &PolyMaybePacked::Scalar(denom),
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
    let (numer, denom) = zero_sum::<F, EF>(&mut SmallRng::seed_from_u64(11), 6);
    let mut prover_challenger = fresh_challenger();
    let (mut proof, _) = prove_base_leaf(
        &numer,
        &PolyMaybePacked::Scalar(denom),
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
    let (numer, denom) = zero_sum::<F, EF>(&mut SmallRng::seed_from_u64(12), 6);
    let mut prover_challenger = fresh_challenger();
    let (proof, _) = prove_base_leaf(
        &numer,
        &PolyMaybePacked::Scalar(denom),
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
    let (numer, denom) = zero_sum::<F, EF>(&mut SmallRng::seed_from_u64(13), 5);
    let mut prover_challenger = fresh_challenger();
    let (proof, _) = prove_base_leaf(
        &numer,
        &PolyMaybePacked::Scalar(denom),
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
        let (numer, denom) = zero_sum::<F, EF>(&mut rng, num_variables);

        let mut prover_challenger = fresh_challenger();
        let (proof, prover_output) = prove_base_leaf(
            &numer,
            &PolyMaybePacked::Scalar(denom.clone()),
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
        let (numer, denom) = zero_sum::<F, EF>(&mut rng, num_variables);
        let packed_denom = PolyMaybePacked::Packed(denom.pack::<F, EF>());
        let mut scalar_challenger = fresh_challenger();
        let (scalar_proof, scalar_output) = prove_base_leaf(
            &numer,
            &PolyMaybePacked::Scalar(denom),
            &mut scalar_challenger,
        );
        let mut packed_challenger = fresh_challenger();
        let (packed_proof, packed_output) =
            prove_base_leaf(&numer, &packed_denom, &mut packed_challenger);

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

/// Round-trip a reduction over a binary tower, in both storage modes.
///
/// A round polynomial is sent at the field's own first four values.
///
/// Over a binary tower those are bit patterns, not the integers zero through three.
///
/// Repeated addition, or naming a node by an integer, folds two of them onto one point.
///
/// The message then no longer pins the round polynomial down.
fn binary_round_trip<F, EF>(seed: u64)
where
    F: TranscriptField + TowerLevel,
    EF: ExtensionField<F>,
    StandardUniform: Distribution<F> + Distribution<EF>,
{
    let challenger = || {
        BinaryChallenger::<F, HashChallenger<u8, Keccak256Hash, 32>>::from_hasher(
            b"p3-fraction-gkr-binary-test".to_vec(),
            Keccak256Hash,
        )
    };

    let mut rng = SmallRng::seed_from_u64(seed);
    for num_variables in 1..=8 {
        let (numer, denom) = zero_sum::<F, EF>(&mut rng, num_variables);

        // Both storages describe the same table, so both must reduce to the same claims.
        let storages = [
            PolyMaybePacked::Scalar(denom.clone()),
            PolyMaybePacked::Packed(denom.pack::<F, EF>()),
        ];
        for storage in storages {
            let mut prover_challenger = challenger();
            let (proof, prover_output) = prove_fractional_gkr(
                Fraction {
                    n: LeafNumerator::Base(&numer),
                    d: &storage,
                },
                &mut prover_challenger,
            );

            let mut verifier_challenger = challenger();
            let verifier_output =
                verify_fractional_gkr::<F, EF, _>(&proof, num_variables, &mut verifier_challenger)
                    .expect("an honest reduction verifies");
            assert_eq!(verifier_output, prover_output);

            // The openings have to be the tables themselves at the point reached.
            assert_eq!(
                prover_output.numerator,
                numer.eval_base::<EF>(&prover_output.point)
            );
            assert_eq!(
                prover_output.denominator,
                denom.eval_ext::<F>(&prover_output.point)
            );
        }
    }
}

#[test]
fn a_binary_field_reduction_round_trips() {
    // One level, where the numerator and the denominator share a type.
    binary_round_trip::<BinaryField128, BinaryField128>(0x0B14_A247);
}

#[test]
fn a_mixed_binary_tower_reduction_round_trips() {
    // Two levels, so the leaf walk runs with a narrow numerator against a wide denominator.
    //
    // That is the arm a single-level pair never reaches.
    binary_round_trip::<BinaryField32, BinaryField128>(0x0B14_A248);
}

#[test]
fn an_extension_numerator_matches_a_base_numerator() {
    // The two numerator storages describe one table, so they must produce one proof.
    //
    // The wider storage changes where the values live, never what they are.
    let mut rng = SmallRng::seed_from_u64(0x0E47_0001);

    for num_variables in 1..=8 {
        let (numer, denom) = zero_sum::<F, EF>(&mut rng, num_variables);
        let denominator = PolyMaybePacked::Scalar(denom);

        // The same numerator, once in the base field and once lifted into the extension.
        let lifted = PolyMaybePacked::<F, EF>::Scalar(Poly::new(
            numer.as_slice().iter().copied().map(EF::from).collect(),
        ));

        let mut base_challenger = fresh_challenger();
        let (base_proof, base_output) = prove_fractional_gkr(
            Fraction {
                n: LeafNumerator::Base(&numer),
                d: &denominator,
            },
            &mut base_challenger,
        );

        let mut ext_challenger = fresh_challenger();
        let (ext_proof, ext_output) = prove_fractional_gkr(
            Fraction {
                n: LeafNumerator::Ext(&lifted),
                d: &denominator,
            },
            &mut ext_challenger,
        );

        assert_eq!(base_proof, ext_proof);
        assert_eq!(base_output, ext_output);
        assert_eq!(
            base_challenger.sample_algebra_element::<EF>(),
            ext_challenger.sample_algebra_element::<EF>()
        );
    }
}

#[test]
fn a_packed_extension_numerator_preserves_the_transcript() {
    // Packing moves the trailing variables into SIMD lanes and changes nothing else.
    //
    // The two storages must therefore agree step for step.
    let mut rng = SmallRng::seed_from_u64(0x0E47_0002);
    let packing_variables = log2_strict_usize(<F as Field>::Packing::WIDTH);

    for num_variables in packing_variables.max(1)..=8 {
        let (numer, denom) = zero_sum::<EF, EF>(&mut rng, num_variables);

        let scalar_numerator = PolyMaybePacked::<F, EF>::Scalar(numer.clone());
        let scalar_denominator = PolyMaybePacked::<F, EF>::Scalar(denom.clone());
        let packed_numerator = PolyMaybePacked::<F, EF>::Packed(numer.pack::<F, EF>());
        let packed_denominator = PolyMaybePacked::<F, EF>::Packed(denom.pack::<F, EF>());

        let mut scalar_challenger = fresh_challenger();
        let (scalar_proof, scalar_output) = prove_fractional_gkr(
            Fraction {
                n: LeafNumerator::Ext(&scalar_numerator),
                d: &scalar_denominator,
            },
            &mut scalar_challenger,
        );

        let mut packed_challenger = fresh_challenger();
        let (packed_proof, packed_output) = prove_fractional_gkr(
            Fraction {
                n: LeafNumerator::Ext(&packed_numerator),
                d: &packed_denominator,
            },
            &mut packed_challenger,
        );

        assert_eq!(packed_proof, scalar_proof);
        assert_eq!(packed_output, scalar_output);
        assert_eq!(
            packed_challenger.sample_algebra_element::<EF>(),
            scalar_challenger.sample_algebra_element::<EF>()
        );

        // The openings must still be the tables themselves at the reduction's point.
        assert_eq!(
            scalar_output.numerator,
            numer.eval_ext::<F>(&scalar_output.point)
        );
        assert_eq!(
            scalar_output.denominator,
            denom.eval_ext::<F>(&scalar_output.point)
        );
    }
}
