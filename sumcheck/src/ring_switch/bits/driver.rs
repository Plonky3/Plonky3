//! One bit-alphabet ring-switching reduction, driven end to end.
//!
//! ```text
//!     in    t(r) = s        a claim about the bit witness
//!     out   t'(r') = s'     a claim about its packing, which a commitment answers
//! ```
//!
//! # A reduction, not a filter
//!
//! A false input claim is not rejected outright.
//! It survives as a false surviving claim, except with the probability below.
//!
//! Two cases the rounds cannot catch are left to whatever discharges the claim:
//!
//! - a closing weight of zero, which constrains the surviving value not at all
//! - a tampered element, with the rest of the proof adapted to the sum it implies
//!
//! # Booleanity is free
//!
//! The packing is a bijection between bit strings and elements of the level.
//! Every bit pattern is an element, and every element is some bit pattern.
//!
//! A commitment to a packed multilinear is therefore a commitment to a bit witness.
//! No range check, no auxiliary constraint, and nothing here to verify.
//!
//! # Soundness
//!
//! The reduction's own error is `(d_log + 2 l') / |EF|` (eprint 2024/504, Theorem 3.5).
//! It runs over `d_log` absorbed coordinates and `l'` rounds:
//!
//! - `d_log / |EF|` from the batching draw that collapses the row claims into one.
//! - `2 l' / |EF|` for the rounds of degree-two sumcheck.
//!
//! Both terms are per-attempt, because the description holds no grinding step.
//! A protocol needing a total bound supplies the grinding outside this run.

use p3_binary_field::TowerLevel;
use p3_challenger::fs::TranscriptField;
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_multilinear_util::point::Point;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::packing::BitPacking;
use super::reduction::{BitRingSwitch, BitRingSwitchError};
use super::tensor::BitTensor;
use super::transcript::{
    BitRingSwitchProverTranscript, BitRingSwitchShape, BitRingSwitchVerifierTranscript,
    TranscriptWidth,
};
use crate::data::SumcheckData;
use crate::error::SumcheckError;
use crate::product_polynomial::ProductPolynomial;
use crate::strategy::{Basis, SumcheckProver, VariableOrder};

/// The messages one bit-alphabet reduction puts on the wire.
///
/// The element travels by rows, one bit per matrix entry:
///
/// ```text
///     by rows          d elements
///     byte per entry   d^2 elements
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "EF: TowerLevel", deserialize = "EF: TowerLevel"))]
pub struct BitRingSwitchProof<EF> {
    /// The tensor element both checks read, by rows and by columns.
    pub tensor: BitTensor<EF>,
    /// The rounds of the batched degree-two sumcheck.
    pub sumcheck: SumcheckData<EF, EF>,
    /// The value of the surviving claim.
    pub final_eval: EF,
}

/// Why a bit-alphabet reduction was rejected.
#[derive(Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum BitRingSwitchProofError {
    /// A list crossing the wire is not the width the description fixes.
    #[error(transparent)]
    Width(#[from] TranscriptWidth),

    /// The reduction could not be set up over the point supplied.
    #[error(transparent)]
    Reduction(#[from] BitRingSwitchError),

    /// The claimed evaluation is not the column reading of the element.
    #[error("the claimed evaluation is not the column reading of the tensor element")]
    ClaimMismatch,

    /// A round of the batched sumcheck failed.
    #[error(transparent)]
    Sumcheck(#[from] SumcheckError),

    /// The surviving claim does not close the sumcheck against the closing weight.
    #[error("the surviving claim does not close the sumcheck")]
    FinalCheck,

    /// The sumcheck carries grinding witnesses this reduction never searches for.
    #[error("the sumcheck carries {actual} proof-of-work witnesses, expected none")]
    NonEmptyPowWitnesses {
        /// Witnesses the proof supplied.
        actual: usize,
    },
}

/// Reduce a claim about a bit witness to one about its packing.
///
/// # Returns
///
/// The proof, the point the rounds ended at, and the surviving claim's value.
///
/// # Panics
///
/// Panics unless the point names the absorbed coordinates and the packing what they leave.
pub fn prove_bit_ring_switch<EF, Challenger>(
    packing: &BitPacking<EF>,
    r: &Point<EF>,
    challenger: &mut Challenger,
) -> (BitRingSwitchProof<EF>, Point<EF>, EF)
where
    EF: TranscriptField + TowerLevel + Send + Sync,
    Challenger: FieldChallenger<EF> + GrindingChallenger<Witness = EF>,
{
    // The description comes from the point handed in, never from a proof.
    let shape = BitRingSwitchShape::new(r.num_variables());
    let rounds = shape.sumcheck_rounds::<EF>();
    assert_eq!(
        packing.num_variables(),
        rounds,
        "the packing must have the {rounds} variables the evaluation point leaves"
    );

    let reduction = BitRingSwitch::new(r).expect("the shape already checked the point's width");
    let tensor = reduction
        .tensor(packing)
        .expect("the packing was just checked against the reduction");

    // The element is a function of the kept coordinates alone.
    // It is therefore ready before the transcript needs it.
    let mut transcript = BitRingSwitchProverTranscript::<Challenger, EF>::new(challenger, shape);
    let r_batch = transcript.statement(r, tensor.rows());

    // The batching challenge arrived after the element, which is the order soundness needs.
    let batch = reduction
        .batch(&r_batch)
        .expect("the draw names the absorbed coordinates by construction");
    let poly = ProductPolynomial::new_unpacked(
        VariableOrder::Prefix,
        packing.poly().clone(),
        batch.weights(),
    );
    let mut prover = SumcheckProver::new(poly, batch.initial_sum(&tensor));
    let mut sumcheck = SumcheckData::default();

    let r_prime = transcript.batched_sumcheck(|challenger| {
        prover.compute_sumcheck_polynomials(&mut sumcheck, challenger, rounds, 0, None)
    });

    // After the last round the evaluation side has folded to the packing at that point.
    // No second pass over the packing is needed to find it.
    let final_eval = prover.evals().as_slice()[0];
    transcript.surviving_claim(final_eval);
    transcript.finish();

    (
        BitRingSwitchProof {
            tensor,
            sumcheck,
            final_eval,
        },
        r_prime,
        final_eval,
    )
}

/// Replay a reduction and return the claim it leaves behind, as a point and a value.
///
/// Discharging that pair against a commitment to the packing is the caller's business.
///
/// # Errors
///
/// - A malformed element, or a point of the wrong width.
/// - A non-empty grinding witness list, since this reduction never grinds.
/// - A claimed evaluation disagreeing with the element's columns.
/// - A failed sumcheck round, or a final claim that does not close it.
pub fn verify_bit_ring_switch<EF, Challenger>(
    proof: &BitRingSwitchProof<EF>,
    r: &Point<EF>,
    claimed_sum: EF,
    challenger: &mut Challenger,
) -> Result<(Point<EF>, EF), BitRingSwitchProofError>
where
    EF: TranscriptField + TowerLevel,
    Challenger: FieldChallenger<EF> + GrindingChallenger<Witness = EF>,
{
    // Both structural rejections below run before the challenger is touched.
    // A malformed proof therefore never leaves a half-advanced transcript.
    if !proof.tensor.is_well_formed() {
        return Err(TranscriptWidth::TensorRows {
            expected: BitTensor::<EF>::DIMENSION,
            actual: proof.tensor.rows().len(),
        }
        .into());
    }
    if !proof.sumcheck.pow_witnesses.is_empty() {
        return Err(BitRingSwitchProofError::NonEmptyPowWitnesses {
            actual: proof.sumcheck.pow_witnesses.len(),
        });
    }

    let reduction = BitRingSwitch::new(r)?;
    let shape = BitRingSwitchShape::new(r.num_variables());
    let rounds = reduction.num_variables();

    let mut transcript = BitRingSwitchVerifierTranscript::<Challenger, EF>::new(challenger, shape);
    let r_batch = transcript.statement(r, proof.tensor.rows())?;

    // The columns are the witness's bit planes at the kept coordinates.
    // The absorbed coordinates weigh them back together, their only use here.
    if reduction.incoming_claim(&proof.tensor) != claimed_sum {
        transcript.abort();
        return Err(BitRingSwitchProofError::ClaimMismatch);
    }

    let batch = reduction
        .batch(&r_batch)
        .expect("the draw names the absorbed coordinates by construction");

    // The initial sum is derived from the element's rows, never taken from the prover.
    // That is what makes a dishonest element catchable at all.
    let mut sum = batch.initial_sum(&proof.tensor);
    let replay = transcript.batched_sumcheck(|challenger| {
        proof
            .sumcheck
            .verify_rounds(challenger, &mut sum, rounds, 0, Basis::Evaluation)
    });
    let r_prime = match replay {
        Ok(point) => point,
        Err(error) => {
            transcript.abort();
            return Err(error.into());
        }
    };

    transcript.surviving_claim(proof.final_eval);
    transcript.finish();

    // The rounds close on the weight multilinear at their end point, times the value.
    // The weight comes through the equality element rather than another pass.
    let closing = batch.closing_weight(&r_prime)?;
    if sum != closing * proof.final_eval {
        return Err(BitRingSwitchProofError::FinalCheck);
    }

    Ok((r_prime, proof.final_eval))
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_binary_field::{BinaryChallenger, BinaryField16};
    use p3_challenger::HashChallenger;
    use p3_field::PrimeCharacteristicRing;
    use p3_keccak::Keccak256Hash;
    use p3_multilinear_util::poly::Poly;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    type EF = BinaryField16;
    type Chal = BinaryChallenger<EF, HashChallenger<u8, Keccak256Hash, 32>>;

    fn challenger() -> Chal {
        Chal::from_hasher(Vec::new(), Keccak256Hash)
    }

    /// A random bit witness of the given byte length.
    fn bits(seed: u64, bytes: usize) -> Vec<u8> {
        let mut rng = SmallRng::seed_from_u64(seed);
        (0..bytes).map(|_| rng.random::<u8>()).collect()
    }

    /// The witness as a multilinear over every variable, one element per bit.
    ///
    /// The definition of the claim being reduced, with no packing anywhere in it.
    fn embedded(witness: &[u8]) -> Poly<EF> {
        Poly::new(
            (0..witness.len() * 8)
                .map(|cell| {
                    if (witness[cell / 8] >> (cell % 8)) & 1 == 1 {
                        EF::ONE
                    } else {
                        EF::ZERO
                    }
                })
                .collect::<Vec<_>>(),
        )
    }

    #[test]
    fn the_reduction_round_trips_and_leaves_a_true_claim() {
        // Invariant: the surviving claim is the truth about the packed polynomial.
        //
        //     in    t(r) = s   at a random point over all 8 variables
        //     out   t'(r')     over the 4 the packing keeps, 4 being absorbed
        let witness = bits(0x81A5, 32);
        let packing = BitPacking::<EF>::new(&witness).unwrap();
        let r = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0x81A6), 8);
        let claim = embedded(&witness).eval_base(&r);

        let mut prover_chal = challenger();
        let (proof, r_prime_p, s_prime_p) = prove_bit_ring_switch(&packing, &r, &mut prover_chal);

        let mut verifier_chal = challenger();
        let (r_prime_v, s_prime_v) =
            verify_bit_ring_switch(&proof, &r, claim, &mut verifier_chal).unwrap();

        assert_eq!(r_prime_p, r_prime_v);
        assert_eq!(s_prime_p, s_prime_v);
        assert_eq!(s_prime_v, packing.poly().eval_base(&r_prime_v));
    }

    #[test]
    fn a_claim_the_element_does_not_support_is_rejected() {
        // The column reading is what ties the element to the incoming claim.
        let witness = bits(0x0AD, 32);
        let packing = BitPacking::<EF>::new(&witness).unwrap();
        let r = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0x0AE), 8);
        let claim = embedded(&witness).eval_base(&r);

        let (proof, _, _) = prove_bit_ring_switch(&packing, &r, &mut challenger());

        let err =
            verify_bit_ring_switch(&proof, &r, claim + EF::ONE, &mut challenger()).unwrap_err();
        assert_eq!(err, BitRingSwitchProofError::ClaimMismatch);
    }

    #[test]
    fn a_tampered_element_breaks_the_reduction() {
        // Invariant: both readings are of the same coefficients.
        //
        //     rows    -> the sum the rounds start from
        //     columns -> the claim they are checked against
        //
        // Mutation: add one to a row, which moves the sum but not the claim.
        let witness = bits(0x7A17, 32);
        let packing = BitPacking::<EF>::new(&witness).unwrap();
        let r = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0x7A18), 8);
        let claim = embedded(&witness).eval_base(&r);

        let (mut proof, _, _) = prove_bit_ring_switch(&packing, &r, &mut challenger());
        let mut rows = proof.tensor.rows().to_vec();
        rows[3] += EF::ONE;
        proof.tensor = BitTensor::try_from(rows).unwrap();

        assert!(verify_bit_ring_switch(&proof, &r, claim, &mut challenger()).is_err());
    }

    #[test]
    fn a_stray_grinding_witness_is_refused_before_the_transcript() {
        // Invariant: this reduction never grinds, so a witness rides along unbound.
        //
        // The rejection is structural, so it may not advance the sponge.
        //
        // A short element cannot be built at all.
        // Every route into one checks the row count, deserialization included.
        //
        // The verifier's shape check is therefore defence in depth, not a reachable path.
        let witness = bits(0x5407, 32);
        let packing = BitPacking::<EF>::new(&witness).unwrap();
        let r = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0x5408), 8);
        let claim = embedded(&witness).eval_base(&r);

        let (mut proof, _, _) = prove_bit_ring_switch(&packing, &r, &mut challenger());
        proof.sumcheck.pow_witnesses.push(EF::ONE);

        assert_eq!(
            verify_bit_ring_switch(&proof, &r, claim, &mut challenger()).unwrap_err(),
            BitRingSwitchProofError::NonEmptyPowWitnesses { actual: 1 }
        );

        // A row count the level does not admit has no constructor.
        assert!(BitTensor::<EF>::try_from(alloc::vec![EF::ONE; 4]).is_err());
    }

    #[test]
    fn a_point_of_the_wrong_width_is_reported_rather_than_asserted() {
        // A point narrower than the absorbed coordinates describes no reduction.
        let witness = bits(0x9107, 32);
        let packing = BitPacking::<EF>::new(&witness).unwrap();
        let r = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0x9108), 8);
        let claim = embedded(&witness).eval_base(&r);
        let (proof, _, _) = prove_bit_ring_switch(&packing, &r, &mut challenger());

        let narrow = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0x9109), 3);
        let err = verify_bit_ring_switch(&proof, &narrow, claim, &mut challenger()).unwrap_err();
        assert!(
            matches!(err, BitRingSwitchProofError::Reduction(_)),
            "{err:?}"
        );
    }

    proptest! {
        // Each case runs a full reduction on both sides, so a few dozen keep the suite fast.
        #![proptest_config(ProptestConfig { cases: 24, ..ProptestConfig::default() })]

        /// Every witness length the level admits, at a random point each time.
        #[test]
        fn the_reduction_round_trips_over_random_inputs(
            log_bytes in 1usize..=6,
            witness_seed: u64,
            point_seed: u64,
        ) {
            let witness = bits(witness_seed, 1 << log_bytes);
            let packing = BitPacking::<EF>::new(&witness).unwrap();
            let variables = log_bytes + 3;
            let r = Point::<EF>::rand(&mut SmallRng::seed_from_u64(point_seed), variables);
            let claim = embedded(&witness).eval_base(&r);

            let (proof, _, s_prime_p) =
                prove_bit_ring_switch(&packing, &r, &mut challenger());
            let (r_prime, s_prime) =
                verify_bit_ring_switch(&proof, &r, claim, &mut challenger()).unwrap();

            prop_assert_eq!(s_prime, s_prime_p);
            prop_assert_eq!(s_prime, packing.poly().eval_base(&r_prime));
        }
    }
}
