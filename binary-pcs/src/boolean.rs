//! A commitment to a function from the hypercube to `{0, 1}`, opened at any point.
//!
//! ```text
//!     commit   a bit witness, bit-sliced
//!     open     the multilinear extension at a point of the challenge field
//! ```
//!
//! # Three pieces, one interface
//!
//! ```text
//!     packing    bits  ->  the elements a commitment holds, by reinterpretation
//!     switch     a claim about the bits  ->  a claim about the packing
//!     opening    that claim  ->  discharged against the commitment
//! ```
//!
//! # Booleanity is free
//!
//! The packing is a bijection between bit strings and elements of the level.
//! Every bit pattern is an element, and every element is some bit pattern.
//!
//! A commitment to a packed multilinear is therefore a commitment to a bit witness.
//! No commitment exists to a function the hypercube sends outside `{0, 1}`.
//!
//! There is no range check here because there is nothing one could rule out.
//!
//! # One opening, not one per claim
//!
//! Several points reduce into one shared claim pool.
//! The pool seals with a single draw, once every surviving claim is bound.
//!
//! The commitment then answers for all of them in one call.
//! The pool's fold closes them with one equality rather than `k`.
//!
//! # What binds what
//!
//! Committing binds the root, and verifying replays that binding.
//! The two sides therefore start from one sponge state.
//!
//! The reduction then binds the opening point itself, before it sends anything.
//! A caller therefore owes no binding of its own, unlike a bare prescribed-point opening.
//!
//! The point the commitment is opened at comes out of the reduction's own rounds.
//!
//! # Soundness
//!
//! Two errors compose by a union bound.
//! The reduction charges `(d_log + 2 l') / |EF|`, and the commitment charges its own budget.
//!
//! Neither is subtracted from the other, and the reduction's term reaches no estimator here.
//! A protocol composing this accounts for both in its own budget.

use alloc::vec;
use alloc::vec::Vec;

use p3_binary_dft::EncodableLevel;
use p3_binary_field::{PackedGf2, TowerLevel, Underlier};
use p3_challenger::fs::TranscriptField;
use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::{Mmcs, MultilinearPcs};
use p3_field::Field;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_security::{SecurityTerm, claim_pool_term};
use p3_sumcheck::layout::{Layout, SuffixProver};
use p3_sumcheck::ring_switch::bits::{
    BitPacking, BitRingSwitch, BitRingSwitchProof, BitRingSwitchProofError, prove_bit_ring_switch,
    verify_bit_ring_switch,
};
use p3_sumcheck::{
    ClaimPool, ClaimPoolError, OpeningBatch, OpeningProtocol, PrescribedPointPcs, TableShape,
    TableSpec,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::error::BinaryPcsError;
use crate::fold::FoldAlphabet;
use crate::packing::{Coordinates, PackError, PackedStack};
use crate::params::BinaryPcsConfig;
use crate::pcs::BinaryPcs;
use crate::proof::BinaryPcsProof;
use crate::prover::BinaryPcsProverData;

/// A commitment to a function from the hypercube to `{0, 1}`.
///
/// The committed object is a bit witness.
/// An opening answers for its multilinear extension at a point of the challenge field.
pub trait BooleanMultilinearPcs<EF, Challenger> {
    /// Succinct binding commitment sent to the verifier.
    type Commitment;
    /// Prover-side data retained between commitment and opening.
    type ProverData;
    /// Opening proof checked by the verifier.
    type Proof;
    /// Why a commitment or an opening was refused.
    type Error;

    /// Variables the committed function has, so `2^n` bits in all.
    fn num_variables(&self) -> usize;

    /// Commit to a bit witness supplied bit-sliced, lane `j` of block `b` being bit `d*b + j`.
    ///
    /// # Errors
    ///
    /// Returns an error unless the witness covers exactly the committed hypercube.
    fn commit_bits<U: Underlier>(
        &self,
        bits: &[PackedGf2<U>],
        challenger: &mut Challenger,
    ) -> Result<(Self::Commitment, Self::ProverData), Self::Error>;

    /// Open the multilinear extension at every point, in one proof.
    ///
    /// No point needs prior transcript binding: each reduction binds its own.
    ///
    /// # Returns
    ///
    /// One value per point, in the order the points were supplied.
    ///
    /// # Errors
    ///
    /// Returns an error unless every point names the committed function's variables.
    fn open_at_points(
        &self,
        prover_data: Self::ProverData,
        points: &[Point<EF>],
        challenger: &mut Challenger,
    ) -> Result<(Vec<EF>, Self::Proof), Self::Error>;

    /// Check one proof against the values it claims at every point.
    ///
    /// # Errors
    ///
    /// Returns an error if a claim, a reduction, the commitment or the fold fails.
    fn verify_at_points(
        &self,
        commitment: &Self::Commitment,
        points: &[Point<EF>],
        values: &[EF],
        proof: &Self::Proof,
        challenger: &mut Challenger,
    ) -> Result<(), Self::Error>;
}

/// A Boolean commitment built from the packing, the bit ring switch and the binary PCS.
///
/// The committed elements hold `d` bits each.
/// The codeword is therefore `d` times shorter than an element-per-bit one.
pub struct BooleanPcs<EF: EncodableLevel, MT, MX> {
    /// The commitment the packed multilinear is discharged against.
    inner: BinaryPcs<EF, EF, MT, MX>,
    /// Variables the bit witness has, which is the packing's plus the absorbed ones.
    num_variables: usize,
}

impl<EF, MT, MX> BooleanPcs<EF, MT, MX>
where
    EF: EncodableLevel + TranscriptField + TowerLevel + FoldAlphabet<EF> + Coordinates,
    MT: Mmcs<EF>,
    MX: Mmcs<EF, Error = MT::Error>,
{
    /// Build a Boolean commitment over a bit witness of `num_variables` variables.
    ///
    /// # Errors
    ///
    /// Returns an error unless the schedule commits exactly the elements the packing holds.
    pub fn new(
        config: BinaryPcsConfig,
        mmcs: MT,
        round_mmcs: MX,
        num_variables: usize,
    ) -> Result<Self, BooleanPcsError<EF, MT::Error>> {
        let packed = Self::packed_variables(num_variables)?;
        if config.num_variables() != packed {
            return Err(BooleanPcsError::ConfigArity {
                expected: packed,
                actual: config.num_variables(),
            });
        }
        Ok(Self {
            inner: BinaryPcs::new(config, mmcs, round_mmcs),
            num_variables,
        })
    }

    /// Variables the packing keeps, the witness's less the ones one element absorbs.
    fn packed_variables(num_variables: usize) -> Result<usize, BooleanPcsError<EF, MT::Error>> {
        let absorbed = BitRingSwitch::<EF>::ABSORBED;
        num_variables
            .checked_sub(absorbed)
            .filter(|_| num_variables >= absorbed)
            .ok_or(BooleanPcsError::WitnessTooNarrow {
                needed: absorbed,
                actual: num_variables,
            })
    }

    /// Variables the committed bit witness has, so `2^n` bits in all.
    ///
    /// The same number the trait reports, reachable without naming a challenger.
    #[must_use]
    pub const fn num_variables(&self) -> usize {
        self.num_variables
    }

    /// The opening schedule a pool of this many claims is discharged through.
    ///
    /// One table of one column, opened directly at one point per claim.
    fn protocol(&self, num_claims: usize) -> OpeningProtocol {
        OpeningProtocol::new(vec![TableSpec::new(
            TableShape::new(self.inner.num_variables(), 1),
            (0..num_claims)
                .map(|_| OpeningBatch::new(vec![0], Vec::new()))
                .collect(),
        )])
    }

    /// Soundness cost of folding a pool of this many claims under one challenge.
    ///
    /// The commitment charges its own budget separately.
    /// The two compose by a union bound in whatever protocol holds both.
    #[must_use]
    pub fn batching_security(num_claims: usize) -> SecurityTerm {
        claim_pool_term(num_claims, EF::bits())
    }

    /// Check that every point names the committed function's variables.
    fn check_points(&self, points: &[Point<EF>]) -> Result<(), BooleanPcsError<EF, MT::Error>> {
        if points.is_empty() {
            return Err(BooleanPcsError::NoPoints);
        }
        for point in points {
            if point.num_variables() != self.num_variables {
                return Err(BooleanPcsError::PointArity {
                    expected: self.num_variables,
                    actual: point.num_variables(),
                });
            }
        }
        Ok(())
    }

    /// The packed multilinear the commitment holds, read back out of the retained table.
    fn packing(
        prover_data: &BinaryPcsProverData<EF, EF, MT>,
    ) -> Result<BitPacking<EF>, BooleanPcsError<EF, MT::Error>> {
        let packed = Poly::new(prover_data.table(0).poly(0).as_slice().to_vec());
        BitPacking::from_packed(packed).map_err(|_| BooleanPcsError::WitnessTooNarrow {
            needed: BitRingSwitch::<EF>::ABSORBED,
            actual: 0,
        })
    }
}

impl<EF, MT, MX, Challenger> BooleanMultilinearPcs<EF, Challenger> for BooleanPcs<EF, MT, MX>
where
    EF: EncodableLevel + TranscriptField + TowerLevel + FoldAlphabet<EF> + Coordinates,
    MT: Mmcs<EF>,
    MX: Mmcs<EF, Error = MT::Error>,
    Challenger: FieldChallenger<EF>
        + GrindingChallenger<Witness = EF>
        + CanSampleUniformBits<EF>
        + CanObserve<MT::Commitment>
        + CanObserve<MX::Commitment>,
{
    type Commitment = MT::Commitment;
    type ProverData = BinaryPcsProverData<EF, EF, MT>;
    type Proof = BooleanProof<EF, MT, MX>;
    type Error = BooleanPcsError<EF, MT::Error>;

    fn num_variables(&self) -> usize {
        Self::num_variables(self)
    }

    fn commit_bits<U: Underlier>(
        &self,
        bits: &[PackedGf2<U>],
        challenger: &mut Challenger,
    ) -> Result<(Self::Commitment, Self::ProverData), Self::Error> {
        // The packing is one copy of the bits, so the witness is never swept for arithmetic.
        let stack = PackedStack::<PackedGf2<U>, EF>::from_columns(&[bits])?;
        if stack.column_num_variables() != self.inner.num_variables() {
            return Err(BooleanPcsError::WitnessArity {
                expected: self.inner.num_variables(),
                actual: stack.column_num_variables(),
            });
        }

        let witness = SuffixProver::<EF, EF>::new_witness(vec![stack.into_table()], 0);
        self.inner
            .commit(witness, challenger)
            .map_err(BooleanPcsError::Commitment)
    }

    fn open_at_points(
        &self,
        prover_data: Self::ProverData,
        points: &[Point<EF>],
        challenger: &mut Challenger,
    ) -> Result<(Vec<EF>, Self::Proof), Self::Error> {
        self.check_points(points)?;
        let packing = Self::packing(&prover_data)?;

        // One reduction per point, each leaving its surviving claim in the shared pool.
        let mut pool = ClaimPool::new(self.inner.num_variables());
        let mut reductions = Vec::with_capacity(points.len());
        let mut values = Vec::with_capacity(points.len());
        let mut surviving_points = Vec::with_capacity(points.len());

        for point in points {
            // The element the reduction sends already holds the witness at the point.
            // Read by columns it is the claimed value, so it costs no pass of its own.
            let reduction = BitRingSwitch::new(point).map_err(BooleanPcsError::Reduction)?;
            let (proof, surviving_point, surviving_value) =
                prove_bit_ring_switch(&packing, point, challenger);

            values.push(reduction.incoming_claim(&proof.tensor));
            pool.deposit(surviving_point.clone(), surviving_value)
                .map_err(BooleanPcsError::Pool)?;
            surviving_points.push(surviving_point);
            reductions.push(proof);
        }

        // Sealing binds every surviving claim, then draws the one challenge.
        // The verifier seals the same pool, so the draw need not cross the wire.
        let _sealed = pool.seal(challenger);

        // Every surviving point came out of a reduction's rounds, so all are bound already.
        let opening = self
            .inner
            .try_open_at(
                prover_data,
                &self.protocol(points.len()),
                &surviving_points,
                challenger,
            )
            .map_err(BooleanPcsError::Commitment)?;

        Ok((
            values,
            BooleanProof {
                reductions,
                opening,
            },
        ))
    }

    fn verify_at_points(
        &self,
        commitment: &Self::Commitment,
        points: &[Point<EF>],
        values: &[EF],
        proof: &Self::Proof,
        challenger: &mut Challenger,
    ) -> Result<(), Self::Error> {
        self.check_points(points)?;
        if values.len() != points.len() || proof.reductions.len() != points.len() {
            return Err(BooleanPcsError::ClaimCount {
                expected: points.len(),
                values: values.len(),
                reductions: proof.reductions.len(),
            });
        }

        // Committing bound the root, so this side replays that binding before anything else.
        self.inner.observe_commitment(commitment, challenger);

        // Each reduction turns its claim about the bits into one about the packing.
        let mut pool = ClaimPool::new(self.inner.num_variables());
        let mut surviving_points = Vec::with_capacity(points.len());
        for ((point, &value), reduction) in points.iter().zip(values).zip(&proof.reductions) {
            let (surviving_point, surviving_value) =
                verify_bit_ring_switch(reduction, point, value, challenger)
                    .map_err(BooleanPcsError::ReductionProof)?;
            pool.deposit(surviving_point.clone(), surviving_value)
                .map_err(BooleanPcsError::Pool)?;
            surviving_points.push(surviving_point);
        }
        let sealed = pool.seal(challenger);

        // One commitment opening answers for every surviving point at once.
        let evals = self
            .inner
            .verify_at(
                commitment,
                &proof.opening,
                &self.protocol(points.len()),
                &surviving_points,
                challenger,
            )
            .map_err(BooleanPcsError::Commitment)?;

        // One fold closes the whole pool, rather than one equality per claim.
        let opened: Option<Vec<EF>> = evals
            .iter()
            .map(|batch| batch.current().first().copied())
            .collect();
        let opened = opened.ok_or(BooleanPcsError::SurvivingClaim)?;
        if opened.len() != points.len() || sealed.fold(opened) != sealed.folded_value() {
            return Err(BooleanPcsError::SurvivingClaim);
        }

        Ok(())
    }
}

/// One Boolean opening: the reductions, and the commitment opening that discharges them.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "EF: TowerLevel, MT::Commitment: Serialize, MT::MultiProof: Serialize, MX::Commitment: Serialize, MX::MultiProof: Serialize",
    deserialize = "EF: TowerLevel, MT::Commitment: Deserialize<'de>, MT::MultiProof: Deserialize<'de>, MX::Commitment: Deserialize<'de>, MX::MultiProof: Deserialize<'de>"
))]
pub struct BooleanProof<EF: Field, MT: Mmcs<EF>, MX: Mmcs<EF>> {
    /// One bit-alphabet ring switch per opening point, in the order the points came in.
    pub reductions: Vec<BitRingSwitchProof<EF>>,
    /// The single commitment opening that discharges every packed claim.
    pub opening: BinaryPcsProof<EF, EF, MT, MX>,
}

/// Why a Boolean commitment or opening was refused.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum BooleanPcsError<EF, MmcsError> {
    /// The witness is narrower than the coordinates one element absorbs.
    #[error("a bit witness of {actual} variables cannot absorb {needed} into one element")]
    WitnessTooNarrow {
        /// Coordinates one element absorbs.
        needed: usize,
        /// Variables the witness has.
        actual: usize,
    },

    /// The commitment schedule does not commit the elements the packing holds.
    #[error("the schedule commits {actual} variables, the packing holds {expected}")]
    ConfigArity {
        /// Variables the packing holds.
        expected: usize,
        /// Variables the schedule commits.
        actual: usize,
    },

    /// The witness does not cover the committed hypercube.
    #[error("the packed witness has {actual} variables, expected {expected}")]
    WitnessArity {
        /// Variables the commitment holds.
        expected: usize,
        /// Variables the witness packs to.
        actual: usize,
    },

    /// No opening point was supplied, so there is nothing to prove.
    #[error("an opening needs at least one point")]
    NoPoints,

    /// The claim counts on the two sides of an opening disagree.
    #[error("{expected} points against {values} values and {reductions} reductions")]
    ClaimCount {
        /// Points supplied.
        expected: usize,
        /// Values supplied.
        values: usize,
        /// Reductions the proof carries.
        reductions: usize,
    },

    /// A surviving claim could not join the pool.
    #[error(transparent)]
    Pool(ClaimPoolError),

    /// The opening point does not name the committed function's variables.
    #[error("the opening point names {actual} variables, expected {expected}")]
    PointArity {
        /// Variables the committed function has.
        expected: usize,
        /// Variables the point names.
        actual: usize,
    },

    /// The bits could not be packed into the elements a commitment holds.
    #[error(transparent)]
    Packing(#[from] PackError),

    /// The reduction could not be set up over the point supplied.
    #[error(transparent)]
    Reduction(p3_sumcheck::ring_switch::bits::BitRingSwitchError),

    /// The reduction's own proof was rejected.
    #[error(transparent)]
    ReductionProof(BitRingSwitchProofError),

    /// The commitment refused the opening.
    #[error(transparent)]
    Commitment(BinaryPcsError<EF, MmcsError>),

    /// The commitment opened a value the reduction did not leave behind.
    ///
    /// Without this the reduction would prove a claim about some other polynomial.
    #[error("the opened value is not the claim the reduction left behind")]
    SurvivingClaim,
}

#[cfg(test)]
mod tests {
    use p3_binary_field::{BinaryField128, Gf2, PackedGf2x64};
    use p3_field::PrimeCharacteristicRing;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::params::BinaryPcsParams;
    use crate::test_util::{MyMmcs, challenger, mmcs};

    type EF = BinaryField128;

    /// Blocks a fixture of the given log bit count holds.
    const fn blocks(log_bits: usize) -> usize {
        1 << (log_bits - 6)
    }

    /// A Boolean commitment over a witness of `log_bits` variables.
    fn boolean_pcs(log_bits: usize) -> BooleanPcs<EF, MyMmcs, MyMmcs> {
        let params = BinaryPcsParams {
            log_inv_rate: 2,
            pow_bits: 0,
            security_level: 40,
        };
        // The packing absorbs seven variables, so the commitment holds the rest.
        let config = BinaryPcsConfig::try_new::<EF, EF>(log_bits - 7, params).unwrap();
        BooleanPcs::new(config, mmcs(), mmcs(), log_bits).unwrap()
    }

    /// A random bit-sliced witness of the given log bit count.
    fn witness(seed: u64, log_bits: usize) -> Vec<PackedGf2x64> {
        let mut rng = SmallRng::seed_from_u64(seed);
        (0..blocks(log_bits))
            .map(|_| PackedGf2x64::new(rng.random::<u64>()))
            .collect()
    }

    /// The witness as a multilinear over every variable, one element per bit.
    ///
    /// The definition of what an opening answers for, with no packing in it.
    fn embedded(bits: &[PackedGf2x64]) -> Poly<EF> {
        Poly::new(
            bits.iter()
                .flat_map(|block| {
                    (0..PackedGf2x64::WIDTH).map(move |lane| {
                        if block.get(lane) == Gf2::ONE {
                            EF::ONE
                        } else {
                            EF::ZERO
                        }
                    })
                })
                .collect::<Vec<EF>>(),
        )
    }

    #[test]
    fn a_boolean_opening_round_trips_at_the_value_the_bits_define() {
        // Invariant: an opening answers for the bit witness, not for its packing.
        //
        // Fixture state: 2^13 bits, of which one element absorbs 2^7.
        //
        //     witness     128 blocks of 64 lanes
        //     commitment   64 elements, so six variables
        //
        // The value is compared against the one-element-per-bit multilinear.
        // That reference shares nothing with the packing or with the reduction.
        const LOG_BITS: usize = 13;

        let bits = witness(0xB001, LOG_BITS);
        let pcs = boolean_pcs(LOG_BITS);
        assert_eq!(pcs.num_variables(), LOG_BITS);

        let mut prover_chal = challenger();
        let (commitment, data) = pcs.commit_bits(&bits, &mut prover_chal).unwrap();

        let point = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0xB002), LOG_BITS);
        let points = alloc::vec![point.clone()];
        let (values, proof) = pcs.open_at_points(data, &points, &mut prover_chal).unwrap();

        assert_eq!(values, alloc::vec![embedded(&bits).eval_base(&point)]);

        let mut verifier_chal = challenger();
        pcs.verify_at_points(&commitment, &points, &values, &proof, &mut verifier_chal)
            .unwrap();
    }

    #[test]
    fn several_points_cost_one_commitment_opening() {
        // Invariant: every surviving claim is discharged by one opening.
        //
        //     - reductions   one per point, each binding its own point
        //     - pool         every surviving claim, sealed by one draw
        //     - opening      one, answering for all of them
        //
        // The fold is what ties them: one equality over the opened values, not four.
        const LOG_BITS: usize = 13;
        const NUM_POINTS: usize = 4;

        let bits = witness(0xB00F, LOG_BITS);
        let pcs = boolean_pcs(LOG_BITS);
        let mut rng = SmallRng::seed_from_u64(0xB010);
        let points: Vec<Point<EF>> = (0..NUM_POINTS)
            .map(|_| Point::<EF>::rand(&mut rng, LOG_BITS))
            .collect();

        let mut prover_chal = challenger();
        let (commitment, data) = pcs.commit_bits(&bits, &mut prover_chal).unwrap();
        let (values, proof) = pcs.open_at_points(data, &points, &mut prover_chal).unwrap();

        // One reduction per point, and exactly one commitment opening for all of them.
        assert_eq!(proof.reductions.len(), NUM_POINTS);

        // Each value is the bit witness at its own point.
        let reference = embedded(&bits);
        for (point, &value) in points.iter().zip(&values) {
            assert_eq!(value, reference.eval_base(point));
        }

        pcs.verify_at_points(&commitment, &points, &values, &proof, &mut challenger())
            .unwrap();

        // One wrong value among four is rejected, wherever it sits.
        for index in 0..NUM_POINTS {
            let mut tampered = values.clone();
            tampered[index] += EF::ONE;
            assert!(
                pcs.verify_at_points(&commitment, &points, &tampered, &proof, &mut challenger())
                    .is_err(),
                "value {index}"
            );
        }

        // A pool of four claims is charged its own labelled term.
        let term = BooleanPcs::<EF, MyMmcs, MyMmcs>::batching_security(NUM_POINTS);
        // Three is the degree of the difference polynomial four claims give.
        assert_eq!(
            term.bits.bits(),
            p3_security::claim_pool_error(NUM_POINTS, 128).bits()
        );
        assert!(term.bits.bits() > 126.0 && term.bits.bits() < 127.0);
    }

    #[test]
    fn a_false_value_is_rejected() {
        // The element's column reading is what ties the claim to the committed bits.
        const LOG_BITS: usize = 13;

        let bits = witness(0xFA15, LOG_BITS);
        let pcs = boolean_pcs(LOG_BITS);
        let mut prover_chal = challenger();
        let (commitment, data) = pcs.commit_bits(&bits, &mut prover_chal).unwrap();

        let points = alloc::vec![Point::<EF>::rand(
            &mut SmallRng::seed_from_u64(0xFA16),
            LOG_BITS
        )];
        let (values, proof) = pcs.open_at_points(data, &points, &mut prover_chal).unwrap();

        let false_values = alloc::vec![values[0] + EF::ONE];
        let err = pcs
            .verify_at_points(
                &commitment,
                &points,
                &false_values,
                &proof,
                &mut challenger(),
            )
            .unwrap_err();
        assert!(matches!(err, BooleanPcsError::ReductionProof(_)), "{err:?}");
    }

    #[test]
    fn an_opening_of_another_commitment_is_rejected() {
        // Invariant: the reduction alone proves nothing about what was committed.
        //
        // Mutation: keep the reduction, swap the commitment for another witness's.
        //
        //     reduction  -> a true claim about witness A's packing
        //     commitment -> witness B's root
        //
        // The surviving claim then answers for the wrong polynomial.
        const LOG_BITS: usize = 13;

        let pcs = boolean_pcs(LOG_BITS);
        let point = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0x0A11), LOG_BITS);

        let points = alloc::vec![point];
        let mut chal = challenger();
        let (_, data) = pcs
            .commit_bits(&witness(0xAAAA, LOG_BITS), &mut chal)
            .unwrap();
        let (values, proof) = pcs.open_at_points(data, &points, &mut chal).unwrap();

        let (other, _) = pcs
            .commit_bits(&witness(0xBBBB, LOG_BITS), &mut challenger())
            .unwrap();

        assert!(
            pcs.verify_at_points(&other, &points, &values, &proof, &mut challenger())
                .is_err()
        );
    }

    #[test]
    fn a_shape_the_commitment_does_not_hold_is_refused() {
        // Every mismatch is reported rather than asserted, because the shapes are inputs.
        let params = BinaryPcsParams {
            log_inv_rate: 2,
            pow_bits: 0,
            security_level: 40,
        };

        // A witness narrower than one element absorbs has no packing at all.
        let config = BinaryPcsConfig::try_new::<EF, EF>(6, params).unwrap();
        let narrow_witness = BooleanPcs::<EF, MyMmcs, MyMmcs>::new(config, mmcs(), mmcs(), 4).err();
        assert!(matches!(
            narrow_witness,
            Some(BooleanPcsError::WitnessTooNarrow {
                needed: 7,
                actual: 4
            })
        ));

        // A schedule committing a different arity than the packing holds is refused.
        let wrong_arity = BooleanPcs::<EF, MyMmcs, MyMmcs>::new(config, mmcs(), mmcs(), 14).err();
        assert!(matches!(
            wrong_arity,
            Some(BooleanPcsError::ConfigArity {
                expected: 7,
                actual: 6
            })
        ));

        // A point of the wrong width reaches neither the reduction nor the commitment.
        let pcs = boolean_pcs(13);
        let mut chal = challenger();
        let (commitment, data) = pcs.commit_bits(&witness(0x5417, 13), &mut chal).unwrap();
        let narrow = alloc::vec![Point::<EF>::rand(&mut SmallRng::seed_from_u64(0x5418), 12)];
        assert!(matches!(
            pcs.open_at_points(data, &narrow, &mut chal).err(),
            Some(BooleanPcsError::PointArity {
                expected: 13,
                actual: 12
            })
        ));
        assert!(matches!(
            pcs.verify_at_points(&commitment, &narrow, &[EF::ZERO], &proof_stub(), &mut chal),
            Err(BooleanPcsError::PointArity { .. })
        ));

        // No point at all describes nothing to prove.
        let (_, data) = pcs.commit_bits(&witness(0x5419, 13), &mut chal).unwrap();
        assert!(matches!(
            pcs.open_at_points(data, &[], &mut chal).err(),
            Some(BooleanPcsError::NoPoints)
        ));
    }

    /// A proof the width check never looks past.
    fn proof_stub() -> BooleanProof<EF, MyMmcs, MyMmcs> {
        let bits = witness(0x57AB, 13);
        let pcs = boolean_pcs(13);
        let mut chal = challenger();
        let (_, data) = pcs.commit_bits(&bits, &mut chal).unwrap();
        let points = alloc::vec![Point::<EF>::rand(&mut SmallRng::seed_from_u64(0x57AC), 13)];
        pcs.open_at_points(data, &points, &mut chal).unwrap().1
    }

    proptest! {
        // Each case runs a whole commitment and opening on both sides.
        #![proptest_config(ProptestConfig { cases: 8, ..ProptestConfig::default() })]

        /// Every witness height the fixture admits, at random points each time.
        #[test]
        fn a_boolean_opening_round_trips_over_random_inputs(
            log_bits in 13usize..=15,
            num_points in 1usize..=3,
            witness_seed: u64,
            point_seed: u64,
        ) {
            let bits = witness(witness_seed, log_bits);
            let pcs = boolean_pcs(log_bits);

            let mut prover_chal = challenger();
            let (commitment, data) = pcs.commit_bits(&bits, &mut prover_chal).unwrap();
            let mut rng = SmallRng::seed_from_u64(point_seed);
            let points: Vec<Point<EF>> = (0..num_points)
                .map(|_| Point::<EF>::rand(&mut rng, log_bits))
                .collect();
            let (values, proof) = pcs.open_at_points(data, &points, &mut prover_chal).unwrap();

            let reference = embedded(&bits);
            for (point, &value) in points.iter().zip(&values) {
                prop_assert_eq!(value, reference.eval_base(point));
            }
            prop_assert!(
                pcs.verify_at_points(&commitment, &points, &values, &proof, &mut challenger())
                    .is_ok()
            );
        }
    }
}
