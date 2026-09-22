//! Discharging a sparse evaluation claim against a dense multilinear commitment.
//!
//! The reduction on its own leaves a claim the caller is trusted to answer.
//!
//! Here that answer is the same call, so an accepted sparse claim is an authenticated one.

use alloc::vec;
use alloc::vec::Vec;

use p3_challenger::fs::{DomainSeparator, FieldUnit, InteractionPattern, TranscriptField};
use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field};
use p3_multilinear_util::point::Point;
use p3_security::{ErrorBits, SecurityTerm};
use p3_util::log2_ceil_usize;
use serde::{Deserialize, Serialize};

use super::error::JaggedOpeningError;
use super::layout::JaggedLayout;
use super::transcript::encode_layout;
use super::{JaggedPoint, JaggedProof};
use crate::prescribed_pcs::{PrescribedOpeningSecurity, PrescribedPointPcs};
use crate::table::{OpeningBatch, OpeningProtocol, TableShape, TableSpec};

/// Version bound into the geometry seal.
const VERSION: u8 = 1;

/// Protocol name separating the geometry seal from every other absorption.
const NAME: &[u8] = b"p3-sumcheck-jagged-layout";

/// Label the composed report files the jagged reductions under.
const REDUCTION_LABEL: &str = "jagged sparse-to-dense reduction";

/// A geometry a transcript has been sealed to.
///
/// Every later challenge depends on the column heights, so no other heights answer the same point.
///
/// Reaching the reduction only through this handle makes that ordering a property of the code.
///
/// The commitment must already be bound, by the proving side on production and by the other on absorption.
#[must_use]
#[derive(Clone, Copy, Debug)]
pub struct BoundJaggedLayout<'a>(&'a JaggedLayout);

/// One geometry and the number of sparse claims one commitment to it answers.
///
/// A caller sizes its commitment from this before it commits, which is before any seal exists.
#[derive(Clone, Copy, Debug)]
pub struct JaggedOpeningShape<'a> {
    /// Geometry whose envelope the commitment binds.
    layout: &'a JaggedLayout,
    /// Sparse claims the one opening discharges.
    claims: usize,
}

impl<'a> JaggedOpeningShape<'a> {
    /// Names one geometry and how many sparse claims one opening of it answers.
    #[must_use]
    pub const fn new(layout: &'a JaggedLayout, claims: usize) -> Self {
        Self { layout, claims }
    }
}

/// The shape a commitment to one jagged geometry must be opened under.
///
/// One column of the dense arity, read once per sparse claim.
impl From<JaggedOpeningShape<'_>> for OpeningProtocol {
    fn from(shape: JaggedOpeningShape<'_>) -> Self {
        Self::new(vec![TableSpec::new(
            TableShape::new(shape.layout.dense_variables(), 1),
            vec![OpeningBatch::new(vec![0], Vec::new()); shape.claims],
        )])
    }
}

impl<'a> BoundJaggedLayout<'a> {
    /// Seals a transcript to one geometry.
    ///
    /// Both sides absorb the same bytes at the same position, so neither can drift from the other.
    pub fn new<F, Challenger>(layout: &'a JaggedLayout, challenger: &mut Challenger) -> Self
    where
        F: TranscriptField,
        Challenger: CanObserve<F>,
    {
        // No message travels here, so the description has no step and the whole binding is the seed.
        let pattern = InteractionPattern::new(Vec::new())
            .expect("an empty description is structurally valid");
        let mut separator = DomainSeparator::<FieldUnit<F>>::new(VERSION, NAME, pattern);
        separator.instance(&encode_layout(layout));
        separator.seed(challenger);

        Self(layout)
    }

    /// Returns the sealed geometry.
    #[must_use]
    pub const fn layout(&self) -> &'a JaggedLayout {
        self.0
    }

    /// Draws a sparse evaluation point for this geometry.
    ///
    /// The seal precedes the draw by construction, which is the ordering the reduction rests on.
    pub fn sample_point<F, EF, Challenger>(&self, challenger: &mut Challenger) -> JaggedPoint<EF>
    where
        F: Field,
        EF: ExtensionField<F>,
        Challenger: FieldChallenger<F>,
    {
        let mut draw = |count| {
            Point::new(
                (0..count)
                    .map(|_| challenger.sample_algebra_element::<EF>())
                    .collect(),
            )
        };

        // Row coordinates come first on both sides.
        let row = draw(self.0.row_variables());
        let column = draw(self.0.column_variables());
        JaggedPoint::new(row, column)
    }

    /// Proves several sparse evaluations and opens the dense commitment that answers them all.
    ///
    /// The witness is the committed vector, so the opening and the reductions speak about one object.
    ///
    /// A trace is read at more than one point in practice, at a zerocheck point and at its successor.
    ///
    /// Those claims share one commitment and one opening rather than taking one each.
    ///
    /// Claims are discharged in the order given, and both sides must present that same order.
    ///
    /// # Errors
    ///
    /// - No claim was supplied.
    /// - A reduction refuses the caller's statement or witness.
    /// - The commitment scheme refuses the opening.
    pub fn open<F, EF, Pcs, Challenger>(
        &self,
        pcs: &Pcs,
        prover_data: Pcs::ProverData,
        dense_witness: &[F],
        claims: &[(JaggedPoint<EF>, EF)],
        challenger: &mut Challenger,
    ) -> Result<JaggedOpening<F, EF, Pcs::Proof>, JaggedOpeningError<Pcs::ProverError>>
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
        Pcs: PrescribedPointPcs<EF, Challenger, Val = F>,
        Challenger: FieldChallenger<F>
            + GrindingChallenger<Witness = F>
            + CanSampleUniformBits<F>
            + CanObserve<Pcs::Commitment>,
    {
        if claims.is_empty() {
            return Err(JaggedOpeningError::NoClaims);
        }

        // Every reduction runs against the same committed vector, in the order the caller gave.
        let mut reductions = Vec::with_capacity(claims.len());
        let mut points = Vec::with_capacity(claims.len());
        for (point, value) in claims {
            let (reduction, claim) = self
                .0
                .prove(dense_witness, point, *value, challenger)?
                .into_parts();
            reductions.push(reduction);
            points.push(claim.point().clone());
        }

        // The reductions fixed the points, so the opening cannot be moved anywhere else.
        let dense = pcs
            .open_at(
                prover_data,
                &OpeningProtocol::from(JaggedOpeningShape::new(self.0, claims.len())),
                &points,
                challenger,
            )
            .map_err(JaggedOpeningError::Commitment)?;

        Ok(JaggedOpening { reductions, dense })
    }

    /// Returns the composed soundness evidence for discharging a number of sparse claims here.
    ///
    /// # Returns
    ///
    /// Nothing for no claims, and nothing when the scheme does not assess this opening protocol.
    ///
    /// # Soundness
    ///
    /// The commitment fixes a candidate set before anything below is drawn, and the scheme reports it.
    ///
    /// Of what this layer draws next, only the sumcheck challenges carry a claim of their own.
    ///
    /// That claim is the one charged over the candidate set here.
    ///
    /// The sparse point is a statement rather than a test, because the caller supplies its value.
    ///
    /// A caller whose own statement rests on that point being unpredictable charges the draw itself.
    ///
    /// The geometry seal absorbs and draws nothing, so it is free.
    ///
    /// The scheme's own draws are already inside the evidence this starts from.
    #[must_use]
    pub fn security<EF, Pcs, Challenger>(
        &self,
        pcs: &Pcs,
        claims: usize,
    ) -> Option<PrescribedOpeningSecurity>
    where
        EF: ExtensionField<Pcs::Val>,
        Pcs: PrescribedPointPcs<EF, Challenger>,
        Challenger: FieldChallenger<Pcs::Val>
            + GrindingChallenger<Witness = Pcs::Val>
            + CanSampleUniformBits<Pcs::Val>
            + CanObserve<Pcs::Commitment>,
    {
        // An empty statement is refused when it is opened, so no figure describes it.
        if claims == 0 {
            return None;
        }

        let protocol = OpeningProtocol::from(JaggedOpeningShape::new(self.0, claims));
        let mut security = pcs.prescribed_security(&protocol)?;
        security.charge_reduction(reduction_term::<EF>(self.0.dense_variables(), claims));
        Some(security)
    }

    /// Verifies several sparse evaluations against the commitment that authenticates them.
    ///
    /// # Soundness
    ///
    /// A reduction alone is satisfiable for any sparse value, because the prover supplies one unknown.
    ///
    /// The opening below removes that freedom by pinning each unknown to the committed vector.
    ///
    /// The whole error is what the scheme charges, plus the reductions charged over its candidate set.
    ///
    /// That composed figure is what this handle reports.
    ///
    /// # Errors
    ///
    /// - No claim was supplied.
    /// - A reduction rejects.
    /// - The commitment scheme rejects the opening.
    /// - The opening returns a different number of readings than there are claims.
    /// - One reading is not the single direct value a claim asks for.
    /// - The committed vector does not take a reduced value at its reduced point.
    pub fn verify<F, EF, Pcs, Challenger>(
        &self,
        pcs: &Pcs,
        commitment: &Pcs::Commitment,
        opening: &JaggedOpening<F, EF, Pcs::Proof>,
        claims: &[(JaggedPoint<EF>, EF)],
        challenger: &mut Challenger,
    ) -> Result<(), JaggedOpeningError<Pcs::Error>>
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
        Pcs: PrescribedPointPcs<EF, Challenger, Val = F>,
        Challenger: FieldChallenger<F>
            + GrindingChallenger<Witness = F>
            + CanSampleUniformBits<F>
            + CanObserve<Pcs::Commitment>,
    {
        if claims.is_empty() {
            return Err(JaggedOpeningError::NoClaims);
        }

        // A proof carrying a different number of reductions describes another statement.
        if opening.reductions.len() != claims.len() {
            return Err(JaggedOpeningError::ReductionCountMismatch {
                expected: claims.len(),
                actual: opening.reductions.len(),
            });
        }

        // Replaying every reduction before the opening keeps both sides in one transcript order.
        let mut values = Vec::with_capacity(claims.len());
        let mut points = Vec::with_capacity(claims.len());
        for ((point, value), reduction) in claims.iter().zip(&opening.reductions) {
            let claim = self.0.verify(point, *value, reduction, challenger)?;
            values.push(*claim.value());
            points.push(claim.point().clone());
        }

        let opened = pcs
            .verify_at(
                commitment,
                &opening.dense,
                &OpeningProtocol::from(JaggedOpeningShape::new(self.0, claims.len())),
                &points,
                challenger,
            )
            .map_err(JaggedOpeningError::Commitment)?;

        // One reading per claim is the only shape the protocol above describes.
        if opened.len() != claims.len() {
            return Err(JaggedOpeningError::OpeningCountMismatch {
                expected: claims.len(),
                actual: opened.len(),
            });
        }
        for (index, batch) in opened.iter().enumerate() {
            if batch.current().len() != 1 || !batch.next().is_empty() {
                return Err(JaggedOpeningError::OpeningShape {
                    reading: index,
                    direct: batch.current().len(),
                    successor: batch.next().len(),
                });
            }

            // Discharging a reduction is this one comparison, and skipping it accepts everything.
            if batch.current()[0] != values[index] {
                return Err(JaggedOpeningError::DenseMismatch { reading: index });
            }
        }

        Ok(())
    }
}

/// The soundness term a number of jagged reductions over one envelope charges.
///
/// The paper bounds one reduction over an envelope of `m` variables by `2m` field inverses.
///
/// A union bound over independent claims multiplies that count.
///
/// The field width is taken one bit short of the order, because the order need not be a power of two.
///
/// The union-bound degree is rounded up to the next power of two, which can only lower the bound.
fn reduction_term<EF: Field>(dense_variables: usize, claims: usize) -> SecurityTerm {
    // No variable leaves no challenge to guess, so such a reduction has nothing to charge.
    let error = if dense_variables == 0 || claims == 0 {
        ErrorBits::from_log2(f64::INFINITY)
    } else {
        let degree = dense_variables.saturating_mul(claims).saturating_mul(2);
        let bits = EF::bits().saturating_sub(1) as f64 - log2_ceil_usize(degree) as f64;
        ErrorBits::from_log2(bits)
    };
    SecurityTerm::new(REDUCTION_LABEL, error)
}

/// Sparse evaluations reduced to dense claims, and the one opening that answers them.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JaggedOpening<F, EF, Proof> {
    /// One reduction per sparse claim, in the order the claims were given.
    reductions: Vec<JaggedProof<F, EF>>,
    /// Opening of the committed vector at every point the reductions produced.
    dense: Proof,
}

impl<F, EF, Proof> JaggedOpening<F, EF, Proof> {
    /// Returns one sparse-to-dense reduction per claim.
    #[must_use]
    pub fn reductions(&self) -> &[JaggedProof<F, EF>] {
        &self.reductions
    }

    /// Returns the opening of the committed vector.
    #[must_use]
    pub const fn dense(&self) -> &Proof {
        &self.dense
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{EF, F, challenger};

    // The first coordinate drawn after a seal is the cheapest witness that the seal moved the sponge.
    fn first_coordinate_after_binding(layout: &JaggedLayout) -> EF {
        let mut challenger = challenger();
        let bound = BoundJaggedLayout::new::<F, _>(layout, &mut challenger);
        bound.sample_point::<F, EF, _>(&mut challenger).row()[0]
    }

    #[test]
    fn sealing_a_transcript_to_one_geometry_excludes_every_other() {
        // Fixture state: nine live cells over four columns, under a three-variable row bound.
        let layout = JaggedLayout::new(3, &[3, 0, 5, 1]).unwrap();
        let baseline = first_coordinate_after_binding(&layout);
        assert_eq!(baseline, first_coordinate_after_binding(&layout));

        // Mutation: one live row moves between two columns, leaving the area and the arity alone.
        let moved = JaggedLayout::new(3, &[2, 1, 5, 1]).unwrap();
        assert_eq!(moved.dense_variables(), layout.dense_variables());
        assert_eq!(moved.area(), layout.area());
        assert_ne!(baseline, first_coordinate_after_binding(&moved));

        // Mutation: the row bound alone, which the heights never mention.
        let taller = JaggedLayout::new(4, &[3, 0, 5, 1]).unwrap();
        assert_ne!(baseline, first_coordinate_after_binding(&taller));
    }

    #[test]
    fn a_drawn_point_has_the_coordinates_the_geometry_addresses() {
        let layout = JaggedLayout::new(5, &[3, 0, 5, 1]).unwrap();
        let mut challenger = challenger();
        let point = BoundJaggedLayout::new::<F, _>(&layout, &mut challenger)
            .sample_point::<F, EF, _>(&mut challenger);

        assert_eq!(point.row().num_variables(), 5);
        assert_eq!(point.column().num_variables(), 2);

        // Two independent draws must not repeat, or the row and column parts would be tied together.
        assert_ne!(point.row()[0], point.column()[0]);
    }

    #[test]
    fn the_dense_opening_reads_one_column_once_per_claim() {
        // Nine live cells sit in a sixteen-cell envelope, which is four variables of one column.
        let layout = JaggedLayout::new(3, &[3, 0, 5, 1]).unwrap();
        let protocol = OpeningProtocol::from(JaggedOpeningShape::new(&layout, 3));

        assert_eq!(protocol.table_shapes(), vec![TableShape::new(4, 1)]);
        assert_eq!(protocol.num_openings(), 3);

        for (table, batch) in protocol.iter_openings() {
            assert_eq!(table, 0);
            assert_eq!(batch.current(), &[0]);
            assert!(batch.next().is_empty());
        }
    }

    #[test]
    fn the_reduction_term_prices_every_challenge_the_claims_draw() {
        // The extension is four thirty-one-bit limbs, and the width is taken one bit short.
        let field_bits = (EF::bits() - 1) as f64;

        // Four envelope variables and one claim draw eight guesses, which is three bits.
        assert_eq!(
            reduction_term::<EF>(4, 1).bits.bits(),
            field_bits - 3.0,
            "one claim over four variables"
        );

        // Four claims over the same envelope draw four times as many, which is two bits more.
        assert_eq!(
            reduction_term::<EF>(4, 4).bits.bits(),
            field_bits - 5.0,
            "four claims over four variables"
        );

        // An envelope of one cell has no round to challenge, so there is nothing to charge.
        assert!(reduction_term::<EF>(0, 1).bits.bits().is_infinite());
    }
}
