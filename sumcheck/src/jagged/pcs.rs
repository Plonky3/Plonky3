//! Discharging a sparse evaluation claim against a dense multilinear commitment.
//!
//! The reduction on its own leaves a claim the caller is trusted to answer.
//!
//! Here that answer is the same call, so an accepted sparse claim is an authenticated one.

use alloc::vec;
use alloc::vec::Vec;
use core::slice;

use p3_challenger::fs::{DomainSeparator, FieldUnit, InteractionPattern, TranscriptField};
use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field};
use p3_multilinear_util::point::Point;
use serde::{Deserialize, Serialize};

use super::error::JaggedOpeningError;
use super::layout::JaggedLayout;
use super::transcript::encode_layout;
use super::{JaggedPoint, JaggedProof};
use crate::prescribed_pcs::PrescribedPointPcs;
use crate::table::{OpeningBatch, OpeningProtocol, TableShape, TableSpec};

/// Version bound into the geometry seal.
const VERSION: u8 = 1;

/// Protocol name separating the geometry seal from every other absorption.
const NAME: &[u8] = b"p3-sumcheck-jagged-layout";

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

/// The shape a commitment to one jagged geometry must be opened under.
///
/// One column of the dense arity, read at one point.
///
/// A caller needs this before it commits, which is before any transcript is sealed.
impl From<&JaggedLayout> for OpeningProtocol {
    fn from(layout: &JaggedLayout) -> Self {
        Self::new(vec![TableSpec::new(
            TableShape::new(layout.dense_variables(), 1),
            vec![OpeningBatch::new(vec![0], Vec::new())],
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

    /// Proves a sparse evaluation and opens the dense commitment that answers it.
    ///
    /// The witness is the committed vector, so the opening and the reduction speak about one object.
    ///
    /// # Errors
    ///
    /// - The reduction refuses the caller's statement or witness.
    /// - The commitment scheme refuses the opening.
    pub fn open<F, EF, Pcs, Challenger>(
        &self,
        pcs: &Pcs,
        prover_data: Pcs::ProverData,
        dense_witness: &[F],
        point: &JaggedPoint<EF>,
        claimed_value: EF,
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
        let (reduction, claim) = self
            .0
            .prove(dense_witness, point, claimed_value, challenger)?
            .into_parts();

        // The reduction fixed the point, so the opening cannot be moved anywhere else.
        let dense = pcs
            .open_at(
                prover_data,
                &OpeningProtocol::from(self.0),
                slice::from_ref(claim.point()),
                challenger,
            )
            .map_err(JaggedOpeningError::Commitment)?;

        Ok(JaggedOpening { reduction, dense })
    }

    /// Verifies a sparse evaluation against the commitment that authenticates it.
    ///
    /// # Soundness
    ///
    /// The reduction alone is satisfiable for any sparse value, because the prover supplies one unknown.
    ///
    /// The opening below removes that freedom by pinning the unknown to the committed vector.
    ///
    /// The whole error is the reduction's own term plus whatever the commitment scheme charges.
    ///
    /// # Errors
    ///
    /// - The reduction rejects.
    /// - The commitment scheme rejects the opening.
    /// - The opening has a shape the reduction did not ask for.
    /// - The committed vector does not take the reduced value at the reduced point.
    pub fn verify<F, EF, Pcs, Challenger>(
        &self,
        pcs: &Pcs,
        commitment: &Pcs::Commitment,
        opening: &JaggedOpening<F, EF, Pcs::Proof>,
        point: &JaggedPoint<EF>,
        claimed_value: EF,
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
        let claim = self
            .0
            .verify(point, claimed_value, &opening.reduction, challenger)?;

        let opened = pcs
            .verify_at(
                commitment,
                &opening.dense,
                &OpeningProtocol::from(self.0),
                slice::from_ref(claim.point()),
                challenger,
            )
            .map_err(JaggedOpeningError::Commitment)?;

        // One batch of one direct reading is the only shape the protocol above describes.
        let [batch] = opened.as_slice() else {
            return Err(JaggedOpeningError::OpeningShape {
                batches: opened.len(),
                direct: 0,
                successor: 0,
            });
        };
        if batch.current().len() != 1 || !batch.next().is_empty() {
            return Err(JaggedOpeningError::OpeningShape {
                batches: 1,
                direct: batch.current().len(),
                successor: batch.next().len(),
            });
        }

        // Discharging the reduction is this one comparison, and skipping it accepts everything.
        if batch.current()[0] != *claim.value() {
            return Err(JaggedOpeningError::DenseMismatch);
        }

        Ok(())
    }
}

/// A sparse evaluation reduced to a dense claim, and the opening that answers it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JaggedOpening<F, EF, Proof> {
    /// Reduction from the sparse statement to one dense evaluation.
    reduction: JaggedProof<F, EF>,
    /// Opening of the committed vector at the point the reduction produced.
    dense: Proof,
}

impl<F, EF, Proof> JaggedOpening<F, EF, Proof> {
    /// Returns the sparse-to-dense reduction.
    #[must_use]
    pub const fn reduction(&self) -> &JaggedProof<F, EF> {
        &self.reduction
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
    fn the_dense_opening_reads_one_column_at_one_point() {
        let layout = JaggedLayout::new(3, &[3, 0, 5, 1]).unwrap();
        let protocol = OpeningProtocol::from(&layout);

        assert_eq!(protocol.table_shapes(), vec![TableShape::new(4, 1)]);
        assert_eq!(protocol.num_openings(), 1);

        let (table, batch) = protocol.iter_openings().next().unwrap();
        assert_eq!(table, 0);
        assert_eq!(batch.current(), &[0]);
        assert!(batch.next().is_empty());
    }
}
