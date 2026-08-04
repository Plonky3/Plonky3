//! Provers for the stacked layout, specialised per residual-sumcheck binding mode.
//!
//! # Modules
//!
//! - Prefix prover: SVO-accumulator preprocessing, packed handoff.
//! - Suffix prover: SVO-accumulator preprocessing, unpacked handoff.

mod claims;
mod prefix;
mod suffix;
#[cfg(test)]
pub(super) mod test_utils;
#[cfg(test)]
mod tests;

use alloc::vec::Vec;

pub use claims::StackedClaims;
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, TwoAdicField};
use p3_matrix::dense::DenseMatrix;
use p3_multilinear_util::point::Point;
pub use prefix::PrefixProver;
pub use suffix::SuffixProver;

use crate::SumcheckData;
use crate::commit::commit_base;
use crate::layout::{LayoutStrategy, Table, Witness};
use crate::strategy::{SumcheckProver, VariableOrder};
use crate::table::{OpeningEvals, OpeningRequest};

/// Stacked-sumcheck prover layout
pub trait Layout<F: TwoAdicField, EF: ExtensionField<F>>: Sized {
    /// Builds this layout from a committed witness.
    fn from_witness(witness: Witness<F>) -> Self;

    /// Builds a witness structure for this layout from source tables.
    fn new_witness(tables: Vec<Table<F>>, folding: usize) -> Witness<F>;

    /// Returns the shared claim state recorded against the stacked polynomial.
    fn claims(&self) -> &StackedClaims<F, EF>;

    /// Commits to the witness and returns the layout.
    ///
    /// # Arguments
    ///
    /// - `dft`                    — base-field DFT used to encode the codeword.
    /// - `mmcs`                   — Merkle commitment scheme over the base field.
    /// - `challenger`             — Fiat–Shamir transcript; absorbs the Merkle root.
    /// - `witness`                — stacked committed polynomial plus its tables.
    /// - `folding`                — folding factor consumed by the first WHIR round.
    /// - `starting_log_inv_rate`  — initial log-inverse rate of the RS code.
    fn commit<Dft, MT, Challenger>(
        dft: &Dft,
        mmcs: &MT,
        challenger: &mut Challenger,
        witness: Witness<F>,
        folding: usize,
        starting_log_inv_rate: usize,
    ) -> (Self, MT::Commitment, MT::ProverData<DenseMatrix<F>>)
    where
        Dft: TwoAdicSubgroupDft<F>,
        MT: Mmcs<F>,
        Challenger: CanObserve<MT::Commitment>,
    {
        // Encode and Merkle-commit the stacked polynomial in the mode's variable order.
        let (root, prover_data) = commit_base(
            Self::variable_order(),
            dft,
            mmcs,
            challenger,
            &witness.poly,
            folding,
            starting_log_inv_rate,
        );

        // The witness is consumed into the layout once its codeword is committed.
        (Self::from_witness(witness), root, prover_data)
    }

    /// Returns the total number of concrete openings recorded so far.
    fn num_claims(&self) -> usize {
        self.claims().num_claims()
    }

    /// Returns the verifier strategy required to replay this committed layout.
    fn strategy() -> LayoutStrategy;

    /// Returns the variable order.
    fn variable_order() -> VariableOrder {
        Self::strategy().variable_order
    }

    /// Returns the number of variables of first round
    fn folding(&self) -> usize {
        self.claims().folding()
    }

    /// Returns the number of variables of the stacked polynomial.
    fn num_variables(&self) -> usize {
        self.claims().num_variables()
    }

    /// Returns the number of variables of table `id`.
    fn num_variables_table(&self, id: usize) -> usize {
        self.claims().num_variables_table(id)
    }

    /// Returns source table `id`.
    fn table(&self, id: usize) -> &Table<F> {
        self.claims().table(id)
    }

    /// Records opening claims for the selected columns of one table at a sampled point.
    ///
    /// - The local-frame opening point is drawn from the transcript.
    /// - Current openings evaluate a column at that point.
    /// - Next openings evaluate the repeat-last successor view at the same point.
    /// - Returned evaluations list all current openings first.
    /// - Returned evaluations list all next openings second.
    fn eval<Ch>(
        &mut self,
        table_idx: usize,
        batch: &OpeningRequest,
        challenger: &mut Ch,
    ) -> OpeningEvals<EF>
    where
        Ch: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // Draw the local-frame opening point as powers of one challenge.
        // This is the standalone-PCS convention: the verifier picks the evaluation point.
        let point = Point::expand_from_univariate(
            challenger.sample_algebra_element(),
            self.num_variables_table(table_idx),
        );
        self.eval_at(table_idx, batch, &point, challenger)
    }

    /// Records opening claims for the selected columns of one table at a prescribed point.
    ///
    /// The caller supplies the local-frame opening point instead of sampling it.
    /// An outer protocol that fixes the point opens its columns here.
    ///
    /// Soundness requires `point` to be sampled from, or bound to, the same `challenger`
    /// before this call (see `PrescribedPointPcs`'s Fiat-Shamir/Soundness doc) — this
    /// method absorbs the evaluations but not the point itself.
    ///
    /// - Current openings evaluate a column at the supplied point.
    /// - Next openings evaluate the repeat-last successor view at the same point.
    /// - The claimed evaluations are absorbed into the transcript.
    /// - The current group is absorbed first.
    /// - Returned evaluations list all current openings first.
    /// - Returned evaluations list all next openings second.
    ///
    /// # Arguments
    ///
    /// - Index of the table whose columns are opened.
    /// - Column indices opened directly and through the successor view.
    /// - Local-frame opening point.
    /// - Fiat-Shamir transcript.
    fn eval_at<Ch>(
        &mut self,
        table_idx: usize,
        batch: &OpeningRequest,
        point: &Point<EF>,
        challenger: &mut Ch,
    ) -> OpeningEvals<EF>
    where
        Ch: FieldChallenger<F> + GrindingChallenger<Witness = F>;

    /// Samples a virtual evaluation on the full stacked polynomial.
    fn add_virtual_eval<Ch>(&mut self, challenger: &mut Ch) -> EF
    where
        Ch: FieldChallenger<F> + GrindingChallenger<Witness = F>;

    /// Processes initial rounds of sumcheck and returns the residual sumcheck prover.
    ///
    /// # Returns
    ///
    /// - Residual sumcheck prover over the unpacked product polynomial.
    /// - Folding challenges sampled during preprocessing.
    fn into_sumcheck<Ch>(
        self,
        sumcheck_data: &mut SumcheckData<F, EF>,
        pow_bits: usize,
        challenger: &mut Ch,
    ) -> (SumcheckProver<F, EF>, Point<EF>)
    where
        Ch: FieldChallenger<F> + GrindingChallenger<Witness = F>;
}
