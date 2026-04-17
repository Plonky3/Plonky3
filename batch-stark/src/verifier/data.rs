use core::fmt::Debug;

use p3_air::{Air, RowWindow};
use p3_commit::PolynomialSpace;
use p3_field::PrimeCharacteristicRing;
use p3_lookup::folder::VerifierConstraintFolderWithLookups;
use p3_lookup::{Lookup, LookupProtocol};
use p3_matrix::dense::RowMajorMatrixView;
use p3_matrix::stack::VerticalPair;
use p3_uni_stark::{VerificationError, VerifierConstraintFolder};

use crate::config::{Domain, StarkGenericConfig as SGC, Val};

/// Structure storing all data needed for verifying one instance's constraints at the out-of-domain point.
pub struct VerifierData<'a, SC: SGC> {
    /// Out-of-domain point at which constraints are evaluated.
    pub(crate) zeta: SC::Challenge,
    /// Challenge used to fold constraints.
    pub(crate) alpha: SC::Challenge,
    /// Main trace evaluated at `zeta`.
    pub(crate) trace_local: &'a [SC::Challenge],
    /// Main trace evaluated at the following point `g * zeta`, where `g` is the subgroup generator.
    pub(crate) trace_next: &'a [SC::Challenge],
    /// Preprocessed trace evaluated at `zeta`.
    pub(crate) preprocessed_local: &'a [SC::Challenge],
    /// Preprocessed trace evaluated at the following point `g * zeta`, where `g` is the subgroup generator.
    pub(crate) preprocessed_next: &'a [SC::Challenge],
    /// Permutation trace evaluated at `zeta`.
    pub(crate) permutation_local: &'a [SC::Challenge],
    /// Permutation trace evaluated at the following point `g * zeta`, where `g` is the subgroup generator.
    pub(crate) permutation_next: &'a [SC::Challenge],
    /// Challenges used for the lookup argument.
    pub(crate) permutation_challenges: &'a [SC::Challenge],
    /// Expected cumulated values for global lookup arguments.
    pub(crate) permutation_values: &'a [SC::Challenge],
    /// Periodic column polynomials evaluated at the OOD point `zeta`.
    pub(crate) periodic_values: &'a [SC::Challenge],
    /// Lookup contexts for this instance.
    pub(crate) lookups: &'a [Lookup<Val<SC>>],
    /// Public values for this instance.
    pub(crate) public_values: &'a [Val<SC>],
    /// Trace domain for this instance.
    pub(crate) trace_domain: Domain<SC>,
    /// Quotient polynomial evaluated at `zeta`.
    pub(crate) quotient: SC::Challenge,
}

impl<'a, SC: SGC> VerifierData<'a, SC> {
    /// Verifies that the folded constraints match the quotient polynomial at zeta.
    ///
    /// This evaluates the AIR constraints at the out-of-domain point and checks
    /// that constraints(zeta) / Z_H(zeta) = quotient(zeta).
    pub fn verify_constraints_with_lookups<A, LG: LookupProtocol, PcsErr: Debug>(
        &self,
        air: &A,
        lookup_gadget: &LG,
    ) -> Result<(), VerificationError<PcsErr>>
    where
        A: for<'b> Air<VerifierConstraintFolderWithLookups<'b, SC>>,
    {
        let sels = self.trace_domain.selectors_at_point(self.zeta);

        let main = VerticalPair::new(
            RowMajorMatrixView::new_row(self.trace_local),
            RowMajorMatrixView::new_row(self.trace_next),
        );

        let preprocessed = VerticalPair::new(
            RowMajorMatrixView::new_row(self.preprocessed_local),
            RowMajorMatrixView::new_row(self.preprocessed_next),
        );

        let preprocessed_window =
            RowWindow::from_two_rows(preprocessed.top.values, preprocessed.bottom.values);
        let inner_folder = VerifierConstraintFolder {
            main,
            preprocessed,
            preprocessed_window,
            periodic_values: self.periodic_values,
            public_values: self.public_values,
            is_first_row: sels.is_first_row,
            is_last_row: sels.is_last_row,
            is_transition: sels.is_transition,
            alpha: self.alpha,
            accumulator: SC::Challenge::ZERO,
        };
        let mut folder = VerifierConstraintFolderWithLookups {
            inner: inner_folder,
            permutation: VerticalPair::new(
                RowMajorMatrixView::new_row(self.permutation_local),
                RowMajorMatrixView::new_row(self.permutation_next),
            ),
            permutation_challenges: self.permutation_challenges,
            permutation_values: self.permutation_values,
        };
        // Evaluate AIR and lookup constraints.
        lookup_gadget.eval_air_and_lookups(air, &mut folder, self.lookups);

        // Check that constraints(zeta) / Z_H(zeta) = quotient(zeta)
        if folder.inner.accumulator * sels.inv_vanishing != self.quotient {
            return Err(VerificationError::OodEvaluationMismatch { index: None });
        }

        Ok(())
    }
}
