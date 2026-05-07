use core::ops::Deref;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrixView;
use p3_multilinear_util::point::Point;
use tracing::{info_span, instrument};

use crate::constraints::statement::initial::InitialStatement;
use crate::fiat_shamir::errors::FiatShamirError;
use crate::parameters::WhirConfig;
use crate::pcs::committer::DenseMatrix;
use crate::pcs::proof::WhirProof;

/// Commits polynomials using a Merkle-based scheme.
///
/// Expands and folds evaluations via DFT, then builds a Merkle tree
/// over the resulting codeword.
#[derive(Debug)]
pub struct CommitmentWriter<'a, EF, F, MT: Mmcs<F>, Challenger>(
    &'a WhirConfig<EF, F, MT, Challenger>,
)
where
    F: Field,
    EF: ExtensionField<F>;

impl<'a, EF, F, MT, Challenger> CommitmentWriter<'a, EF, F, MT, Challenger>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    MT: Mmcs<F>,
{
    pub const fn new(params: &'a WhirConfig<EF, F, MT, Challenger>) -> Self {
        Self(params)
    }

    /// Commits a polynomial via DFT expansion and Merkle tree construction.
    ///
    /// # Algorithm
    ///
    /// 1. Transpose evaluations for reverse variable order, then zero-pad.
    /// 2. Apply DFT to produce the codeword matrix.
    /// 3. Commit the matrix into a Merkle tree.
    /// 4. Sample OOD challenge points and record evaluations in the transcript.
    #[instrument(skip_all)]
    pub fn commit<Dft>(
        &self,
        dft: &Dft,
        proof: &mut WhirProof<F, EF, MT>,
        challenger: &mut Challenger,
        statement: &mut InitialStatement<F, EF>,
    ) -> Result<MT::ProverData<DenseMatrix<F>>, FiatShamirError>
    where
        Dft: TwoAdicSubgroupDft<F>,
        Challenger: CanObserve<MT::Commitment>,
    {
        // Transpose for reverse variable order, then zero-pad to the expanded domain.
        let padded = info_span!("transpose & pad").in_scope(|| {
            let num_vars = statement.num_variables();
            let mut mat = RowMajorMatrixView::new(
                statement.poly.as_slice(),
                1 << (num_vars - self.folding_factor.at_round(0)),
            )
            .transpose();
            mat.pad_to_height(
                1 << (num_vars + self.starting_log_inv_rate - self.folding_factor.at_round(0)),
                F::ZERO,
            );
            mat
        });

        // Apply DFT to produce the Reed-Solomon codeword.
        let folded_matrix = info_span!("dft", height = padded.height(), width = padded.width())
            .in_scope(|| dft.dft_batch(padded).to_row_major_matrix());

        // Build Merkle tree and extract commitment root.
        let (root, prover_data) =
            info_span!("commit_matrix").in_scope(|| self.mmcs.commit_matrix(folded_matrix));

        proof.initial_commitment = Some(root.clone());
        challenger.observe(root);

        // Sample and evaluate out-of-domain challenge points.
        (0..self.0.commitment_ood_samples).for_each(|_| {
            let point = Point::expand_from_univariate(
                challenger.sample_algebra_element(),
                self.num_variables,
            );
            let eval = info_span!("ood evaluation").in_scope(|| statement.evaluate(&point));
            proof.initial_ood_answers.push(eval);
            challenger.observe_algebra_element(eval);
        });

        Ok(prover_data)
    }
}

impl<EF, F, MT: Mmcs<F>, Challenger> Deref for CommitmentWriter<'_, EF, F, MT, Challenger>
where
    F: Field,
    EF: ExtensionField<F>,
{
    type Target = WhirConfig<EF, F, MT, Challenger>;

    fn deref(&self) -> &Self::Target {
        self.0
    }
}
