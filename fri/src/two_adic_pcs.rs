use core::marker::PhantomData;

use alloc::vec::Vec;
use p3_challenger::FieldChallenger;
use p3_commit::{
    DirectMmcs, OpenedValues, OpenedValuesForMatrix, OpenedValuesForRound, Pcs, UnivariatePcs,
    UnivariatePcsWithLde,
};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, TwoAdicField};
use p3_interpolation::interpolate_coset;
use p3_matrix::{
    bitrev::{BitReversableMatrix, BitReversedMatrixView},
    dense::RowMajorMatrixView,
    Dimensions, Matrix, MatrixRows,
};
use serde::{Deserialize, Serialize};
use tracing::info_span;

use crate::{prover, verifier::VerificationErrorForFriConfig, FriConfig, FriProof};

pub struct TwoAdicFriPcs<FC, Val, Dft, M> {
    fri: FC,
    dft: Dft,
    mmcs: M,
    _phantom: PhantomData<Val>,
}

impl<FC, Val, Dft, M> TwoAdicFriPcs<FC, Val, Dft, M> {
    fn coset_shift(&self) -> Val
    where
        Val: TwoAdicField,
    {
        Val::generator()
    }
}

#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct TwoAdicFriPcsProof<FC: FriConfig> {
    pub(crate) fri_proof: FriProof<FC>,
}

impl<FC, Val, Dft, M, In> Pcs<Val, In> for TwoAdicFriPcs<FC, Val, Dft, M>
where
    Val: TwoAdicField,
    FC: FriConfig,
    FC::Challenge: ExtensionField<Val>,
    FC::Challenger: FieldChallenger<Val>,
    Dft: TwoAdicSubgroupDft<Val>,
    M: 'static + for<'a> DirectMmcs<Val, Mat<'a> = RowMajorMatrixView<'a, Val>>,
    In: MatrixRows<Val>,
{
    type Commitment = M::Commitment;
    type ProverData = M::ProverData;
    type Proof = TwoAdicFriPcsProof<FC>;
    type Error = VerificationErrorForFriConfig<FC>;

    fn commit_batches(&self, polynomials: Vec<In>) -> (Self::Commitment, Self::ProverData) {
        self.commit_shifted_batches(polynomials, Val::one())
    }
}

impl<FC, Val, Dft, M, In> UnivariatePcsWithLde<Val, FC::Challenge, In, FC::Challenger>
    for TwoAdicFriPcs<FC, Val, Dft, M>
where
    Val: TwoAdicField,
    FC: FriConfig,
    FC::Challenge: ExtensionField<Val>,
    FC::Challenger: FieldChallenger<Val>,
    Dft: TwoAdicSubgroupDft<Val>,
    M: 'static + for<'a> DirectMmcs<Val, Mat<'a> = RowMajorMatrixView<'a, Val>>,
    In: MatrixRows<Val>,
{
    type Lde<'a> = BitReversedMatrixView<M::Mat<'a>> where Self: 'a;

    fn coset_shift(&self) -> Val {
        self.coset_shift()
    }

    fn log_blowup(&self) -> usize {
        self.fri.log_blowup()
    }

    fn get_ldes<'a, 'b>(&'a self, prover_data: &'b Self::ProverData) -> Vec<Self::Lde<'b>>
    where
        'a: 'b,
    {
        // We committed to the bit-reversed LDE, so now we wrap it to return in natural order.
        self.mmcs
            .get_matrices(prover_data)
            .into_iter()
            .map(|m| BitReversedMatrixView::new(m))
            .collect()
    }

    fn commit_shifted_batches(
        &self,
        polynomials: Vec<In>,
        coset_shift: Val,
    ) -> (Self::Commitment, Self::ProverData) {
        let shift = Val::generator() / coset_shift;
        let ldes = info_span!("compute all coset LDEs").in_scope(|| {
            polynomials
                .into_iter()
                .map(|poly| {
                    let input = poly.to_row_major_matrix();
                    // Commit to the bit-reversed LDE.
                    self.dft
                        .coset_lde_batch(input, self.fri.log_blowup(), shift)
                        .bit_reverse_rows()
                        .to_row_major_matrix()
                })
                .collect()
        });
        self.mmcs.commit(ldes)
    }
}

impl<FC, Val, Dft, M, In> UnivariatePcs<Val, FC::Challenge, In, FC::Challenger>
    for TwoAdicFriPcs<FC, Val, Dft, M>
where
    Val: TwoAdicField,
    FC: FriConfig,
    FC::Challenge: ExtensionField<Val>,
    FC::Challenger: FieldChallenger<Val>,
    Dft: TwoAdicSubgroupDft<Val>,
    M: 'static + for<'a> DirectMmcs<Val, Mat<'a> = RowMajorMatrixView<'a, Val>>,
    In: MatrixRows<Val>,
{
    fn open_multi_batches(
        &self,
        prover_data_and_points: &[(&Self::ProverData, &[Vec<FC::Challenge>])],
        challenger: &mut FC::Challenger,
    ) -> (OpenedValues<FC::Challenge>, Self::Proof) {
        let coset_shift = self.coset_shift();

        // Use Barycentric interpolation to evaluate each matrix at a given point.
        let eval_at_points = |matrix: M::Mat<'_>, points: &[FC::Challenge]| {
            points
                .iter()
                .map(|&point| {
                    let (low_coset, _) =
                        matrix.split_rows(matrix.height() >> self.fri.log_blowup());
                    interpolate_coset(&BitReversedMatrixView::new(low_coset), coset_shift, point)
                })
                .collect::<OpenedValuesForMatrix<FC::Challenge>>()
        };

        let all_opened_values = info_span!("compute opened values with Lagrange interpolation")
            .in_scope(|| {
                prover_data_and_points
                    .iter()
                    .map(|(data, points_per_matrix)| {
                        let matrices = self.mmcs.get_matrices(data);

                        matrices
                            .iter()
                            .zip(*points_per_matrix)
                            .map(|(&mat, points_for_mat)| eval_at_points(mat, points_for_mat))
                            .collect::<OpenedValuesForRound<FC::Challenge>>()
                    })
                    .collect::<OpenedValues<FC::Challenge>>()
            });

        let input = todo!();
        let (fri_proof, query_indices) = prover::prove(&self.fri, input, challenger);

        todo!()
    }

    fn verify_multi_batches(
        &self,
        commits_and_points: &[(Self::Commitment, &[Vec<FC::Challenge>])],
        dims: &[Vec<Dimensions>],
        values: OpenedValues<FC::Challenge>,
        proof: &Self::Proof,
        challenger: &mut FC::Challenger,
    ) -> Result<(), Self::Error> {
        todo!()
    }
}
