use alloc::vec::Vec;
use core::marker::PhantomData;

use itertools::Itertools;
use p3_challenger::FieldChallenger;
use p3_commit::{
    DirectMmcs, OpenedValues, OpenedValuesForMatrix, OpenedValuesForPoint, OpenedValuesForRound,
    Pcs, UnivariatePcs, UnivariatePcsWithLde,
};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::extension::HasFrobenius;
use p3_field::{ExtensionField, TwoAdicField};
use p3_interpolation::interpolate_coset;
use p3_matrix::dense::RowMajorMatrixView;
use p3_matrix::{MatrixRowSlices, MatrixRows};
use tracing::{info_span, instrument};

use crate::quotient::QuotientMmcs;
use crate::{Ldt, Opening};

pub struct LdtBasedPcs<Val, EF, Dft, M, L, Challenger> {
    dft: Dft,
    mmcs: M,
    ldt: L,
    _phantom: PhantomData<(Val, EF, Challenger)>,
}

impl<Val, EF, Dft, M, L, Challenger> LdtBasedPcs<Val, EF, Dft, M, L, Challenger> {
    pub fn new(dft: Dft, mmcs: M, ldt: L) -> Self {
        Self {
            dft,
            mmcs,
            ldt,
            _phantom: PhantomData,
        }
    }
}

impl<Val, EF, In, Dft, M, L, Challenger> UnivariatePcsWithLde<Val, EF, In, Challenger>
    for LdtBasedPcs<Val, EF, Dft, M, L, Challenger>
where
    Val: TwoAdicField,
    EF: ExtensionField<Val> + TwoAdicField + HasFrobenius<Val>,
    In: MatrixRows<Val>,
    Dft: TwoAdicSubgroupDft<Val>,
    M: 'static + for<'a> DirectMmcs<Val, Mat<'a> = RowMajorMatrixView<'a, Val>>,
    L: Ldt<Val, QuotientMmcs<Val, EF, M>, Challenger>,
    Challenger: FieldChallenger<Val>,
{
    fn coset_shift(&self) -> Val {
        Val::generator()
    }

    fn log_blowup(&self) -> usize {
        self.ldt.log_blowup()
    }

    fn get_ldes<'a, 'b>(
        &'a self,
        prover_data: &'b Self::ProverData,
    ) -> Vec<RowMajorMatrixView<'b, Val>>
    where
        'a: 'b,
    {
        self.mmcs.get_matrices(prover_data)
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
                    self.dft
                        .coset_lde_batch(input, self.ldt.log_blowup(), shift)
                })
                .collect()
        });
        self.mmcs.commit(ldes)
    }
}

impl<Val, EF, In, Dft, M, L, Challenger> Pcs<Val, In>
    for LdtBasedPcs<Val, EF, Dft, M, L, Challenger>
where
    Val: TwoAdicField,
    EF: ExtensionField<Val> + TwoAdicField + HasFrobenius<Val>,
    In: MatrixRows<Val>,
    Dft: TwoAdicSubgroupDft<Val>,
    M: 'static + for<'a> DirectMmcs<Val, Mat<'a> = RowMajorMatrixView<'a, Val>>,
    for<'a> M::Mat<'a>: MatrixRowSlices<Val>,
    L: Ldt<Val, QuotientMmcs<Val, EF, M>, Challenger>,
    Challenger: FieldChallenger<Val>,
{
    type Commitment = M::Commitment;
    type ProverData = M::ProverData;
    type Proof = L::Proof;
    type Error = L::Error;

    fn commit_batches(&self, polynomials: Vec<In>) -> (Self::Commitment, Self::ProverData) {
        self.commit_shifted_batches(polynomials, Val::one())
    }
}

impl<Val, EF, In, Dft, M, L, Challenger> UnivariatePcs<Val, EF, In, Challenger>
    for LdtBasedPcs<Val, EF, Dft, M, L, Challenger>
where
    Val: TwoAdicField,
    EF: ExtensionField<Val> + TwoAdicField + HasFrobenius<Val>,
    In: MatrixRows<Val>,
    Dft: TwoAdicSubgroupDft<Val>,
    M: 'static + for<'a> DirectMmcs<Val, Mat<'a> = RowMajorMatrixView<'a, Val>>,
    L: Ldt<Val, QuotientMmcs<Val, EF, M>, Challenger>,
    Challenger: FieldChallenger<Val>,
{
    #[instrument(name = "prove batch opening", skip_all)]
    fn open_multi_batches(
        &self,
        prover_data_and_points: &[(&Self::ProverData, &[Vec<EF>])],
        challenger: &mut Challenger,
    ) -> (OpenedValues<EF>, Self::Proof) {
        // Use Barycentric interpolation to evaluate each matrix at a given point.
        let eval_at_points = |matrix: M::Mat<'_>, points: Vec<EF>| {
            points
                .iter()
                .map(|&point| {
                    let low_coset = matrix.vertically_strided(self.ldt.blowup(), 0);
                    let shift = Val::generator();
                    interpolate_coset(&low_coset, shift, point)
                })
                .collect::<OpenedValuesForMatrix<EF>>()
        };

        let all_opened_values = info_span!("compute opened values with Lagrange interpolation")
            .in_scope(|| {
                prover_data_and_points
                    .iter()
                    .map(|(data, points_per_matrix)| {
                        let matrices = self.mmcs.get_matrices(data);

                        matrices
                            .iter()
                            .enumerate()
                            .map(|(i, &mat)| {
                                let eval_points_for_matrix = points_per_matrix[i].clone();
                                eval_at_points(mat, eval_points_for_matrix)
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<OpenedValuesForRound<EF>>>()
            });

        let (prover_data, all_points): (Vec<_>, Vec<_>) =
            prover_data_and_points.iter().copied().unzip();

        let coset_shift: Val =
            <Self as UnivariatePcsWithLde<Val, EF, In, Challenger>>::coset_shift(self);

        let quotient_mmcs = all_points
            .into_iter()
            .zip(&all_opened_values)
            .map(
                |(points_for_matrix, opened_values_for_round_per_matrix): (
                    &[Vec<EF>],
                    &OpenedValuesForRound<EF>,
                )| {
                    debug_assert!(
                        points_for_matrix.len() == points_for_matrix.len(),
                        "points_per_matrix.len() == {}, opened_values_for_round_per_matrix.len() = {}", 
                        points_for_matrix.len(), points_for_matrix.len() 
                    );
                    let opened_values_for_round_per_matrix =
                        transpose(opened_values_for_round_per_matrix.to_vec());

                    let openings = opened_values_for_round_per_matrix
                        .into_iter()
                        .map(
                            |opened_values_for_matrix_per_point: OpenedValuesForMatrix<EF>| {
                                opened_values_for_matrix_per_point
                                    .iter()
                                    .enumerate()
                                    .flat_map(|(point_ind, opened_values_for_matrix): (usize, &OpenedValuesForPoint<EF>)| {
                                        let points = points_for_matrix[point_ind].clone();
                                        points
                                            .iter()
                                            .map(|&point| {
                                                Opening::<Val, EF>::new(
                                                    point,
                                                    opened_values_for_matrix.clone(),
                                                )
                                            })
                                            .collect::<Vec<_>>()
                                    })
                                    .collect::<Vec<Opening<Val, EF>>>()
                            },
                        )
                        .collect();
                    QuotientMmcs::<Val, EF, _> {
                        inner: self.mmcs.clone(),
                        openings,
                        coset_shift,
                        _phantom: PhantomData,
                    }
                },
            )
            .collect_vec();

        let proof = self.ldt.prove(&quotient_mmcs, &prover_data, challenger);
        (all_opened_values, proof)
    }

    fn verify_multi_batches(
        &self,
        _commits_and_points: &[(Self::Commitment, &[EF])],
        _values: OpenedValues<EF>,
        _proof: &Self::Proof,
        _challenger: &mut Challenger,
    ) -> Result<(), Self::Error> {
        Ok(()) // TODO
    }
}

fn transpose<T: Clone>(vec: Vec<Vec<T>>) -> Vec<Vec<T>> {
    let n = vec.len();
    let m = vec[0].len();
    (0..m)
        .map(|r| (0..n).map(|c| vec[c][r].clone()).collect())
        .collect()
}
