use alloc::vec::Vec;
use core::marker::PhantomData;

use itertools::Itertools;
use p3_challenger::FieldChallenger;
use p3_commit::{
    DirectMmcs, OpenedValues, OpenedValuesForPoint, OpenedValuesForRound, Pcs, UnivariatePcs,
    UnivariatePcsWithLde,
};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_interpolation::interpolate_coset;
use p3_matrix::dense::RowMajorMatrixView;
use p3_matrix::{Dimensions, MatrixRowSlices, MatrixRows};
use tracing::{info_span, instrument};

use crate::quotient::QuotientMmcs;
use crate::{Ldt, Opening};

pub struct LdtBasedPcs<Val, Domain, EF, Dft, M, L, Challenger> {
    dft: Dft,
    mmcs: M,
    ldt: L,
    _phantom: PhantomData<(Val, Domain, EF, Challenger)>,
}

impl<Val, Domain, EF, Dft, M, L, Challenger> LdtBasedPcs<Val, Domain, EF, Dft, M, L, Challenger> {
    pub fn new(dft: Dft, mmcs: M, ldt: L) -> Self {
        Self {
            dft,
            mmcs,
            ldt,
            _phantom: PhantomData,
        }
    }
}

impl<Val, Domain, EF, In, Dft, M, L, Challenger>
    UnivariatePcsWithLde<Val, Domain, EF, In, Challenger>
    for LdtBasedPcs<Val, Domain, EF, Dft, M, L, Challenger>
where
    Val: Field,
    Domain: ExtensionField<Val> + TwoAdicField,
    EF: ExtensionField<Val> + ExtensionField<Domain> + TwoAdicField,
    In: MatrixRows<Val>,
    Dft: TwoAdicSubgroupDft<Domain>,
    M: 'static + for<'a> DirectMmcs<Domain, Mat<'a> = RowMajorMatrixView<'a, Domain>>,
    L: Ldt<Val, EF, QuotientMmcs<Domain, EF, M>, Challenger>,
    Challenger: FieldChallenger<Val>,
{
    fn coset_shift(&self) -> Domain {
        Domain::generator()
    }

    fn log_blowup(&self) -> usize {
        self.ldt.log_blowup()
    }

    fn get_ldes<'a, 'b>(
        &'a self,
        prover_data: &'b Self::ProverData,
    ) -> Vec<RowMajorMatrixView<'b, Domain>>
    where
        'a: 'b,
    {
        self.mmcs.get_matrices(prover_data)
    }

    fn commit_shifted_batches(
        &self,
        polynomials: Vec<In>,
        coset_shift: Domain,
    ) -> (Self::Commitment, Self::ProverData) {
        let shift = Domain::generator() / coset_shift;
        let ldes = info_span!("compute all coset LDEs").in_scope(|| {
            polynomials
                .into_iter()
                .map(|poly| {
                    let input = poly.to_row_major_matrix().to_ext::<Domain>();
                    self.dft
                        .coset_lde_batch(input, self.ldt.log_blowup(), shift)
                })
                .collect()
        });
        self.mmcs.commit(ldes)
    }
}

impl<Val, Domain, EF, In, Dft, M, L, Challenger> Pcs<Val, In>
    for LdtBasedPcs<Val, Domain, EF, Dft, M, L, Challenger>
where
    Val: Field,
    Domain: ExtensionField<Val> + TwoAdicField,
    EF: ExtensionField<Val> + ExtensionField<Domain> + TwoAdicField,
    In: MatrixRows<Val>,
    Dft: TwoAdicSubgroupDft<Domain>,
    M: 'static + for<'a> DirectMmcs<Domain, Mat<'a> = RowMajorMatrixView<'a, Domain>>,
    for<'a> M::Mat<'a>: MatrixRowSlices<Domain>,
    L: Ldt<Val, EF, QuotientMmcs<Domain, EF, M>, Challenger>,
    Challenger: FieldChallenger<Val>,
{
    type Commitment = M::Commitment;
    type ProverData = M::ProverData;
    type Proof = L::Proof;
    type Error = L::Error;

    fn commit_batches(&self, polynomials: Vec<In>) -> (Self::Commitment, Self::ProverData) {
        self.commit_shifted_batches(polynomials, Domain::one())
    }
}

impl<Val, Domain, EF, In, Dft, M, L, Challenger> UnivariatePcs<Val, Domain, EF, In, Challenger>
    for LdtBasedPcs<Val, Domain, EF, Dft, M, L, Challenger>
where
    Val: Field,
    Domain: ExtensionField<Val> + TwoAdicField,
    EF: ExtensionField<Val> + ExtensionField<Domain> + TwoAdicField,
    In: MatrixRows<Val>,
    Dft: TwoAdicSubgroupDft<Domain>,
    M: 'static + for<'a> DirectMmcs<Domain, Mat<'a> = RowMajorMatrixView<'a, Domain>>,
    L: Ldt<Val, EF, QuotientMmcs<Domain, EF, M>, Challenger>,
    Challenger: FieldChallenger<Val>,
{
    #[instrument(name = "prove batch opening", skip_all)]
    fn open_multi_batches(
        &self,
        prover_data_and_points: &[(&Self::ProverData, &[EF])],
        challenger: &mut Challenger,
    ) -> (OpenedValues<EF>, Self::Proof) {
        // Use Barycentric interpolation to evaluate each matrix at a given point.
        let eval_at_point = |matrices: &[M::Mat<'_>], point| {
            matrices
                .iter()
                .map(|mat| {
                    let low_coset = mat.vertically_strided(self.ldt.blowup(), 0);
                    let shift = Domain::generator();
                    interpolate_coset(&low_coset, shift, point)
                })
                .collect::<OpenedValuesForPoint<EF>>()
        };

        let all_opened_values = info_span!("compute opened values with Lagrange interpolation")
            .in_scope(|| {
                prover_data_and_points
                    .iter()
                    .map(|(data, points)| {
                        let matrices = self.mmcs.get_matrices(data);
                        points
                            .iter()
                            .map(|&point| eval_at_point(&matrices, point))
                            .collect::<OpenedValuesForRound<EF>>()
                    })
                    .collect::<OpenedValues<EF>>()
            });

        let (prover_data, all_points): (Vec<_>, Vec<_>) =
            prover_data_and_points.iter().copied().unzip();

        let coset_shift: Domain =
            <Self as UnivariatePcsWithLde<Val, Domain, EF, In, Challenger>>::coset_shift(self);

        let quotient_mmcs = all_points
            .into_iter()
            .zip(&all_opened_values)
            .map(|(points, opened_values_for_round_by_point)| {
                let opened_values_for_round_by_matrix =
                    transpose(opened_values_for_round_by_point.to_vec());

                let openings = opened_values_for_round_by_matrix
                    .into_iter()
                    .map(|opened_values_for_mat| {
                        points
                            .iter()
                            .zip(opened_values_for_mat)
                            .map(|(&point, opened_values_for_point)| Opening::<EF> {
                                point,
                                values: opened_values_for_point,
                            })
                            .collect()
                    })
                    .collect();
                QuotientMmcs::<Domain, EF, _> {
                    inner: self.mmcs.clone(),
                    openings,
                    coset_shift,
                }
            })
            .collect_vec();

        let proof = self.ldt.prove(&quotient_mmcs, &prover_data, challenger);
        (all_opened_values, proof)
    }

    fn verify_multi_batches(
        &self,
        commits_and_points: &[(Self::Commitment, &[EF])],
        dims: &[Vec<Dimensions>],
        values: OpenedValues<EF>,
        proof: &Self::Proof,
        challenger: &mut Challenger,
    ) -> Result<(), Self::Error> {
        let (commits, points): (Vec<Self::Commitment>, Vec<&[EF]>) =
            commits_and_points.iter().cloned().unzip();
        let coset_shift: Domain =
            <Self as UnivariatePcsWithLde<Val, Domain, EF, In, Challenger>>::coset_shift(self);
        let (dims, quotient_mmcs): (Vec<_>, Vec<_>) = points
            .into_iter()
            .zip_eq(values)
            .zip_eq(dims)
            .map(
                |((points, opened_values_for_round_by_point), dims): (
                    (&[EF], OpenedValuesForRound<EF>),
                    &Vec<Dimensions>,
                )| {
                    let opened_values_for_round_by_matrix =
                        transpose(opened_values_for_round_by_point.to_vec());
                    let openings = opened_values_for_round_by_matrix
                        .into_iter()
                        .map(|opened_values_for_matrix| {
                            points
                                .iter()
                                .zip(opened_values_for_matrix)
                                .map(|(&point, opened_values_for_point)| Opening::<EF> {
                                    point,
                                    values: opened_values_for_point,
                                })
                                .collect()
                        })
                        .collect_vec();
                    (
                        dims.iter()
                            .map(|d| Dimensions {
                                width: d.width * openings.len(),
                                height: d.height << self.ldt.log_blowup(),
                            })
                            .collect_vec(),
                        QuotientMmcs::<Domain, EF, _> {
                            inner: self.mmcs.clone(),
                            openings,
                            coset_shift,
                        },
                    )
                },
            )
            .unzip();
        self.ldt
            .verify(&quotient_mmcs, &dims, &commits, proof, challenger)
    }
}

fn transpose<T: Clone>(vec: Vec<Vec<T>>) -> Vec<Vec<T>> {
    let n = vec.len();
    let m = vec[0].len();
    (0..m)
        .map(|r| (0..n).map(|c| vec[c][r].clone()).collect())
        .collect()
}
