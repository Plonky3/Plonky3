use alloc::vec::Vec;
use core::marker::PhantomData;

use itertools::Itertools;
use p3_challenger::FieldChallenger;
use p3_commit::{
    DirectMmcs, OpenedValues, OpenedValuesForPoint, OpenedValuesForRound, Pcs, UnivariatePcs,
};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_interpolation::interpolate_coset;
use p3_matrix::dense::RowMajorMatrixView;
use p3_matrix::MatrixRows;
use tracing::instrument;

use crate::quotient::QuotientMmcs;
use crate::{Ldt, Opening};

pub struct LdtBasedPcs<Val, Domain, EF, Dft, M, L, Challenger> {
    dft: Dft,
    added_bits: usize,
    mmcs: M,
    ldt: L,
    _phantom_val: PhantomData<Val>,
    _phantom_dom: PhantomData<Domain>,
    _phantom_ef: PhantomData<EF>,
    _phantom_challenger: PhantomData<Challenger>,
}

impl<Val, Domain, EF, Dft, M, L, Challenger> LdtBasedPcs<Val, Domain, EF, Dft, M, L, Challenger> {
    pub fn new(dft: Dft, added_bits: usize, mmcs: M, ldt: L) -> Self {
        Self {
            dft,
            added_bits,
            mmcs,
            ldt,
            _phantom_val: PhantomData,
            _phantom_dom: PhantomData,
            _phantom_ef: PhantomData,
            _phantom_challenger: PhantomData,
        }
    }
}

impl<Val, Domain, EF, In, Dft, M, L, Challenger> Pcs<Val, In>
    for LdtBasedPcs<Val, Domain, EF, Dft, M, L, Challenger>
where
    Val: Field,
    Domain: ExtensionField<Val> + TwoAdicField,
    EF: ExtensionField<Val> + ExtensionField<Domain>,
    In: MatrixRows<Val>,
    Dft: TwoAdicSubgroupDft<Domain>,
    M: DirectMmcs<Domain>,
    L: Ldt<Val, EF, QuotientMmcs<Domain, EF, M>, Challenger>,
    Challenger: FieldChallenger<Val>,
{
    type Commitment = M::Commitment;
    type ProverData = M::ProverData;
    type Proof = L::Proof;
    type Error = L::Error;

    fn commit_batches(&self, polynomials: Vec<In>) -> (Self::Commitment, Self::ProverData) {
        // TODO: Streaming?
        let shift = Domain::multiplicative_group_generator();
        let ldes = polynomials
            .into_iter()
            .map(|poly| {
                let input = poly.to_row_major_matrix().to_ext::<Domain>();
                self.dft.coset_lde_batch(input, self.added_bits, shift)
            })
            .collect();
        self.mmcs.commit(ldes)
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
    #[instrument(name="prove batch opening", skip_all)]
    fn open_multi_batches(
        &self,
        prover_data_and_points: &[(&Self::ProverData, &[EF])],
        challenger: &mut Challenger,
    ) -> (OpenedValues<EF>, Self::Proof) {
        let all_opened_values = prover_data_and_points
            .iter()
            .map(|(data, points)| {
                points
                    .iter()
                    .map(|&point| {
                        self.mmcs
                            .get_matrices(data)
                            .into_iter()
                            .map(|mat| {
                                let low_coset = mat.vertically_strided(1 << self.added_bits, 0);
                                let shift = Domain::multiplicative_group_generator();
                                interpolate_coset(&low_coset, shift, point)
                            })
                            .collect::<OpenedValuesForPoint<EF>>()
                    })
                    .collect::<OpenedValuesForRound<EF>>()
            })
            .collect::<OpenedValues<EF>>();

        let (prover_data, all_points): (Vec<_>, Vec<_>) =
            prover_data_and_points.iter().copied().unzip();

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
                    _phantom_f: PhantomData,
                }
            })
            .collect_vec();

        let proof = self.ldt.prove(&quotient_mmcs, &prover_data, challenger);
        (all_opened_values, proof)
    }

    fn verify_multi_batches(
        &self,
        _commits_and_points: &[(Self::Commitment, &[EF])],
        _values: OpenedValues<EF>,
        _proof: &Self::Proof,
    ) -> Result<(), Self::Error> {
        todo!()
    }
}

fn transpose<T: Clone>(vec: Vec<Vec<T>>) -> Vec<Vec<T>> {
    let n = vec.len();
    let m = vec[0].len();
    (0..m)
        .map(|r| (0..n).map(|c| vec[c][r].clone()).collect())
        .collect()
}
