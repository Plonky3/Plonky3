use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::FieldChallenger;
use p3_commit::{DirectMmcs, OpenedValues, Pcs, UnivariatePcs};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{AbstractExtensionField, ExtensionField, Field, TwoAdicField};
use p3_interpolation::interpolate_coset;
use p3_matrix::dense::RowMajorMatrixView;
use p3_matrix::MatrixRows;

use crate::quotient::QuotientMmcs;
use crate::Ldt;

pub struct LdtBasedPcs<Val, Domain, Dft, M, L, Challenger> {
    dft: Dft,
    added_bits: usize,
    mmcs: M,
    ldt: L,
    _phantom_val: PhantomData<Val>,
    _phantom_dom: PhantomData<Domain>,
    _phantom_challenger: PhantomData<Challenger>,
}

impl<Val, Domain, Dft, M, L, Challenger> LdtBasedPcs<Val, Domain, Dft, M, L, Challenger> {
    pub fn new(dft: Dft, added_bits: usize, mmcs: M, ldt: L) -> Self {
        Self {
            dft,
            added_bits,
            mmcs,
            ldt,
            _phantom_val: PhantomData,
            _phantom_dom: PhantomData,
            _phantom_challenger: PhantomData,
        }
    }
}

impl<Val, Domain, In, Dft, M, L, Challenger> Pcs<Val, In>
    for LdtBasedPcs<Val, Domain, Dft, M, L, Challenger>
where
    Val: Field,
    Domain: ExtensionField<Val> + TwoAdicField,
    In: MatrixRows<Val>,
    Dft: TwoAdicSubgroupDft<Domain>,
    M: DirectMmcs<Domain>,
    L: Ldt<Val, Domain, QuotientMmcs<Domain, M>, Challenger>,
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
                let input = poly.to_row_major_matrix().map(Domain::from_base);
                self.dft.coset_lde_batch(input, self.added_bits, shift)
            })
            .collect();
        self.mmcs.commit(ldes)
    }
}

impl<Val, Domain, In, Dft, M, L, Challenger> UnivariatePcs<Val, Domain, In, Challenger>
    for LdtBasedPcs<Val, Domain, Dft, M, L, Challenger>
where
    Val: Field,
    Domain: ExtensionField<Val> + TwoAdicField,
    In: MatrixRows<Val>,
    Dft: TwoAdicSubgroupDft<Domain>,
    M: 'static + for<'a> DirectMmcs<Domain, Mat<'a> = RowMajorMatrixView<'a, Domain>>,
    L: Ldt<Val, Domain, QuotientMmcs<Domain, M>, Challenger>,
    Challenger: FieldChallenger<Val>,
{
    fn open_multi_batches<EF>(
        &self,
        prover_data_and_points: &[(&Self::ProverData, &[EF])],
        challenger: &mut Challenger,
    ) -> (OpenedValues<EF>, Self::Proof)
    where
        EF: ExtensionField<Domain> + TwoAdicField,
    {
        let openings = prover_data_and_points
            .iter()
            .map(|(data, points)| {
                self.mmcs
                    .get_matrices(data)
                    .into_iter()
                    .map(|mat| {
                        let low_coset = mat.vertically_strided(1 << self.added_bits, 0);
                        let shift = Domain::multiplicative_group_generator();
                        points
                            .iter()
                            .map(|&point| interpolate_coset(&low_coset, shift, point))
                            .collect()
                    })
                    .collect()
            })
            .collect();
        let quotient_mmcs = QuotientMmcs {
            inner: self.mmcs.clone(),
            opened_point: Domain::ZERO, // TODO: points
            opened_eval: Domain::ZERO,  // TODO
        };
        let prover_data: Vec<_> = prover_data_and_points
            .iter()
            .map(|(prover_data, _points)| *prover_data)
            .collect();
        let proof = self.ldt.prove(&quotient_mmcs, &prover_data, challenger);
        (openings, proof)
    }

    fn verify_multi_batches<EF>(
        &self,
        _commits_and_points: &[(Self::Commitment, &[EF])],
        _values: OpenedValues<EF>,
        _proof: &Self::Proof,
    ) -> Result<(), Self::Error>
    where
        EF: AbstractExtensionField<Domain>,
    {
        todo!()
    }
}
