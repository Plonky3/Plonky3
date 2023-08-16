use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::FieldChallenger;
use p3_commit::{DirectMmcs, Pcs, UnivariatePcs};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{AbstractExtensionField, ExtensionField, Field, TwoAdicField};
use p3_matrix::MatrixRows;

use crate::Ldt;

pub struct LdtBasedPcs<Val, Dom, Dft, M, L> {
    dft: Dft,
    added_bits: usize,
    mmcs: M,
    _phantom_val: PhantomData<Val>,
    _phantom_dom: PhantomData<Dom>,
    _phantom_l: PhantomData<L>,
}

impl<Val, Dom, Dft, M, L> LdtBasedPcs<Val, Dom, Dft, M, L> {
    pub fn new(dft: Dft, added_bits: usize, mmcs: M) -> Self {
        Self {
            dft,
            added_bits,
            mmcs,
            _phantom_val: PhantomData,
            _phantom_dom: PhantomData,
            _phantom_l: PhantomData,
        }
    }
}

impl<Val, Dom, In, Dft, M, L, Challenger> Pcs<Val, In, Challenger>
    for LdtBasedPcs<Val, Dom, Dft, M, L>
where
    Val: Field,
    Dom: ExtensionField<Val> + TwoAdicField,
    In: MatrixRows<Val>,
    Dft: TwoAdicSubgroupDft<Dom>,
    M: DirectMmcs<Dom>,
    L: Ldt<Dom, M, Challenger>,
    Challenger: FieldChallenger<Val> + FieldChallenger<Dom>,
{
    type Commitment = M::Commitment;
    type ProverData = M::ProverData;
    type Proof = L::Proof;
    type Error = L::Error;

    fn commit_batches(&self, polynomials: Vec<In>) -> (Self::Commitment, Self::ProverData) {
        // TODO: Streaming?
        let shift = Dom::multiplicative_group_generator();
        let ldes = polynomials
            .into_iter()
            .map(|poly| {
                let input = poly.to_row_major_matrix().map(Dom::from_base);
                self.dft.coset_lde_batch(input, self.added_bits, shift)
            })
            .collect();
        self.mmcs.commit(ldes)
    }
}

impl<Val, Dom, In, Dft, M, L, Challenger> UnivariatePcs<Val, In, Challenger>
    for LdtBasedPcs<Val, Dom, Dft, M, L>
where
    Val: Field,
    Dom: ExtensionField<Val> + TwoAdicField,
    In: MatrixRows<Val>,
    Dft: TwoAdicSubgroupDft<Dom>,
    M: DirectMmcs<Dom>,
    L: Ldt<Dom, M, Challenger>,
    Challenger: FieldChallenger<Val> + FieldChallenger<Dom>,
{
    fn open_multi_batches<EF>(
        &self,
        _prover_data: &[&Self::ProverData],
        _points: &[EF],
        _challenger: &mut Challenger,
    ) -> (Vec<Vec<Vec<EF>>>, Self::Proof)
    where
        EF: AbstractExtensionField<Val>,
    {
        todo!()
    }

    fn verify_multi_batches<EF>(
        &self,
        _commits: &[Self::Commitment],
        _points: &[EF],
        _values: &[Vec<Vec<EF>>],
        _proof: &Self::Proof,
    ) -> Result<(), Self::Error>
    where
        EF: AbstractExtensionField<Val>,
    {
        todo!()
    }
}
