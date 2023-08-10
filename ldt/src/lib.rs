//! This crate contains a framework for low-degree tests (LDTs).

#![no_std]

extern crate alloc;

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::FieldChallenger;
use p3_commit::{DirectMMCS, UnivariatePCS, MMCS, PCS};
use p3_field::{AbstractExtensionField, ExtensionField, Field, TwoAdicField};
use p3_lde::TwoAdicLDE;
use p3_matrix::MatrixRows;

/// A batch low-degree test (LDT).
pub trait LDT<F: Field, M: MMCS<F>, Chal: FieldChallenger<F>> {
    type Proof;
    type Error;

    /// Prove that each column of each matrix in `codewords` is a codeword.
    fn prove(&self, inputs: &[M::ProverData], challenger: &mut Chal) -> Self::Proof;

    fn verify(
        &self,
        input_commits: &[M::Commitment],
        proof: &Self::Proof,
        challenger: &mut Chal,
    ) -> Result<(), Self::Error>;
}

pub struct LDTBasedPCS<Val, Dom, LDE, M, L> {
    lde: LDE,
    added_bits: usize,
    mmcs: M,
    _phantom_val: PhantomData<Val>,
    _phantom_dom: PhantomData<Dom>,
    _phantom_l: PhantomData<L>,
}

impl<Val, Dom, LDE, M, L> LDTBasedPCS<Val, Dom, LDE, M, L> {
    pub fn new(lde: LDE, added_bits: usize, mmcs: M) -> Self {
        Self {
            lde,
            added_bits,
            mmcs,
            _phantom_val: PhantomData,
            _phantom_dom: PhantomData,
            _phantom_l: PhantomData,
        }
    }
}

impl<Val, Dom, In, LDE, M, L, Chal> PCS<Val, In, Chal> for LDTBasedPCS<Val, Dom, LDE, M, L>
where
    Val: Field,
    Dom: ExtensionField<Val> + TwoAdicField,
    In: for<'a> MatrixRows<'a, Val>,
    LDE: TwoAdicLDE<Val, Dom>,
    M: DirectMMCS<Dom>,
    L: LDT<Dom, M, Chal>,
    Chal: FieldChallenger<Val> + FieldChallenger<Dom>,
{
    type Commitment = M::Commitment;
    type ProverData = M::ProverData;
    type Proof = L::Proof;
    type Error = L::Error;

    fn commit_batches(&self, polynomials: Vec<In>) -> (Self::Commitment, Self::ProverData) {
        // TODO: Streaming?
        let ldes = polynomials
            .into_iter()
            .map(|poly| {
                self.lde
                    .lde_batch(poly.to_row_major_matrix(), self.added_bits)
            })
            .collect();
        self.mmcs.commit(ldes)
    }
}

impl<Val, Dom, In, LDE, M, L, Chal> UnivariatePCS<Val, In, Chal>
    for LDTBasedPCS<Val, Dom, LDE, M, L>
where
    Val: Field,
    Dom: ExtensionField<Val> + TwoAdicField,
    In: for<'a> MatrixRows<'a, Val>,
    LDE: TwoAdicLDE<Val, Dom>,
    M: DirectMMCS<Dom>,
    L: LDT<Dom, M, Chal>,
    Chal: FieldChallenger<Val> + FieldChallenger<Dom>,
{
    fn open_multi_batches<EF>(
        &self,
        _prover_data: &[&Self::ProverData],
        _points: &[EF],
        _challenger: &mut Chal,
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
