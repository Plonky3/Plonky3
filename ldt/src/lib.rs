//! This crate contains a framework for low-degree tests (LDTs).

#![no_std]

extern crate alloc;

use alloc::vec::Vec;
use core::marker::PhantomData;
use p3_challenger::Challenger;
use p3_commit::{DirectMMCS, MMCS};
use p3_commit::{UnivariatePCS, PCS};
use p3_field::{AbstractExtensionField, ExtensionField, Field, TwoAdicField};
use p3_lde::TwoAdicLDE;
use p3_matrix::dense::RowMajorMatrix;

/// A batch low-degree test (LDT).
pub trait LDT<F: Field, M: MMCS<F>> {
    type Proof;
    type Error;

    /// Prove that each column of each matrix in `codewords` is a codeword.
    fn prove<Chal>(&self, codewords: &[M::ProverData], challenger: &mut Chal) -> Self::Proof
    where
        Chal: Challenger<F>;

    fn verify<Chal>(
        &self,
        codeword_commits: &[M::Commitment],
        proof: &Self::Proof,
        challenger: &mut Chal,
    ) -> Result<(), Self::Error>
    where
        Chal: Challenger<F>;
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

impl<Val, Dom, LDE, M, L> PCS<Val> for LDTBasedPCS<Val, Dom, LDE, M, L>
where
    Val: Field,
    Dom: ExtensionField<Val> + TwoAdicField,
    LDE: TwoAdicLDE<Val, Dom>,
    M: DirectMMCS<Dom, Mat = RowMajorMatrix<LDE::Res>>,
    L: LDT<Dom, M>,
{
    type Commitment = M::Commitment;
    type ProverData = M::ProverData;
    type Proof = L::Proof;
    type Error = L::Error;

    fn commit_batches(
        &self,
        polynomials: Vec<RowMajorMatrix<Val>>,
    ) -> (Self::Commitment, Self::ProverData) {
        // TODO: Streaming?
        let ldes = polynomials
            .into_iter()
            .map(|poly| self.lde.lde_batch(poly, self.added_bits))
            .collect();
        self.mmcs.commit(ldes)
    }

    fn get_committed_value(
        &self,
        _prover_data: &Self::ProverData,
        _poly: usize,
        _value: usize,
    ) -> Val {
        todo!()
    }
}

impl<Val, Dom, LDE, M, L> UnivariatePCS<Val> for LDTBasedPCS<Val, Dom, LDE, M, L>
where
    Val: Field,
    Dom: ExtensionField<Val> + TwoAdicField,
    LDE: TwoAdicLDE<Val, Dom>,
    M: DirectMMCS<Dom, Mat = RowMajorMatrix<LDE::Res>>,
    L: LDT<Dom, M>,
{
    fn open_multi_batches<EF, Chal>(
        &self,
        _prover_data: &[Self::ProverData],
        _points: &[EF],
        _challenger: &mut Chal,
    ) -> (Vec<Vec<Vec<EF>>>, Self::Proof)
    where
        EF: AbstractExtensionField<Val>,
        Chal: Challenger<Val>,
    {
        todo!()
    }

    fn verify_multi_batches<EF, Chal>(
        &self,
        _commits: &[Self::Commitment],
        _points: &[EF],
        _values: &[Vec<Vec<EF>>],
        _proof: &Self::Proof,
    ) -> Result<(), Self::Error>
    where
        EF: AbstractExtensionField<Val>,
        Chal: Challenger<Val>,
    {
        todo!()
    }
}
