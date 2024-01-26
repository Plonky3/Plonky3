use core::marker::PhantomData;

use alloc::vec::Vec;
use p3_challenger::FieldChallenger;
use p3_commit::{DirectMmcs, Mmcs, Pcs, UnivariatePcs};
use p3_field::{ExtensionField, Field};
use p3_matrix::MatrixRows;
use serde::{Deserialize, Serialize};

use crate::{FriConfig, FriProof};

pub struct TwoAdicFriPcs<FC, Val, Dft, M> {
    fri_config: FC,
    dft: Dft,
    mmcs: M,
    _phantom: PhantomData<Val>,
}

#[derive(Serialize, Deserialize)]
pub struct TwoAdicFriPcsProof<FC: FriConfig> {
    fri_proof: FriProof<FC>,
}

impl<FC, Val, Dft, M, In> Pcs<Val, In> for TwoAdicFriPcs<FC, Val, Dft, M>
where
    Val: Field,
    M: Mmcs<Val>,
    In: MatrixRows<Val>,
{
    type Commitment = M::Commitment;
    type ProverData = M::ProverData;
    type Proof = ();
    type Error = ();

    fn commit_batches(&self, polynomials: Vec<In>) -> (Self::Commitment, Self::ProverData) {
        todo!()
    }
}

/*
impl<Val, Challenge, In, Challenger> UnivariatePcs<Val, Challenge, In, Challenger> for TwoAdicFriPcs
where
    Val: Field,
    Challenge: ExtensionField<Val>,
    In: MatrixRows<Val>,
    Challenger: FieldChallenger<Val>,
{
    fn open_multi_batches(
        &self,
        prover_data_and_points: &[(&Self::ProverData, &[Vec<Challenge>])],
        challenger: &mut Challenger,
    ) -> (p3_commit::OpenedValues<Challenge>, Self::Proof) {
        todo!()
    }
    fn verify_multi_batches(
        &self,
        commits_and_points: &[(Self::Commitment, &[Vec<Challenge>])],
        dims: &[Vec<p3_matrix::Dimensions>],
        values: p3_commit::OpenedValues<Challenge>,
        proof: &Self::Proof,
        challenger: &mut Challenger,
    ) -> Result<(), Self::Error> {
        todo!()
    }
}

impl<Val, Challenge, In, Challenger> UnivariatePcsWithLde<Val, Challenge, In, Challenger>
    for TwoAdicFriPcs
{
    type Lde<'a>
    where
        Self: 'a;

    fn coset_shift(&self) -> Val {
        todo!()
    }

    fn log_blowup(&self) -> usize {
        todo!()
    }

    fn get_ldes<'a, 'b>(&'a self, prover_data: &'b Self::ProverData) -> Vec<Self::Lde<'b>>
    where
        'a: 'b,
    {
        todo!()
    }

    fn commit_shifted_batches(
        &self,
        polynomials: Vec<In>,
        coset_shift: Val,
    ) -> (Self::Commitment, Self::ProverData) {
        todo!()
    }
}
*/
