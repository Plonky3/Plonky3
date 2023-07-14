//! An implementation of the FRI low-degree test (LDT).

#![no_std]

extern crate alloc;

use core::marker::PhantomData;

use p3_challenger::Challenger;
use p3_commit::{DirectMMCS, MMCS};
use p3_field::{ExtensionField, Field};
use p3_ldt::{LDTBasedPCS, LDT};

use crate::proof::FriProof;
use crate::prover::prove;
use crate::verifier::verify;

mod config;
mod proof;
mod prover;
mod verifier;

pub use config::*;
pub use proof::*;

pub struct FriLDT<F, Challenge, M, MC>
where
    F: Field,
    Challenge: ExtensionField<F>,
    M: MMCS<F>,
    MC: DirectMMCS<F>,
{
    config: FriConfig,
    _phantom_f: PhantomData<F>,
    _phantom_fe: PhantomData<Challenge>,
    _phantom_m: PhantomData<M>,
    _phantom_mc: PhantomData<MC>,
}

impl<F, Challenge, M, MC> LDT<F, M> for FriLDT<F, Challenge, M, MC>
where
    F: Field,
    Challenge: ExtensionField<F>,
    M: MMCS<F>,
    MC: DirectMMCS<F>,
{
    type Proof = FriProof<F, Challenge, M, MC>;
    type Error = ();

    fn prove<Chal>(&self, codewords: &[M::ProverData], challenger: &mut Chal) -> Self::Proof
    where
        Chal: Challenger<F>,
    {
        prove::<F, Challenge, M, MC, Chal>(codewords, &self.config, challenger)
    }

    fn verify<Chal>(
        &self,
        _codeword_commits: &[M::Commitment],
        proof: &Self::Proof,
        challenger: &mut Chal,
    ) -> Result<(), Self::Error>
    where
        Chal: Challenger<F>,
    {
        verify::<F, Challenge, M, MC, Chal>(proof, challenger)
    }
}

pub type FRIBasedPCS<Val, Dom, Challenge, LDE, M, MC> =
    LDTBasedPCS<Val, Dom, LDE, M, FriLDT<Dom, Challenge, M, MC>>;
