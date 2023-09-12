use alloc::vec;
use core::fmt;

use p3_air::Air;
use p3_challenger::FieldChallenger;
use p3_commit::UnivariatePcs;
use p3_field::{AbstractField, TwoAdicField};

use crate::{ConstraintFolder, Proof, StarkConfig};

pub fn verify<SC, A>(
    config: &SC,
    _air: &A,
    challenger: &mut SC::Challenger,
    proof: &Proof<SC>,
) -> Result<(), VerificationError>
where
    SC: StarkConfig,
    A: for<'a> Air<ConstraintFolder<'a, SC>>,
{
    let degree_bits = 0; // TODO
    let g_subgroup = SC::Domain::primitive_root_of_unity(degree_bits);
    let zeta: SC::Challenge = challenger.sample_ext_element();
    let Proof {
        commitments,
        opened_values,
        opening_proof,
    } = proof;
    let local_and_next = [zeta, zeta * g_subgroup];
    let commits_and_points = &[
        (commitments.trace.clone(), local_and_next.as_slice()),
        (commitments.quotient_chunks.clone(), &[zeta.square()]),
    ];
    let values = vec![
        vec![vec![
            opened_values.trace_local.clone(),
            opened_values.trace_next.clone(),
        ]],
        vec![vec![opened_values.quotient_chunks.clone()]],
    ];
    config
        .pcs()
        .verify_multi_batches(commits_and_points, values, opening_proof)
        .map_err(|_| VerificationError)?;
    Ok(())
}

#[derive(Debug)]
pub struct VerificationError;

impl fmt::Display for VerificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Proof verification failed")
    }
}
