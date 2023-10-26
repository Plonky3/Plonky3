use itertools::Itertools;
use p3_challenger::{CanObserve, CanSampleBits};

use crate::{FriConfig, FriProof, QueryProof};

pub(crate) fn verify<FC: FriConfig>(
    config: &FC,
    input_mmcs: &[FC::InputMmcs],
    proof: &FriProof<FC>,
    challenger: &mut FC::Challenger,
) -> Result<(), ()> {
    for com in &proof.commit_phase_commits {
        challenger.observe(com.clone());
    }

    if proof.query_proofs.len() != config.num_queries() {
        return Err(());
    }

    let log_max_height = 0; // TODO
    let query_indices = (0..config.num_queries()).map(|_| challenger.sample_bits(log_max_height));

    for (index, proof) in query_indices.zip_eq(&proof.query_proofs) {
        verify_query(config, input_mmcs, index, proof)?;
    }

    Ok(())
}

fn verify_query<FC: FriConfig>(
    _config: &FC,
    _input_mmcs: &[FC::InputMmcs],
    _index: usize,
    _proof: &QueryProof<FC>,
) -> Result<(), ()> {
    // TODO
    Ok(())
}
