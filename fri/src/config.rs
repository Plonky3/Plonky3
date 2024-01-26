use core::marker::PhantomData;

use p3_challenger::{CanObserve, CanSample, GrindingChallenger};
use p3_commit::{DirectMmcs, Mmcs};
use p3_field::TwoAdicField;

pub trait FriConfig {
    type Challenge: TwoAdicField;

    type CommitPhaseMmcs: DirectMmcs<Self::Challenge>;

    type Challenger: GrindingChallenger
        + CanObserve<<Self::CommitPhaseMmcs as Mmcs<Self::Challenge>>::Commitment>
        + CanSample<Self::Challenge>;

    fn commit_phase_mmcs(&self) -> &Self::CommitPhaseMmcs;

    fn num_queries(&self) -> usize;

    fn log_blowup(&self) -> usize;

    fn blowup(&self) -> usize {
        1 << self.log_blowup()
    }

    fn proof_of_work_bits(&self) -> usize;
}

pub struct FriConfigImpl<Challenge, CommitPhaseMmcs, Challenger> {
    log_blowup: usize,
    num_queries: usize,
    proof_of_work_bits: usize,
    commit_phase_mmcs: CommitPhaseMmcs,
    _phantom: PhantomData<(Challenge, Challenger)>,
}

impl<Challenge, CommitPhaseMmcs, Challenger> FriConfigImpl<Challenge, CommitPhaseMmcs, Challenger> {
    pub fn new(
        log_blowup: usize,
        num_queries: usize,
        proof_of_work_bits: usize,
        commit_phase_mmcs: CommitPhaseMmcs,
    ) -> Self {
        Self {
            log_blowup,
            num_queries,
            commit_phase_mmcs,
            proof_of_work_bits,
            _phantom: PhantomData,
        }
    }
}

impl<Challenge, CommitPhaseMmcs, Challenger> FriConfig
    for FriConfigImpl<Challenge, CommitPhaseMmcs, Challenger>
where
    Challenge: TwoAdicField,
    CommitPhaseMmcs: DirectMmcs<Challenge>,
    Challenger: GrindingChallenger
        + CanObserve<<CommitPhaseMmcs as Mmcs<Challenge>>::Commitment>
        + CanSample<Challenge>,
{
    type Challenge = Challenge;
    type CommitPhaseMmcs = CommitPhaseMmcs;
    type Challenger = Challenger;

    fn commit_phase_mmcs(&self) -> &CommitPhaseMmcs {
        &self.commit_phase_mmcs
    }

    fn num_queries(&self) -> usize {
        self.num_queries
    }

    fn log_blowup(&self) -> usize {
        self.log_blowup
    }

    fn proof_of_work_bits(&self) -> usize {
        self.proof_of_work_bits
    }
}
