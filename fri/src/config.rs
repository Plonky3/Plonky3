use core::marker::PhantomData;

use p3_commit::{DirectMMCS, MMCS};
use p3_field::{ExtensionField, PrimeField64, TwoAdicField};

pub trait FriConfig {
    type Val: PrimeField64;
    type Challenge: ExtensionField<Self::Val> + TwoAdicField;

    type InputMmcs: MMCS<Self::Val>;
    type CommitPhaseMmcs: DirectMMCS<Self::Challenge>;

    fn input_mmcs(&self) -> &Self::InputMmcs;
    fn commit_phase_mmcs(&self) -> &Self::CommitPhaseMmcs;

    fn num_queries(&self) -> usize;
    // TODO: grinding bits
}

pub struct FriConfigImpl<Val, Challenge, InputMmcs, CommitPhaseMmcs> {
    num_queries: usize,
    input_mmcs: InputMmcs,
    commit_phase_mmcs: CommitPhaseMmcs,
    _phantom_val: PhantomData<Val>,
    _phantom_challenge: PhantomData<Challenge>,
}

impl<Val, Challenge, InputMmcs, CommitPhaseMmcs>
    FriConfigImpl<Val, Challenge, InputMmcs, CommitPhaseMmcs>
{
    pub fn new(
        num_queries: usize,
        input_mmcs: InputMmcs,
        commit_phase_mmcs: CommitPhaseMmcs,
    ) -> Self {
        Self {
            num_queries,
            input_mmcs,
            commit_phase_mmcs,
            _phantom_val: PhantomData,
            _phantom_challenge: PhantomData,
        }
    }
}

impl<Val, Challenge, InputMmcs, CommitPhaseMmcs> FriConfig
    for FriConfigImpl<Val, Challenge, InputMmcs, CommitPhaseMmcs>
where
    Val: PrimeField64,
    Challenge: ExtensionField<Val> + TwoAdicField,
    InputMmcs: MMCS<Val>,
    CommitPhaseMmcs: DirectMMCS<Challenge>,
{
    type Val = Val;
    type Challenge = Challenge;
    type InputMmcs = InputMmcs;
    type CommitPhaseMmcs = CommitPhaseMmcs;

    fn input_mmcs(&self) -> &InputMmcs {
        &self.input_mmcs
    }

    fn commit_phase_mmcs(&self) -> &CommitPhaseMmcs {
        &self.commit_phase_mmcs
    }

    fn num_queries(&self) -> usize {
        self.num_queries
    }
}
