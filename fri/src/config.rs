use core::marker::PhantomData;

use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{DirectMmcs, Mmcs};
use p3_field::{ExtensionField, PrimeField64, TwoAdicField};

pub trait FriConfig {
    type Val: PrimeField64;
    type Challenge: ExtensionField<Self::Val> + TwoAdicField;

    type InputMmcs: Mmcs<Self::Val>;
    type CommitPhaseMmcs: DirectMmcs<Self::Challenge>;

    type Challenger: FieldChallenger<Self::Val>
        + CanObserve<<Self::InputMmcs as Mmcs<Self::Val>>::Commitment>
        + CanObserve<<Self::CommitPhaseMmcs as Mmcs<Self::Challenge>>::Commitment>;

    fn input_mmcs(&self) -> &Self::InputMmcs;
    fn commit_phase_mmcs(&self) -> &Self::CommitPhaseMmcs;

    fn num_queries(&self) -> usize;
    // TODO: grinding bits
}

pub struct FriConfigImpl<Val, Challenge, InputMmcs, CommitPhaseMmcs, Challenger> {
    num_queries: usize,
    input_mmcs: InputMmcs,
    commit_phase_mmcs: CommitPhaseMmcs,
    _phantom_val: PhantomData<Val>,
    _phantom_challenge: PhantomData<Challenge>,
    _phantom_chal: PhantomData<Challenger>,
}

impl<Val, Challenge, InputMmcs, CommitPhaseMmcs, Challenger>
    FriConfigImpl<Val, Challenge, InputMmcs, CommitPhaseMmcs, Challenger>
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
            _phantom_chal: PhantomData,
        }
    }
}

impl<Val, Challenge, InputMmcs, CommitPhaseMmcs, Challenger> FriConfig
    for FriConfigImpl<Val, Challenge, InputMmcs, CommitPhaseMmcs, Challenger>
where
    Val: PrimeField64,
    Challenge: ExtensionField<Val> + TwoAdicField,
    InputMmcs: Mmcs<Val>,
    CommitPhaseMmcs: DirectMmcs<Challenge>,
    Challenger: FieldChallenger<Val>
        + CanObserve<<InputMmcs as Mmcs<Val>>::Commitment>
        + CanObserve<<CommitPhaseMmcs as Mmcs<Challenge>>::Commitment>,
{
    type Val = Val;
    type Challenge = Challenge;
    type InputMmcs = InputMmcs;
    type CommitPhaseMmcs = CommitPhaseMmcs;
    type Challenger = Challenger;

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
