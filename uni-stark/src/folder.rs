use alloc::vec::Vec;

use p3_air::{AirBuilder, TwoRowMatrixView};
use p3_field::{AbstractField, Field};

use crate::{PackedChallenge, PackedVal, StarkGenericConfig, Val};

pub struct ProverConstraintFolder<'a, SC: StarkGenericConfig> {
    pub main: TwoRowMatrixView<'a, PackedVal<SC>>,
    pub public_inputs: &'a Vec<Val<SC>>,
    pub is_first_row: PackedVal<SC>,
    pub is_last_row: PackedVal<SC>,
    pub is_transition: PackedVal<SC>,
    pub alpha: SC::Challenge,
    pub accumulator: PackedChallenge<SC>,
}

pub struct VerifierConstraintFolder<'a, Challenge> {
    pub main: TwoRowMatrixView<'a, Challenge>,
    pub public_inputs: &'a Vec<Challenge>,
    pub is_first_row: Challenge,
    pub is_last_row: Challenge,
    pub is_transition: Challenge,
    pub alpha: Challenge,
    pub accumulator: Challenge,
}

impl<'a, SC: StarkGenericConfig> AirBuilder for ProverConstraintFolder<'a, SC> {
    type F = Val<SC>;
    type Expr = PackedVal<SC>;
    type Var = PackedVal<SC>;
    type M = TwoRowMatrixView<'a, PackedVal<SC>>;

    fn main(&self) -> Self::M {
        self.main
    }

    fn public_inputs(&self) -> Vec<Self::F> {
        self.public_inputs.clone()
    }

    fn is_first_row(&self) -> Self::Expr {
        self.is_first_row
    }

    fn is_last_row(&self) -> Self::Expr {
        self.is_last_row
    }

    fn is_transition_window(&self, size: usize) -> Self::Expr {
        if size == 2 {
            self.is_transition
        } else {
            panic!("uni-stark only supports a window size of 2")
        }
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        let x: PackedVal<SC> = x.into();
        self.accumulator *= PackedChallenge::<SC>::from_f(self.alpha);
        self.accumulator += x;
    }
}

impl<'a, Challenge: Field> AirBuilder for VerifierConstraintFolder<'a, Challenge> {
    type F = Challenge;
    type Expr = Challenge;
    type Var = Challenge;
    type M = TwoRowMatrixView<'a, Challenge>;

    fn main(&self) -> Self::M {
        self.main
    }

    fn public_inputs(&self) -> Vec<Self::F> {
        self.public_inputs.clone()
    }

    fn is_first_row(&self) -> Self::Expr {
        self.is_first_row
    }

    fn is_last_row(&self) -> Self::Expr {
        self.is_last_row
    }

    fn is_transition_window(&self, size: usize) -> Self::Expr {
        if size == 2 {
            self.is_transition
        } else {
            panic!("uni-stark only supports a window size of 2")
        }
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        let x: Challenge = x.into();
        self.accumulator *= self.alpha;
        self.accumulator += x;
    }
}
