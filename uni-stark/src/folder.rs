use p3_air::{AirBuilder, TwoRowMatrixView};
use p3_field::Field;

use crate::StarkConfig;

pub struct ProverConstraintFolder<'a, SC: StarkConfig> {
    pub(crate) main: TwoRowMatrixView<'a, SC::PackedDomain>,
    pub(crate) is_first_row: SC::PackedDomain,
    pub(crate) is_last_row: SC::PackedDomain,
    pub(crate) is_transition: SC::PackedDomain,
    pub(crate) alpha: SC::Challenge,
    pub(crate) accumulator: SC::PackedChallenge,
}

pub struct VerifierConstraintFolder<'a, Challenge> {
    pub(crate) main: TwoRowMatrixView<'a, Challenge>,
    pub(crate) is_first_row: Challenge,
    pub(crate) is_last_row: Challenge,
    pub(crate) is_transition: Challenge,
    pub(crate) alpha: Challenge,
    pub(crate) accumulator: Challenge,
}

impl<'a, SC: StarkConfig> AirBuilder for ProverConstraintFolder<'a, SC> {
    type F = SC::Domain;
    type Expr = SC::PackedDomain;
    type Var = SC::PackedDomain;
    type M = TwoRowMatrixView<'a, SC::PackedDomain>;

    fn main(&self) -> Self::M {
        self.main
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
        let x: SC::PackedDomain = x.into();
        self.accumulator *= self.alpha;
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
