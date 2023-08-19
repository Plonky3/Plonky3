use p3_air::{AirBuilder, TwoRowMatrixView};

use crate::StarkConfig;

pub struct ConstraintFolder<'a, SC: StarkConfig> {
    pub(crate) main: TwoRowMatrixView<'a, SC::PackedDomain>,
    pub(crate) is_first_row: SC::PackedDomain,
    pub(crate) is_last_row: SC::PackedDomain,
    pub(crate) is_transition: SC::PackedDomain,
    pub(crate) alpha: SC::Challenge,
    pub(crate) accumulator: SC::PackedChallenge,
}

impl<'a, SC: StarkConfig> AirBuilder for ConstraintFolder<'a, SC> {
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
