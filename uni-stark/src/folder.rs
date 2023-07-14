use core::marker::PhantomData;

use p3_air::{AirBuilder, TwoRowMatrixView};
use p3_field::{AbstractExtensionField, ExtensionField, Field, PackedField};

pub struct ConstraintFolder<'a, F, Challenge, PackedChallenge>
where
    F: Field,
{
    pub(crate) main: TwoRowMatrixView<'a, F::Packing>,
    pub(crate) is_first_row: F::Packing,
    pub(crate) is_last_row: F::Packing,
    pub(crate) is_transition: F::Packing,
    pub(crate) alpha: Challenge,
    pub(crate) accumulator: PackedChallenge,
    pub(crate) _phantom_f: PhantomData<F>,
}

impl<'a, F, Challenge, PackedChallenge> AirBuilder
    for ConstraintFolder<'a, F, Challenge, PackedChallenge>
where
    F: Field,
    Challenge: ExtensionField<F>,
    PackedChallenge: PackedField<Scalar = Challenge> + AbstractExtensionField<F::Packing>,
{
    type F = F;
    type Expr = F::Packing;
    type Var = F::Packing;
    type M = TwoRowMatrixView<'a, F::Packing>;

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
        let x: F::Packing = x.into();
        self.accumulator *= self.alpha;
        self.accumulator += x;
    }
}
