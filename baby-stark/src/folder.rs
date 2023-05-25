use core::marker::PhantomData;
use p3_air::{AirBuilder, TwoRowMatrixView};
use p3_field::{ExtensionField, Field};

pub struct ConstraintFolder<'a, F, Challenge>
where
    F: Field,
{
    pub(crate) main: TwoRowMatrixView<'a, F::Packing>,
    pub(crate) is_first_row: F::Packing,
    pub(crate) is_last_row: F::Packing,
    pub(crate) is_transition: F::Packing,
    pub(crate) alpha: Challenge,
    pub(crate) accumulator: Challenge,
    pub(crate) _phantom_f: PhantomData<F>,
}

impl<'a, F, Challenge> AirBuilder for ConstraintFolder<'a, F, Challenge>
where
    F: Field,
    Challenge: ExtensionField<F>,
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
            panic!("baby-stark only supports a window size of 2")
        }
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        todo!()
    }
}
