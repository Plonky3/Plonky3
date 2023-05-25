use core::marker::PhantomData;
use core::ops::{Add, Mul, Sub};
use p3_air::{AirBuilder, TwoRowMatrixView};
use p3_field::{AbstractionOf, Field};

pub struct BasicFoldingAirBuilder<'a, F, Expr, Var> {
    pub(crate) main: TwoRowMatrixView<'a, Var>,
    pub(crate) is_first_row: Expr,
    pub(crate) is_last_row: Expr,
    pub(crate) is_transition: Expr,
    pub(crate) _phantom_f: PhantomData<F>,
}

impl<'a, F, Expr, Var> AirBuilder for BasicFoldingAirBuilder<'a, F, Expr, Var>
where
    F: Field,
    Expr: AbstractionOf<F>
        + Add<Var, Output = Expr>
        + Sub<Var, Output = Expr>
        + Mul<Var, Output = Expr>,
    Var: Into<Expr>
        + Copy
        + Add<F, Output = Expr>
        + Add<Var, Output = Expr>
        + Add<Expr, Output = Expr>
        + Sub<F, Output = Expr>
        + Sub<Var, Output = Expr>
        + Sub<Expr, Output = Expr>
        + Mul<F, Output = Expr>
        + Mul<Var, Output = Expr>
        + Mul<Expr, Output = Expr>,
{
    type F = F;
    type Expr = Expr;
    type Var = Var;
    type M = TwoRowMatrixView<'a, Var>;

    fn main(&self) -> Self::M {
        self.main
    }

    fn is_first_row(&self) -> Self::Expr {
        self.is_first_row.clone()
    }

    fn is_last_row(&self) -> Self::Expr {
        self.is_last_row.clone()
    }

    fn is_transition_window(&self, size: usize) -> Self::Expr {
        if size == 2 {
            self.is_transition.clone()
        } else {
            todo!()
        }
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        todo!()
    }
}
