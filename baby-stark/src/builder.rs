use core::marker::PhantomData;
use core::ops::{Add, Mul, Sub};
use p3_air::{AirBuilder, TwoRowMatrixView};
use p3_field::{AbstractionOf, Field};

pub struct BasicFoldingAirBuilder<'a, F, Exp, Var> {
    pub(crate) main: TwoRowMatrixView<'a, Var>,
    pub(crate) is_first_row: Exp,
    pub(crate) is_last_row: Exp,
    pub(crate) is_transition: Exp,
    pub(crate) _phantom_f: PhantomData<F>,
}

impl<'a, F, Exp, Var> AirBuilder for BasicFoldingAirBuilder<'a, F, Exp, Var>
where
    F: Field,
    Exp:
        AbstractionOf<F> + Add<Var, Output = Exp> + Sub<Var, Output = Exp> + Mul<Var, Output = Exp>,
    Var: Into<Exp>
        + Copy
        + Add<F, Output = Exp>
        + Add<Var, Output = Exp>
        + Add<Exp, Output = Exp>
        + Sub<F, Output = Exp>
        + Sub<Var, Output = Exp>
        + Sub<Exp, Output = Exp>
        + Mul<F, Output = Exp>
        + Mul<Var, Output = Exp>
        + Mul<Exp, Output = Exp>,
{
    type F = F;
    type Exp = Exp;
    type Var = Var;
    type M = TwoRowMatrixView<'a, Var>;

    fn main(&self) -> Self::M {
        self.main
    }

    fn is_first_row(&self) -> Self::Exp {
        self.is_first_row.clone()
    }

    fn is_last_row(&self) -> Self::Exp {
        self.is_last_row.clone()
    }

    fn is_transition_window(&self, size: usize) -> Self::Exp {
        if size == 2 {
            self.is_transition.clone()
        } else {
            todo!()
        }
    }

    fn assert_zero<I: Into<Self::Exp>>(&mut self, x: I) {
        todo!()
    }
}
