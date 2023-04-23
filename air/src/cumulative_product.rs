use crate::virtual_column::VirtualColumn;
use crate::window::PermutationWindow;
use crate::{Air, AirTypes, AirWindow, ConstraintConsumer};
use alloc::vec::Vec;
use core::marker::PhantomData;
use p3_field::field::Field;
use p3_matrix::Matrix;

pub struct CumulativeProductAir<T, W, CC, Inner>
where
    T: AirTypes,
    W: AirWindow<T::Var>,
    CC: ConstraintConsumer<T>,
    Inner: Air<T, W, CC>,
{
    inner: Inner,
    updates: Vec<CumulativeProductUpdate<T::F>>,
    _phantom_w: PhantomData<W>,
    _phantom_cc: PhantomData<CC>,
}

impl<T, W, CC, Inner> Air<T, W, CC> for CumulativeProductAir<T, W, CC, Inner>
where
    T: AirTypes,
    W: PermutationWindow<T::Var>,
    CC: ConstraintConsumer<T>,
    Inner: Air<T, W, CC>,
{
    fn eval(&self, window: &W, constraints: &mut CC) {
        self.inner.eval(window, constraints);
        let main_local = window.main().row(0);
        let perm_local = window.permutation().row(0);
        let perm_next = window.permutation().row(1);

        for update in &self.updates {
            let one = T::F::ONE;

            // TODO: Get verifier randomness from the window. Will be extension field elements.
            let alpha = T::Exp::from(one);
            let gamma = T::Exp::from(one);

            // TODO: Get from configurable part of window. Will be extension field elements.
            let z_local = perm_local[0];
            let z_next = perm_next[0];

            let filter = update.filter.apply::<T>(main_local);
            let applied_terms = update.terms.iter().map(|x| x.apply::<T>(perm_local));
            let reduced_term = applied_terms
                .rev()
                .reduce(|acc, x| acc * alpha.clone() + x)
                .unwrap()
                + gamma;

            let multiply = filter * (reduced_term - one) + one;
            constraints.global(z_next - z_local * multiply);
        }
    }
}

pub struct CumulativeProductUpdate<F: Field> {
    pub filter: VirtualColumn<F>,
    pub terms: Vec<VirtualColumn<F>>,
}
