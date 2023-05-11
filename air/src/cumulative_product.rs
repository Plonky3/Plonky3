// use crate::virtual_column::VirtualColumn;
// use crate::window::PermutationWindow;
// use crate::{Air, AirTypes, AirWindow, ConstraintConsumer};
// use alloc::vec::Vec;
// use core::marker::PhantomData;
// use p3_field::{Field, FieldLike};
// use p3_matrix::Matrix;
//
// pub struct CumulativeProductAir<T, W, Inner>
// where
//     T: AirTypes,
//     W: AirWindow<T>,
//     Inner: Air<T, W>,
// {
//     inner: Inner,
//     updates: Vec<CumulativeProductUpdate<T::F>>,
//     _phantom_w: PhantomData<W>,
// }
//
// impl<T, W, Inner> Air<T, W> for CumulativeProductAir<T, W, Inner>
// where
//     T: AirTypes,
//     W: PermutationWindow<T>,
//     Inner: Air<T, W>,
// {
//     fn eval<CC>(&self, constraints: &mut CC)
//     where
//         CC: ConstraintConsumer<T, W>,
//     {
//         self.inner.eval(constraints);
//         let main = constraints.window().main();
//         let main_local = main.row(0);
//         let permutation = constraints.window().permutation();
//         let perm_local = permutation.row(0);
//         let perm_next = permutation.row(1);
//
//         for update in &self.updates {
//             let one = T::F::ONE;
//
//             // TODO: Get verifier randomness from the window. Will be extension field elements.
//             let alpha = T::Exp::from(one);
//             let gamma = T::Exp::from(one);
//
//             // TODO: Get from configurable part of window. Will be extension field elements.
//             let z_local = perm_local[0];
//             let z_next = perm_next[0];
//
//             let filter = update.filter.apply::<T>(main_local);
//             let applied_terms = update.terms.iter().map(|x| x.apply::<T>(perm_local));
//             let reduced_term = applied_terms
//                 .rev()
//                 .reduce(|acc, x| acc * alpha.clone() + x)
//                 .unwrap()
//                 + gamma;
//
//             let multiply = filter * (reduced_term - one) + one;
//             constraints.assert_zero(z_next - z_local * multiply);
//         }
//     }
// }
//
// pub struct CumulativeProductUpdate<F: Field> {
//     pub filter: VirtualColumn<F>,
//     pub terms: Vec<VirtualColumn<F>>,
// }
