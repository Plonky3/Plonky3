// use core::ops::{Add, Mul, Sub};
//
// /// Provides an evaluation frame, performs arithmetic on variables, and consumes constraints.
// pub trait Evaluator<'a> {
//     type Exp: Add<Self::Exp, Output = Self::Exp>
//         + Sub<Self::Exp, Output = Self::Exp>
//         + Mul<Self::Exp, Output = Self::Exp>;
//
//     fn trace_local(&self) -> &'a [Self::Exp];
//     fn trace_next(&self) -> &'a [Self::Exp];
//
//     fn add(&mut self, x: Self::Exp, y: Self::Exp) -> Self::Exp;
//     fn sub(&mut self, x: Self::Exp, y: Self::Exp) -> Self::Exp;
//     fn mul(&mut self, x: Self::Exp, y: Self::Exp) -> Self::Exp;
//
//     fn when(&'a mut self, filter: Self::Exp) -> FilteredEvaluator<'a, Self> {
//         FilteredEvaluator {
//             inner: self,
//             filter,
//         }
//     }
//
//     fn assert_zero(&mut self, x: Self::Exp);
//
//     fn assert_eq(&mut self, x: Self::Exp, y: Self::Exp) {
//         let diff = self.sub(x, y);
//         self.assert_zero(diff);
//     }
// }
//
// pub struct FilteredEvaluator<'a, E: Evaluator<'a> + ?Sized> {
//     inner: &'a mut E,
//     filter: E::Exp,
// }
//
// pub struct NativeEvaluator<'a> {}
