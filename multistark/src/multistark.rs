use crate::config::Config;
use crate::stark::Stark;

pub trait MultiStark<C: Config, const N: usize> {
    fn starks(&self) -> [&dyn Stark<C>; N];
}

// #[cfg(test)]
// mod tests {
//     use crate::config::Config;
//     use crate::constraint_consumer::ConstraintConsumer;
//     use crate::multistark::MultiStark;
//     use crate::stark::Stark;
//     use crate::variables::StarkEvaluationVars;
//     use core::marker::PhantomData;
//     use hyperfield::field::{Field, FieldExtension};
//     use hyperfield::packed::PackedField;
//
//     struct FooStark<C: Config> {
//         _phantom: PhantomData<C>,
//     }
//
//     impl<C: Config> FooStark<C> {
//         fn eval<FE, P>(
//             &self,
//             vars: StarkEvaluationVars<P>,
//             constraints: &mut ConstraintConsumer<FE, C::FE, P>,
//         ) where
//             FE: FieldExtension<C::F>,
//             // C::FE: FieldExtension<FE>,
//             P: PackedField<Scalar = C::F>,
//         {
//             constraints.constraint(vars.local_values[0] + vars.local_values[1]);
//         }
//     }
//
//     impl<C: Config> Stark<C> for FooStark<C> {
//         fn columns(&self) -> usize {
//             10
//         }
//
//         fn eval_packed_base(
//             &self,
//             vars: StarkEvaluationVars<<C::F as Field>::Packing>,
//             constraints: &mut ConstraintConsumer<C::F, C::FE, <C::F as Field>::Packing>,
//         ) {
//             self.eval::<C::F, C::F::Packing>(vars, constraints)
//         }
//
//         fn eval_ext(
//             &self,
//             vars: StarkEvaluationVars<C::FE>,
//             constraints: &mut ConstraintConsumer<C::FE, C::FE, C::FE>,
//         ) {
//             self.eval::<C::FE, C::FE>(vars, constraints)
//         }
//     }
//
//     // struct BarStark<C: Config> {
//     //     _phantom: PhantomData<C>,
//     // }
//     //
//     // impl<C: Config> BarStark<C> {
//     //     fn eval<FE, P>(
//     //         &self,
//     //         vars: StarkEvaluationVars<P>,
//     //         constraints: &mut ConstraintConsumer<FE, C::FE, P>,
//     //     ) where
//     //         FE: Field<Base = C::F>,
//     //         C::FE: Field<Base = FE>,
//     //         P: PackedField<Scalar = FE>,
//     //     {
//     //         constraints.constraint(vars.local_values[0] + vars.local_values[1]);
//     //     }
//     // }
//     //
//     // impl<C: Config> Stark<C> for BarStark<C> {
//     //     fn columns(&self) -> usize {
//     //         10
//     //     }
//     //
//     //     fn eval_packed_base(
//     //         &self,
//     //         vars: StarkEvaluationVars<C::P>,
//     //         constraints: &mut ConstraintConsumer<C::F, C::FE, C::P>,
//     //     ) {
//     //         self.eval::<C::F, C::P>(vars, constraints)
//     //     }
//     //
//     //     fn eval_ext(
//     //         &self,
//     //         vars: StarkEvaluationVars<C::FE>,
//     //         constraints: &mut ConstraintConsumer<C::FE, C::FE, C::PE>,
//     //     ) {
//     //         self.eval::<C::FE, C::FE>(vars, constraints)
//     //     }
//     // }
//     //
//     // struct FooBarStarks<C: Config> {
//     //     foo: FooStark<C>,
//     //     bar: BarStark<C>,
//     // }
//     //
//     // impl<C: Config> MultiStark<C, 2> for FooBarStarks<C> {
//     //     fn starks(&self) -> [&dyn Stark<C>; 2] {
//     //         [&self.foo, &self.bar]
//     //     }
//     // }
//
//     #[test]
//     fn foo() {}
// }
