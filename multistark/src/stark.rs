use crate::config::Config;
use crate::constraint_consumer::ConstraintConsumer;
use crate::variables::StarkEvaluationVars;
use hyperfield::field::Field;
use hyperfield::trivial_extension::TrivialExtension;

pub trait Stark<C: Config> {
    fn columns(&self) -> usize;

    fn eval_packed_base(
        &self,
        vars: StarkEvaluationVars<<C::F as Field>::Packing>,
        constraints: &mut ConstraintConsumer<C::F, C::FE, <C::F as Field>::Packing>,
    );

    fn eval_ext(
        &self,
        vars: StarkEvaluationVars<C::FE>,
        constraints: &mut ConstraintConsumer<C::FE, TrivialExtension<C::FE>, C::FE>,
    );
}
