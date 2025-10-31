//! Functions and structures that return or store common data shared by the prover and the verifier.
//!
//! For example, we get the lookups before calling either `prove_multi` or `verify_multi`, and we can pass the lookup data to both the prover and verifier.

use crate::{Challenge, config::SGC};
use alloc::vec::Vec;
use p3_field::{BasedVectorSpace, Field};
use p3_lookup::lookup_traits::{AirLookupHandler, Lookup};
use p3_uni_stark::{SymbolicAirBuilder, SymbolicExpression, Val};

/// Struct storing data common to both the prover and verifier.
pub struct CommonData<F: Field> {
    /// The lookups used by each STARK instance.
    pub lookups: Vec<Lookup<F>>,
}

impl<F: Field> CommonData<F> {
    pub fn new(lookups: Vec<Lookup<F>>) -> Self {
        Self { lookups }
    }
}

/// Function to extract lookups from multiple AIRs.
pub fn extract_lookups<SC, A>(airs: &mut [A]) -> Vec<Vec<Lookup<Val<SC>>>>
where
    SC: SGC,
    SymbolicExpression<SC::Challenge>: From<SymbolicExpression<Val<SC>>>,
    A: AirLookupHandler<SymbolicAirBuilder<Val<SC>, SC::Challenge>>,
    Challenge<SC>: BasedVectorSpace<Val<SC>>,
{
    let mut all_lookups = Vec::with_capacity(airs.len());
    for air in airs {
        let air_lookups =
            <A as AirLookupHandler<SymbolicAirBuilder<Val<SC>, SC::Challenge>>>::get_lookups(air);
        all_lookups.push(air_lookups);
    }
    all_lookups
}
