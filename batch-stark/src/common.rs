//! Functions and structures that return or store common data shared by the prover and the verifier.
//!
//! For example, we get the lookups before calling either `prove_multi` or `verify_multi`, and we can pass the lookup data to both the prover and verifier.

use crate::{Challenge, config::SGC};
use alloc::vec::Vec;
use hashbrown::HashMap;
use p3_challenger::FieldChallenger;
use p3_field::{BasedVectorSpace, Field};
use p3_lookup::lookup_traits::{AirLookupHandler, Kind, Lookup, LookupGadget};
use p3_uni_stark::{SymbolicAirBuilder, SymbolicExpression, Val};

/// Struct storing data common to both the prover and verifier.
/// TODO: Add preprocessed commitments.
pub struct CommonData<F: Field> {
    /// The lookups used by each STARK instance.
    pub lookups: Vec<Vec<Lookup<F>>>,
}

impl<F: Field> CommonData<F> {
    pub fn new(lookups: Vec<Vec<Lookup<F>>>) -> Self {
        Self { lookups }
    }
}

/// Function to extract lookups from multiple AIRs.
pub fn common_data<SC, A>(airs: &mut [A]) -> CommonData<Val<SC>>
where
    SC: SGC,
    SymbolicExpression<SC::Challenge>: From<SymbolicExpression<Val<SC>>>,
    A: AirLookupHandler<SymbolicAirBuilder<Val<SC>, SC::Challenge>>,
    Challenge<SC>: BasedVectorSpace<Val<SC>>,
{
    let mut lookups = Vec::with_capacity(airs.len());
    for air in airs {
        let air_lookups =
            <A as AirLookupHandler<SymbolicAirBuilder<Val<SC>, SC::Challenge>>>::get_lookups(air);
        lookups.push(air_lookups);
    }
    CommonData { lookups }
}

pub(crate) fn get_perm_challenges<'a, SC: SGC, LG: LookupGadget, A>(
    challenger: &mut SC::Challenger,
    all_lookups: &[Vec<Lookup<Val<SC>>>],
    airs: &[A],
    lookup_gadget: &LG,
) -> Vec<Vec<SC::Challenge>> {
    let num_challenges_per_lookup = lookup_gadget.num_challenges();
    let mut global_perm_challenges = HashMap::new();
    let mut challenges_per_instance = Vec::with_capacity(airs.len());
    for contexts in all_lookups {
        let num_challenges = contexts.len() * num_challenges_per_lookup;
        let mut instance_challenges = Vec::with_capacity(num_challenges);
        for context in contexts {
            let cs = match &context.kind {
                Kind::Global(name) => {
                    let cs = global_perm_challenges.entry(name).or_insert_with(|| {
                        (0..num_challenges_per_lookup)
                            .map(|_| challenger.sample_algebra_element())
                            .collect::<Vec<SC::Challenge>>()
                    });
                    cs.clone()
                }
                Kind::Local => (0..num_challenges_per_lookup)
                    .map(|_| challenger.sample_algebra_element())
                    .collect(),
            };
            instance_challenges.extend(cs);
        }
        challenges_per_instance.push(instance_challenges);
    }
    challenges_per_instance
}
