//! Functions and structures that return or store common data shared by the prover and the verifier.
//!
//! For example, we get the lookups before calling either `prove_multi` or `verify_multi`, and we can pass the lookup data to both the prover and verifier.

use alloc::vec::Vec;

use hashbrown::HashMap;
use p3_challenger::FieldChallenger;
use p3_field::BasedVectorSpace;
use p3_lookup::lookup_traits::{AirLookupHandler, Kind, Lookup, LookupGadget};
use p3_uni_stark::{SymbolicAirBuilder, SymbolicExpression, Val};

use crate::Challenge;
use crate::config::SGC;

/// Struct storing data common to both the prover and verifier.
/// TODO: Add preprocessed commitments.
#[derive(Debug)]
pub struct CommonData<SC: SGC> {
    /// The lookups used by each STARK instance.
    /// There is one `Vec<Lookup<Val<SC>>>` per STARK instance.
    /// They are stored in the same order as the STARK instance inputs provided to `new`.
    pub lookups: Vec<Vec<Lookup<Val<SC>>>>,
}

impl<SC: SGC> CommonData<SC> {
    pub const fn new(lookups: Vec<Vec<Lookup<Val<SC>>>>) -> Self {
        Self { lookups }
    }
}

impl<SC> CommonData<SC>
where
    SC: SGC,
    SymbolicExpression<SC::Challenge>: From<SymbolicExpression<Val<SC>>>,
    Challenge<SC>: BasedVectorSpace<Val<SC>>,
{
    /// Get the total number of lookup contexts across all instances.
    pub fn from_airs<A>(airs: &mut [A]) -> Self
    where
        A: AirLookupHandler<SymbolicAirBuilder<Val<SC>, SC::Challenge>>,
    {
        let lookups = airs.iter_mut().map(|air| air.get_lookups()).collect();
        Self { lookups }
    }
}

pub(crate) fn get_perm_challenges<SC: SGC, LG: LookupGadget>(
    challenger: &mut SC::Challenger,
    all_lookups: &[Vec<Lookup<Val<SC>>>],
    lookup_gadget: &LG,
) -> Vec<Vec<SC::Challenge>> {
    let num_challenges_per_lookup = lookup_gadget.num_challenges();
    let mut global_perm_challenges = HashMap::new();

    all_lookups
        .iter()
        .map(|contexts| {
            // Pre-allocate for the instance's challenges.
            let num_challenges = contexts.len() * num_challenges_per_lookup;
            let mut instance_challenges = Vec::with_capacity(num_challenges);

            // We should avoid a bunch of allocations here
            for context in contexts {
                match &context.kind {
                    Kind::Global(name) => {
                        // Get or create the global challenges.
                        let cs: &mut Vec<SC::Challenge> =
                            global_perm_challenges.entry(name).or_insert_with(|| {
                                (0..num_challenges_per_lookup)
                                    .map(|_| challenger.sample_algebra_element())
                                    .collect()
                            });
                        // Extend from the slice directly without vector cloning
                        instance_challenges.extend(cs.iter().copied());
                    }
                    Kind::Local => {
                        instance_challenges.extend(
                            (0..num_challenges_per_lookup)
                                .map(|_| challenger.sample_algebra_element::<SC::Challenge>()),
                        );
                    }
                }
            }
            instance_challenges
        })
        .collect()
}
