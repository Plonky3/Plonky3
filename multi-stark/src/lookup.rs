//! Planning for the multilinear lookup argument.
//!
//! [`LookupPlan`] extracts symbolic lookup declarations from a collection of
//! AIRs, assigns their buses, and lays out the fraction tables consumed by the
//! fractional-GKR protocol.

use alloc::collections::BTreeMap;
use alloc::collections::btree_map::Entry;
use alloc::string::String;
use alloc::vec::Vec;
use core::cmp::Reverse;

use p3_air::symbolic::{BaseEntry, BaseLeaf, SymbolicExpr};
use p3_air::{Air, BaseAir, SymbolicExpression};
use p3_field::{ExtensionField, Field, PackedValue, PrimeField};
use p3_lookup::{InteractionSymbolicBuilder, Kind, Lookups, check_multiplicity_height_bound};
use p3_util::{log2_ceil_usize, log2_strict_usize};

/// Whether a lookup expression reads one of the AIR's fixed periodic columns.
fn uses_periodic_column<F>(expression: &SymbolicExpression<F>) -> bool {
    match expression {
        SymbolicExpr::Leaf(BaseLeaf::Variable(variable)) => variable.entry == BaseEntry::Periodic,
        SymbolicExpr::Leaf(_) => false,
        SymbolicExpr::Add { x, y, .. }
        | SymbolicExpr::Sub { x, y, .. }
        | SymbolicExpr::Mul { x, y, .. } => uses_periodic_column(x) || uses_periodic_column(y),
        SymbolicExpr::Neg { x, .. } => uses_periodic_column(x),
    }
}

/// One logical lookup bus.
///
/// Global buses are shared by name. Each local declaration owns a bus scoped
/// to its AIR instance and declaration index.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum LookupBus {
    Global {
        name: String,
    },
    Local {
        instance_index: usize,
        interaction_index: usize,
    },
}

/// One active AIR's position and symbolic lookup declarations.
pub(crate) struct LookupInstancePlan<F: Field> {
    /// Original position of this AIR in the multi-AIR input arrays.
    pub(crate) air_index: usize,
    /// Base-two logarithm of this AIR instance's trace height.
    pub(crate) num_variables: usize,
    /// First scalar leaf owned by this instance in the materialized lookup arrays.
    pub(crate) base_offset: usize,
    /// Symbolic lookup declarations extracted from this AIR, in protocol order.
    pub(crate) lookups: Lookups<F>,
    /// Bus identifier for each entry of `lookups`, in the same order.
    pub(crate) bus_ids: Vec<usize>,
}

/// Deterministic metadata shared by lookup materialization and AIR coupling.
///
/// This stores no trace rows and allocates no numerator or denominator table.
/// Physical contribution blocks are recovered from an instance's `base_offset`,
/// `num_variables`, and declaration-order contribution index.
pub struct LookupPlan<F: Field> {
    /// Lookup-active AIR instances, sorted by descending trace height.
    pub(crate) instances: Vec<LookupInstancePlan<F>>,
    /// Largest payload tuple width among all planned lookup declarations.
    pub(crate) max_width: usize,
    /// Number of distinct local and named global buses in this plan.
    pub(crate) num_buses: usize,
    /// Variable count of the padded materialized lookup arrays consumed by GKR.
    pub(crate) num_variables: usize,
}

impl<F: Field> LookupPlan<F> {
    /// Extract lookup declarations once, assign buses, and place exact-height blocks.
    ///
    /// # Errors
    ///
    /// Returns an error if the lookup multiplicity height bound reaches the
    /// characteristic of `F`.
    pub fn build<EF, A>(
        airs: &[&A],
        num_variables: &[usize],
    ) -> Result<Option<Self>, p3_lookup::LookupError>
    where
        F: PrimeField,
        EF: ExtensionField<F>,
        A: BaseAir<F> + Air<InteractionSymbolicBuilder<F, EF>>,
    {
        assert_eq!(airs.len(), num_variables.len());

        // Symbolically evaluate each AIR once. The resulting public weights and
        // trace heights determine the multiplicity soundness bound on both sides.
        let lookups = airs
            .iter()
            .map(|&air| Lookups::from_air::<EF, _>(air))
            .collect::<Vec<_>>();
        for lookup in lookups.iter().flat_map(|lookups| lookups.iter()) {
            assert!(
                lookup.flags.is_none(),
                "multi-STARK lookups do not support exclusive interactions"
            );
            assert!(
                !lookup
                    .elements
                    .iter()
                    .flatten()
                    .chain(lookup.multiplicities.iter())
                    .any(uses_periodic_column),
                "multi-STARK lookup expressions do not support periodic columns"
            );
        }
        let trace_heights = num_variables
            .iter()
            .map(|&num_variables| 1usize << num_variables)
            .collect::<Vec<_>>();
        check_multiplicity_height_bound(&lookups, &trace_heights)?;

        // Retain only instances that emit at least one atomic lookup
        // contribution. Empty local declarations are inert.
        let mut active = lookups
            .into_iter()
            .zip(num_variables.iter().copied())
            .enumerate()
            .filter_map(|(air_index, (lookups, num_variables))| {
                lookups
                    .iter()
                    .any(|lookup| !lookup.elements.is_empty())
                    .then_some(LookupInstancePlan {
                        air_index,
                        num_variables,
                        lookups,
                        base_offset: 0,
                        bus_ids: Vec::new(),
                    })
            })
            .collect::<Vec<_>>();

        // Tallest traces come first. Since every height is a power of two, all
        // preceding blocks then leave `base_offset` aligned for every later instance.
        active.sort_by_key(|instance| Reverse(instance.num_variables));

        // No active lookup means this batch has no lookup transcript or proof section.
        if active.is_empty() {
            return Ok(None);
        }

        // Packed denominator blocks require at least one complete SIMD word per trace.
        let packing_variables = log2_strict_usize(F::Packing::WIDTH);
        assert!(
            active
                .iter()
                .all(|instance| instance.num_variables >= packing_variables),
            "lookup-active AIR trace height must be at least the field packing width"
        );

        // Bus IDs domain-separate unrelated lookup arguments. `max_width` determines
        // how many beta powers are needed; `active_height` measures unpadded GKR leaves.
        let mut bus_to_id = BTreeMap::new();
        let mut max_width = 0;
        let mut active_height = 0;

        for instance in &mut active {
            let trace_height = 1usize << instance.num_variables;

            // Every atomic tuple owns one contiguous block of this instance's trace
            // height. All blocks for an instance are adjacent in declaration order.
            instance.base_offset = active_height;
            let contribution_count = instance
                .lookups
                .iter()
                .map(|lookup| lookup.elements.len())
                .sum::<usize>();
            let instance_height = contribution_count * trace_height;
            active_height += instance_height;

            // Record one bus ID per lookup declaration. Local buses are scoped to
            // their declaring AIR, while equally named global buses share an ID.
            instance.bus_ids = instance
                .lookups
                .iter()
                .enumerate()
                .map(|(lookup_index, lookup)| {
                    let bus = match &lookup.kind {
                        Kind::Local => LookupBus::Local {
                            instance_index: instance.air_index,
                            interaction_index: lookup_index,
                        },
                        Kind::Global(name) => LookupBus::Global { name: name.clone() },
                    };

                    // The widest payload fixes the first beta power reserved for
                    // bus separation in `Challenges::new`.
                    let width = lookup.elements.first().map_or(0, Vec::len);
                    let next_id = bus_to_id.len();

                    // The first declaration fixes both the ID and payload schema.
                    // Reusing a named global bus with another width would otherwise
                    // let a shorter tuple alias a longer tuple ending in zeroes.
                    let bus_id = match bus_to_id.entry(bus) {
                        Entry::Vacant(entry) => {
                            entry.insert((next_id, width));
                            next_id
                        }
                        Entry::Occupied(entry) => {
                            let &(bus_id, established_width) = entry.get();
                            match entry.key() {
                                LookupBus::Global { name } => assert_eq!(
                                    width, established_width,
                                    "named global lookup bus `{name}` uses payload width {width}, \
                                     but its established width is {established_width}"
                                ),
                                _ => unreachable!(),
                            }
                            bus_id
                        }
                    };
                    max_width = max_width.max(width);
                    bus_id
                })
                .collect();
        }

        Ok(Some(Self {
            instances: active,
            max_width,
            num_buses: bus_to_id.len(),
            num_variables: log2_ceil_usize(active_height),
        }))
    }
}
