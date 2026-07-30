use alloc::collections::BTreeMap;
use alloc::collections::btree_map::Entry;
use alloc::string::String;
use alloc::vec::Vec;
use core::cmp::Reverse;

use p3_air::symbolic::{BaseEntry, BaseLeaf, SymbolicExpr};
use p3_air::{Air, BaseAir, SymbolicExpression};
use p3_challenger::FieldChallenger;
use p3_field::{ExtensionField, Field, PackedValue, PrimeField};
use p3_lookup::{
    Challenges, InteractionSymbolicBuilder, Kind, Lookups, check_multiplicity_height_bound,
};
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::{Poly, PolyMaybePacked};
use p3_multilinear_util::split_eq::SplitEq;
use p3_sumcheck::layout::Table;
use p3_util::{log2_ceil_usize, log2_strict_usize};
use thiserror::Error;

use crate::fractional_gkr::{
    Fraction, FractionGkrError, FractionGkrOutput, FractionGkrProof, prove_fractional_gkr,
    verify_fractional_gkr,
};

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
pub(crate) struct LookupPlan<F: Field> {
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
    pub(crate) fn build<EF, A>(airs: &[&A], num_variables: &[usize]) -> Option<Self>
    where
        EF: ExtensionField<F>,
        A: BaseAir<F> + Air<InteractionSymbolicBuilder<F, EF>>,
    {
        assert_eq!(airs.len(), num_variables.len());

        // Symbolically evaluate each AIR once and retain only instances that emit
        // at least one atomic lookup contribution. Empty local declarations are inert.
        let mut active = airs
            .iter()
            .zip(num_variables)
            .enumerate()
            .filter_map(|(air_index, (&air, &num_variables))| {
                let lookups = Lookups::from_air::<EF, _>(air);
                for lookup in lookups.iter() {
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
            return None;
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
                                LookupBus::Local { .. } => {
                                    debug_assert_eq!(width, established_width);
                                }
                            }
                            bus_id
                        }
                    };
                    max_width = max_width.max(width);
                    bus_id
                })
                .collect();
        }

        Some(Self {
            instances: active,
            max_width,
            num_buses: bus_to_id.len(),
            num_variables: log2_ceil_usize(active_height),
        })
    }
}

/// Initial claim and static weights consumed by the forthcoming AIR interaction sumcheck.
///
/// For an atomic contribution `b` with multiplicity `m_b`, denominator `D_b`,
/// and trace-height suffix point `q_b`, the next reduction proves
///
/// `claimed_sum = sum_b weight_b * sum_x eq(q_b, x) * (m_b(x) + theta * (D_b(x) - 1))`.
///
/// Subtracting one from each active denominator accounts for the materializer's
/// neutral `0 / 1` padding without introducing a synthetic padding interaction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct AirLinkClaim<EF> {
    /// Fractional-GKR reduction point fixed as the zerocheck point suffix.
    pub(crate) point: Point<EF>,
    /// `N(q) + theta * (D(q) - 1)` at the fractional GKR output point `q`.
    pub(crate) claimed_sum: EF,
    /// Coefficients `[theta, theta * beta, theta * beta^2, ...]` used when the
    /// AIR is re-evaluated during zerocheck.
    ///
    /// Coefficient `i` multiplies payload field `i`; iterator zipping ignores the
    /// unused tail for narrower payloads. The interaction folder thereby forms
    /// `theta * fingerprint`, then adds the corresponding bus offset and
    /// multiplicity. Including the materialized block weight, this reconstructs
    /// `weight * (m + theta * (D - 1))`, whose sum is the AIR-side form of the
    /// fractional GKR claim `N(point) + theta * (D(point) - 1)`.
    pub(crate) theta_beta_powers: Vec<EF>,
    /// Lookup weights keyed by original AIR index.
    ///
    /// Protocol ordering remains defined by [`LookupPlan::instances`].
    pub(crate) links_by_air: BTreeMap<usize, AirLinkInstance<EF>>,
}

/// Lookup-link metadata for one AIR instance.
///
/// Sparse AIR-link claims contain only lookup-active instances.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct AirLinkInstance<EF> {
    /// Length of the local-lookup prefix in `lookups`; global lookups follow it.
    pub(crate) num_local_lookups: usize,
    /// Link metadata for this AIR's lookup declarations, locals before globals.
    pub(crate) lookups: Vec<AirLinkLookup<EF>>,
}

impl<EF: Copy> AirLinkInstance<EF> {
    /// Whether this AIR contributes no lookup expressions to zerocheck.
    pub(crate) const fn is_empty(&self) -> bool {
        self.lookups.is_empty()
    }

    /// Local lookup declarations in AIR emission order.
    pub(crate) fn local_lookups(&self) -> &[AirLinkLookup<EF>] {
        &self.lookups[..self.num_local_lookups]
    }

    /// Global lookup declarations in AIR emission order.
    pub(crate) fn global_lookups(&self) -> &[AirLinkLookup<EF>] {
        &self.lookups[self.num_local_lookups..]
    }

    pub(crate) fn map<T>(&self, mut f: impl FnMut(EF) -> T) -> AirLinkInstance<T> {
        AirLinkInstance {
            num_local_lookups: self.num_local_lookups,
            lookups: self
                .lookups
                .iter()
                .map(|lookup| AirLinkLookup {
                    theta_bus_offset: f(lookup.theta_bus_offset),
                    block_weights: lookup.block_weights.iter().copied().map(&mut f).collect(),
                })
                .collect(),
        }
    }
}

/// AIR-side coefficients linking one symbolic lookup declaration to its GKR blocks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct AirLinkLookup<EF> {
    /// `theta * (bus_prefix - 1)`, shared by every denominator in this lookup.
    pub(crate) theta_bus_offset: EF,
    /// Equality weight selecting each atomic contribution's physical block.
    pub(crate) block_weights: Vec<EF>,
}

impl<EF: Field> AirLinkClaim<EF> {
    fn new<F: Field>(
        plan: &LookupPlan<F>,
        alpha: EF,
        beta: EF,
        output: FractionGkrOutput<EF>,
        theta: EF,
    ) -> Self
    where
        EF: ExtensionField<F>,
    {
        // Reconstruct the scalar challenges used by the AIR-side lookup expressions.
        let challenges = Challenges::new(alpha, beta, plan.max_width, plan.num_buses);
        let theta_beta_powers = beta
            .powers()
            .take(plan.max_width)
            .map(|beta_i| theta * beta_i)
            .collect();

        let num_variables = plan.num_variables;
        // Attach each atomic contribution to the GKR block that materialized it.
        let links_by_air = plan
            .instances
            .iter()
            .map(|planned| {
                // The leading variables select this contribution's physical block;
                // the remaining variables address a row of the instance trace.
                let prefix_len = num_variables - planned.num_variables;
                let point_prefix = &output.point.as_slice()[..prefix_len];

                // `base_offset` is trace-height aligned, so shifting converts it
                // directly into the first block index owned by this instance.
                let mut block_index = planned.base_offset >> planned.num_variables;
                let lookups = planned
                    .lookups
                    .iter()
                    .zip(&planned.bus_ids)
                    .map(|(lookup, &bus_id)| {
                        // Every atomic tuple occupies the next exact-height block.
                        // Its equality weight selects that block at the GKR point.
                        let block_weights = lookup
                            .elements
                            .iter()
                            .map(|_| {
                                let vertex = Point::<F>::hypercube(block_index, prefix_len);
                                block_index += 1;
                                Point::eval_eq(vertex.as_slice(), point_prefix)
                            })
                            .collect();
                        AirLinkLookup {
                            // Preserve only the lookup metadata needed by the AIR folder.
                            theta_bus_offset: theta * (challenges.bus_prefix[bus_id] - EF::ONE),
                            block_weights,
                        }
                    })
                    .collect();

                (
                    planned.air_index,
                    AirLinkInstance {
                        // `Lookups::from_air` stores locals before globals; the folder
                        // uses this boundary to address both kinds in declaration order.
                        num_local_lookups: planned
                            .lookups
                            .iter()
                            .take_while(|lookup| lookup.kind == Kind::Local)
                            .count(),
                        lookups,
                    },
                )
            })
            .collect();

        // Combine the two GKR openings into the initial AIR-link claim.
        Self {
            point: output.point,
            claimed_sum: output.numerator + theta * (output.denominator - EF::ONE),
            theta_beta_powers,
            links_by_air,
        }
    }

    /// Recover one private lookup-link claim per active AIR from the materialized GKR leaves.
    ///
    /// The leading coordinates of `point` select contribution blocks, while the
    /// suffix evaluates rows inside a block. Instances are grouped by height so
    /// every block in one stage reuses the same suffix equality table.
    fn evaluate_instance_claims<F: Field>(
        &self,
        plan: &LookupPlan<F>,
        fraction: &Fraction<Poly<F>, PolyMaybePacked<F, EF>>,
        theta: EF,
    ) -> BTreeMap<usize, EF>
    where
        EF: ExtensionField<F>,
    {
        let mut claims_by_air = BTreeMap::new();

        for stage in plan
            .instances
            .chunk_by(|left, right| left.num_variables == right.num_variables)
        {
            let stage_num_variables = stage[0].num_variables;
            let suffix = self.point.get_subpoint_over_range(
                plan.num_variables - stage_num_variables..plan.num_variables,
            );
            let eq_suffix = SplitEq::<F, EF>::new_packed(&suffix, EF::ONE);

            for planned in stage {
                let linked = &self.links_by_air[&planned.air_index];
                let block_len = 1 << planned.num_variables;
                let packed_block_len = block_len / F::Packing::WIDTH;
                let mut start = planned.base_offset;
                let mut claim = EF::ZERO;

                for &weight in linked
                    .lookups
                    .iter()
                    .flat_map(|lookup| &lookup.block_weights)
                {
                    let end = start + block_len;
                    let packed_start = start / F::Packing::WIDTH;
                    let numerator =
                        eq_suffix.eval_base(Poly::new(&fraction.n.as_slice()[start..end]));
                    let denominator = match &fraction.d {
                        PolyMaybePacked::Scalar(denominators) => {
                            eq_suffix.eval_ext(Poly::new(&denominators.as_slice()[start..end]))
                        }
                        PolyMaybePacked::Packed(denominators) => eq_suffix.eval_packed(Poly::new(
                            &denominators.as_slice()[packed_start..packed_start + packed_block_len],
                        )),
                    };

                    claim += weight * (numerator + theta * (denominator - EF::ONE));
                    start = end;
                }

                assert!(claims_by_air.insert(planned.air_index, claim).is_none());
            }
        }

        claims_by_air
    }
}

/// Prover-private state retained for an active lookup reduction.
pub(crate) struct ActiveLookupRuntime<EF> {
    /// Private lookup-link claims keyed by original caller AIR index.
    pub(crate) claims_by_air: BTreeMap<usize, EF>,
    /// Verifier-reconstructible claim and metadata shared with zerocheck.
    pub(crate) air_link: AirLinkClaim<EF>,
}

/// Prover-side lookup state passed into the coupled zerocheck.
pub(crate) enum LookupRuntime<EF> {
    /// No AIR declares lookup interactions.
    Inactive,
    /// Lookup materialization and fractional GKR produced coupling state.
    Active(ActiveLookupRuntime<EF>),
}

impl<EF: Field> LookupRuntime<EF> {
    /// Validate the active AIR-link alignment consumed by zerocheck.
    ///
    /// # Panics
    ///
    /// Panics if active AIR-link claims and metadata are inconsistent with the
    /// caller AIR count.
    pub(crate) fn validate(&self, num_airs: usize) {
        if let Self::Active(runtime) = self {
            runtime.validate(num_airs);
        }
    }
}

impl<EF: Field> ActiveLookupRuntime<EF> {
    /// Validate the active AIR-link alignment consumed by zerocheck.
    ///
    /// # Panics
    ///
    /// Panics if active AIR-link claims and metadata are inconsistent with the
    /// caller AIR count.
    pub(crate) fn validate(&self, num_airs: usize) {
        let air_link = &self.air_link;

        assert!(air_link.point.num_variables() > 0);
        assert!(!air_link.links_by_air.is_empty());
        assert!(self.claims_by_air.keys().eq(air_link.links_by_air.keys()));
        assert!(air_link.links_by_air.iter().all(|(&air_index, link)| {
            air_index < num_airs && !link.is_empty() && link.num_local_lookups <= link.lookups.len()
        }));
    }
}

#[derive(Debug, Error)]
pub enum LookupError {
    #[error("lookup proof expected but absent")]
    MissingProof,
    #[error("lookup proof present but no AIR declares interactions")]
    UnexpectedProof,
    #[error(transparent)]
    MultiplicityHeightBound(#[from] p3_lookup::LookupError),
    #[error("fractional GKR: {0}")]
    FractionGkr(#[from] FractionGkrError),
    #[error("lookup fractional sum is nonzero")]
    NonZeroFractionalSum,
}

/// Materialize lookup leaves and prove their fractional reduction.
pub(crate) fn prove_lookup<F, EF, A, Challenger>(
    airs: &[&A],
    main: &[&Table<F>],
    preprocessed: &[Option<&Table<F>>],
    public_values: &[&[F]],
    challenger: &mut Challenger,
) -> (Option<FractionGkrProof<EF>>, LookupRuntime<EF>)
where
    F: Field,
    EF: ExtensionField<F>,
    A: BaseAir<F> + Air<InteractionSymbolicBuilder<F, EF>>,
    Challenger: FieldChallenger<F>,
{
    // The four input slices use the original AIR index as their shared key.
    assert_eq!(airs.len(), main.len());
    assert_eq!(airs.len(), preprocessed.len());
    assert_eq!(airs.len(), public_values.len());

    // Planning phase: extract lookup declarations, order lookup-active AIRs by
    // height, and assign every atomic contribution a padded GKR block. A
    // lookup-free batch skips the transcript phase and returns explicit inactive
    // prover state.
    let num_variables = main
        .iter()
        .map(|table| table.num_variables())
        .collect::<Vec<_>>();
    let Some(plan) = LookupPlan::build::<EF, A>(airs, &num_variables) else {
        return (None, LookupRuntime::Inactive);
    };

    // Fingerprint phase: beta combines payload coordinates, while alpha and
    // the reserved beta power give each lookup bus a distinct prefix.
    let alpha: EF = challenger.sample_algebra_element();
    let beta: EF = challenger.sample_algebra_element();

    // Fractional-GKR phase: materialize every `multiplicity / denominator`
    // contribution, prove that their padded sum is zero, and reduce the two
    // materialized polynomials to openings at one GKR output point.
    let fraction = plan.materialize_fraction(main, preprocessed, public_values, alpha, beta);
    let (fractional_gkr, output) = prove_fractional_gkr(&fraction, challenger);
    assert_eq!(fractional_gkr.root_numerator, EF::ZERO);

    // Sample theta only after GKR fixes its output point and openings. It folds
    // the numerator opening and the shifted denominator opening into the single
    // AIR-link claim `N(point) + theta * (D(point) - 1)` that zerocheck carries.
    let theta: EF = challenger.sample_algebra_element();
    let air_link = AirLinkClaim::new(&plan, alpha, beta, output, theta);

    // Recover the private per-AIR decomposition needed when native-height
    // zerocheck stages activate. These values affect only prover bookkeeping;
    // the global sumcheck starts from `air_link.claimed_sum`.
    let claims_by_air = air_link.evaluate_instance_claims(&plan, &fraction, theta);
    debug_assert_eq!(
        claims_by_air.values().copied().sum::<EF>(),
        air_link.claimed_sum
    );

    // Return serialized proof material separately from the private state that
    // drives the coupled zerocheck prover.
    let runtime = ActiveLookupRuntime {
        claims_by_air,
        air_link,
    };

    (Some(fractional_gkr), LookupRuntime::Active(runtime))
}

/// Verify the fractional GKR phase and recover the claims needed by AIR coupling.
pub(crate) fn verify_lookup<F, EF, A, Challenger>(
    airs: &[&A],
    num_variables: &[usize],
    proof: Option<&FractionGkrProof<EF>>,
    challenger: &mut Challenger,
) -> Result<Option<AirLinkClaim<EF>>, LookupError>
where
    F: PrimeField,
    EF: ExtensionField<F>,
    A: BaseAir<F> + Air<InteractionSymbolicBuilder<F, EF>>,
    Challenger: FieldChallenger<F>,
{
    assert_eq!(airs.len(), num_variables.len());

    // Rebuild the deterministic lookup layout from the verifier's AIRs and
    // trace heights; no prover-supplied layout metadata is trusted.
    let plan = LookupPlan::build::<EF, A>(airs, num_variables);

    // A lookup proof exists exactly when at least one AIR emits an atomic
    // lookup contribution.
    let (plan, proof) = match (plan, proof) {
        (None, None) => return Ok(None),
        (None, Some(_)) => return Err(LookupError::UnexpectedProof),
        (Some(_), None) => return Err(LookupError::MissingProof),
        (Some(plan), Some(proof)) => (plan, proof),
    };

    // Enforce the public multiplicity bound only on the verifier side. Heights
    // are aligned with the active lookup sets retained by the plan.
    let trace_heights = plan
        .instances
        .iter()
        .map(|instance| 1usize << instance.num_variables)
        .collect::<Vec<_>>();
    check_multiplicity_height_bound(
        plan.instances.iter().map(|instance| &instance.lookups),
        &trace_heights,
    )?;

    // Replay the lookup challenges from the statement-bound transcript.
    let alpha: EF = challenger.sample_algebra_element();
    let beta: EF = challenger.sample_algebra_element();

    // Verify the fractional reduction over the padded lookup layout and recover
    // its numerator and denominator openings at the GKR output point.
    let output = verify_fractional_gkr::<F, EF, _>(proof, plan.num_variables, challenger)?;
    if proof.root_numerator != EF::ZERO {
        return Err(LookupError::NonZeroFractionalSum);
    }

    // As on the prover side, sample theta only after GKR fixes its output. It
    // folds the two openings into one AIR-link claim. Zerocheck derives the
    // native interaction-group claims when their trace-height stage activates.
    let theta: EF = challenger.sample_algebra_element();
    Ok(Some(AirLinkClaim::new(&plan, alpha, beta, output, theta)))
}

#[cfg(test)]
mod tests {
    extern crate std;

    use alloc::borrow::Cow;
    use alloc::vec;

    use p3_air::{AirBuilder, WindowAccess};
    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_lookup::{Count, InteractionBuilder};
    use p3_multilinear_util::poly::{Poly, PolyMaybePacked};
    use p3_util::log2_strict_usize;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::zerocheck::AirZerocheck;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type Challenger = DuplexChallenger<F, Perm, 16, 8>;

    struct BalancedLookupAir;

    impl BaseAir<F> for BalancedLookupAir {
        fn width(&self) -> usize {
            1
        }
    }

    struct PaddedLookupAir;

    impl BaseAir<F> for PaddedLookupAir {
        fn width(&self) -> usize {
            1
        }
    }

    struct ExcessiveMultiplicityAir;

    impl BaseAir<F> for ExcessiveMultiplicityAir {
        fn width(&self) -> usize {
            1
        }
    }

    impl<AB> Air<AB> for ExcessiveMultiplicityAir
    where
        AB: AirBuilder<F = F> + InteractionBuilder,
    {
        fn eval(&self, builder: &mut AB) {
            let value = builder.main().current_slice()[0];
            builder.push_local_interaction([(
                vec![value.into()],
                Count::bounded(AB::Expr::ONE, u32::MAX),
            )]);
        }
    }

    impl<AB> Air<AB> for PaddedLookupAir
    where
        AB: AirBuilder<F = F> + InteractionBuilder,
    {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let value = main.current_slice()[0];
            builder.push_local_interaction([
                (vec![value.into()], Count::bounded(AB::Expr::ONE, 1)),
                (vec![value.into()], Count::provided(-AB::Expr::ONE)),
                (vec![value.into()], Count::bounded(AB::Expr::ONE, 1)),
            ]);
        }
    }

    impl<AB> Air<AB> for BalancedLookupAir
    where
        AB: AirBuilder<F = F> + InteractionBuilder,
    {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let value = main.current_slice()[0];
            let next = main.next_slice()[0];
            let provided = value * value - next;
            let requested = value * value - next;
            builder.push_local_interaction([
                (vec![requested], Count::bounded(AB::Expr::ONE, 1)),
                (vec![provided], Count::provided(-AB::Expr::ONE)),
            ]);
        }
    }

    struct DegreeLookupAir {
        degree: usize,
    }

    struct ExclusiveLookupAir;

    impl BaseAir<F> for ExclusiveLookupAir {
        fn width(&self) -> usize {
            2
        }
    }

    impl<AB> Air<AB> for ExclusiveLookupAir
    where
        AB: AirBuilder<F = F> + InteractionBuilder,
    {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let flag = main.current_slice()[0];
            let value = main.current_slice()[1];
            builder.push_exclusive_interaction(
                "exclusive",
                [(
                    flag.into(),
                    Count::bounded(AB::Expr::ONE, 1),
                    vec![value.into()],
                )],
            );
        }
    }

    struct PeriodicLookupAir;

    impl BaseAir<F> for PeriodicLookupAir {
        fn width(&self) -> usize {
            1
        }

        fn num_periodic_columns(&self) -> usize {
            1
        }

        fn periodic_columns(&self) -> Cow<'_, [Vec<F>]> {
            Cow::Owned(vec![vec![F::ZERO, F::ONE]])
        }
    }

    impl<AB> Air<AB> for PeriodicLookupAir
    where
        AB: AirBuilder<F = F> + InteractionBuilder,
    {
        fn eval(&self, builder: &mut AB) {
            let periodic = builder.periodic_values()[0];
            builder.push_interaction("periodic", [periodic], Count::bounded(AB::Expr::ONE, 1));
        }
    }

    impl BaseAir<F> for DegreeLookupAir {
        fn width(&self) -> usize {
            5
        }
    }

    impl<AB> Air<AB> for DegreeLookupAir
    where
        AB: AirBuilder<F = F> + InteractionBuilder,
    {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let first: AB::Expr = main.current_slice()[0].into();
            let value = main
                .current_slice()
                .iter()
                .take(self.degree)
                .fold(AB::Expr::ONE, |product, &value| product * value);
            // Keep a genuine degree-one ordinary constraint beside the
            // higher-degree lookup expression while remaining identically satisfied.
            builder.assert_zero(first.clone() - first);
            builder.push_local_interaction([
                (vec![value.clone()], Count::bounded(AB::Expr::ONE, 1)),
                (vec![value], Count::provided(-AB::Expr::ONE)),
            ]);
        }
    }

    fn challenger() -> Challenger {
        let mut rng = SmallRng::seed_from_u64(0x100C_A11E);
        Challenger::new(Perm::new_from_rng_128(&mut rng))
    }

    fn test_lookup_num_variables() -> usize {
        log2_strict_usize(<F as Field>::Packing::WIDTH).max(4)
    }

    #[test]
    fn lookup_plan_rejects_trace_shorter_than_packing_width() {
        let packing_variables = log2_strict_usize(<F as Field>::Packing::WIDTH);
        if packing_variables == 0 {
            // Scalar packing has width one, so no nonempty trace can be shorter.
            return;
        }

        let air = BalancedLookupAir;
        assert!(
            std::panic::catch_unwind(|| {
                LookupPlan::build::<EF, _>(&[&air], &[packing_variables - 1])
            })
            .is_err()
        );
    }

    #[test]
    #[should_panic(expected = "multi-STARK lookups do not support exclusive interactions")]
    fn lookup_plan_rejects_exclusive_interactions() {
        let air = ExclusiveLookupAir;
        let _ = LookupPlan::build::<EF, _>(&[&air], &[test_lookup_num_variables()]);
    }

    #[test]
    #[should_panic(expected = "multi-STARK lookup expressions do not support periodic columns")]
    fn lookup_plan_rejects_periodic_expressions() {
        let air = PeriodicLookupAir;
        let _ = LookupPlan::build::<EF, _>(&[&air], &[test_lookup_num_variables()]);
    }

    #[test]
    fn lookup_verifier_rejects_multiplicity_bound_at_field_characteristic() {
        let air = ExcessiveMultiplicityAir;
        let mut challenger = challenger();
        assert!(matches!(
            verify_lookup::<F, EF, _, _>(
                &[&air],
                &[test_lookup_num_variables()],
                Some(&FractionGkrProof {
                    root_numerator: EF::ZERO,
                    root_denominator: EF::ONE,
                    layers: Vec::new(),
                }),
                &mut challenger,
            ),
            Err(LookupError::MultiplicityHeightBound(
                p3_lookup::LookupError::MultiplicityHeightBoundExceeded { .. }
            ))
        ));
    }

    #[test]
    fn materialize_and_fractional_gkr_round_trip() {
        let air = BalancedLookupAir;
        let mut rng = SmallRng::seed_from_u64(0x100C_600D);
        let main = Table::<F>::rand(&mut rng, 1, 6);
        let public_values: &[F] = &[];

        let mut prover_challenger = challenger();
        let (lookup_proof, prover) = prove_lookup::<F, EF, _, _>(
            &[&air],
            &[&main],
            &[None],
            &[public_values],
            &mut prover_challenger,
        );
        let lookup_proof = lookup_proof.unwrap();

        assert_eq!(lookup_proof.root_numerator, EF::ZERO);

        let mut verifier_challenger = challenger();
        let verifier = verify_lookup::<F, EF, _, _>(
            &[&air],
            &[main.num_variables()],
            Some(&lookup_proof),
            &mut verifier_challenger,
        )
        .unwrap()
        .unwrap();

        let LookupRuntime::Active(prover) = prover else {
            panic!("lookup-active AIR must produce active runtime data");
        };
        assert_eq!(verifier, prover.air_link);
        assert_eq!(
            prover.claims_by_air.values().copied().sum::<EF>(),
            prover.air_link.claimed_sum
        );

        let prover_final: EF = prover_challenger.sample_algebra_element();
        let verifier_final: EF = verifier_challenger.sample_algebra_element();
        assert_eq!(prover_final, verifier_final);
    }

    #[test]
    fn air_link_claim_decomposes_gkr_openings_into_lookup_blocks() {
        let air = PaddedLookupAir;
        let mut rng = SmallRng::seed_from_u64(0xA17_11A5);
        let main = Table::<F>::rand(&mut rng, 1, test_lookup_num_variables());
        let plan = LookupPlan::build::<EF, _>(&[&air], &[main.num_variables()]).unwrap();
        assert_eq!(1 << plan.num_variables, 4 << main.num_variables());

        let alpha: EF = rng.random();
        let beta: EF = rng.random();
        let theta: EF = rng.random();
        let point = Point::<EF>::rand(&mut rng, plan.num_variables);
        let materialization = plan.materialize_fraction(&[&main], &[None], &[&[]], alpha, beta);
        let numer = materialization.n;
        let denom = materialization.d;
        let output = FractionGkrOutput {
            numerator: numer.eval_base::<EF>(&point),
            denominator: denom.eval(&point),
            point,
        };
        let air_link = AirLinkClaim::new(&plan, alpha, beta, output.clone(), theta);

        let planned = &plan.instances[0];
        let linked = &air_link.links_by_air[&planned.air_index];
        let suffix = Point::new(
            output.point.as_slice()[plan.num_variables - planned.num_variables..].to_vec(),
        );
        let block_len = 1 << planned.num_variables;
        let mut expected = EF::ZERO;
        let mut contribution_index = 0;
        for lookup in &linked.lookups {
            for &weight in &lookup.block_weights {
                let start = planned.base_offset + contribution_index * block_len;
                let packing_width = <F as Field>::Packing::WIDTH;
                let packed_start = start / packing_width;
                let packed_block_len = block_len / packing_width;
                let block_numer =
                    Poly::new(&numer.as_slice()[start..start + block_len]).eval_base::<EF>(&suffix);
                let block_denom = match &denom {
                    PolyMaybePacked::Scalar(denominators) => {
                        Poly::new(&denominators.as_slice()[start..start + block_len])
                            .eval_ext::<F>(&suffix)
                    }
                    PolyMaybePacked::Packed(denominators) => Poly::new(
                        &denominators.as_slice()[packed_start..packed_start + packed_block_len],
                    )
                    .eval_packed::<F, EF>(&suffix),
                };
                expected += weight * (block_numer + theta * (block_denom - EF::ONE));
                contribution_index += 1;
            }
        }

        assert_eq!(air_link.claimed_sum, expected);
    }

    #[test]
    fn fractional_gkr_is_coupled_to_air_interactions() {
        let air = BalancedLookupAir;
        let mut rng = SmallRng::seed_from_u64(0xA17_C0DE);
        let main = Table::<F>::rand(&mut rng, 1, 6);
        let public_values: &[F] = &[];

        let mut prover_challenger = challenger();
        let (lookup_proof, lookup) = prove_lookup::<F, EF, _, _>(
            &[&air],
            &[&main],
            &[None],
            &[public_values],
            &mut prover_challenger,
        );
        let lookup_proof = lookup_proof.unwrap();
        let airs = [&air];
        let sumcheck = AirZerocheck::new(&airs, 0);
        let (proof, prover_point) = sumcheck.prove_with_lookup(
            &[None],
            &[&main],
            &[public_values],
            lookup,
            &mut prover_challenger,
        );

        let mut verifier_challenger = challenger();
        let verifier_lookup = verify_lookup::<F, EF, _, _>(
            &[&air],
            &[main.num_variables()],
            Some(&lookup_proof),
            &mut verifier_challenger,
        )
        .unwrap()
        .unwrap();
        let verifier_point = sumcheck
            .verify_with_lookup(
                &proof,
                &[main.num_variables()],
                &[public_values],
                Some(&verifier_lookup),
                &mut verifier_challenger,
            )
            .unwrap();

        assert_eq!(verifier_point, prover_point);
        let prover_final: EF = prover_challenger.sample_algebra_element();
        let verifier_final: EF = verifier_challenger.sample_algebra_element();
        assert_eq!(prover_final, verifier_final);
    }

    #[test]
    fn coupled_stage_preserves_mixed_air_degrees() {
        let airs = [
            DegreeLookupAir { degree: 2 },
            DegreeLookupAir { degree: 3 },
            DegreeLookupAir { degree: 3 },
            DegreeLookupAir { degree: 5 },
        ];
        let air_refs = airs.iter().collect::<Vec<_>>();
        let mut rng = SmallRng::seed_from_u64(0xA17_DE6EE);
        let tables = (0..airs.len())
            .map(|_| Table::<F>::rand(&mut rng, 5, test_lookup_num_variables()))
            .collect::<Vec<_>>();
        let table_refs = tables.iter().collect::<Vec<_>>();
        let preprocessed = alloc::vec![None; airs.len()];
        let public_values = alloc::vec![&[][..]; airs.len()];

        let mut prover_challenger = challenger();
        let (lookup_proof, lookup) = prove_lookup::<F, EF, _, _>(
            &air_refs,
            &table_refs,
            &preprocessed,
            &public_values,
            &mut prover_challenger,
        );
        let lookup_proof = lookup_proof.unwrap();
        let sumcheck = AirZerocheck::new(&air_refs, 0);
        let (proof, prover_point) = sumcheck.prove_with_lookup(
            &preprocessed,
            &table_refs,
            &public_values,
            lookup,
            &mut prover_challenger,
        );

        let log_heights = tables.iter().map(Table::num_variables).collect::<Vec<_>>();
        let mut verifier_challenger = challenger();
        let verifier_lookup = verify_lookup::<F, EF, _, _>(
            &air_refs,
            &log_heights,
            Some(&lookup_proof),
            &mut verifier_challenger,
        )
        .unwrap()
        .unwrap();
        let verifier_point = sumcheck
            .verify_with_lookup(
                &proof,
                &log_heights,
                &public_values,
                Some(&verifier_lookup),
                &mut verifier_challenger,
            )
            .unwrap();

        assert_eq!(verifier_point, prover_point);
        assert_eq!(
            prover_challenger.sample_algebra_element::<EF>(),
            verifier_challenger.sample_algebra_element::<EF>()
        );
    }

    #[test]
    fn mixed_height_lookups_activate_at_their_air_stages() {
        let air = BalancedLookupAir;
        let mut rng = SmallRng::seed_from_u64(0xA17_57A6E);
        let tall = Table::<F>::rand(&mut rng, 1, 6);
        let short = Table::<F>::rand(&mut rng, 1, 4);
        let public_values: &[F] = &[];
        let airs = [&air, &air];
        let publics = [public_values, public_values];

        let mut prover_challenger = challenger();
        let (lookup_proof, lookup) = prove_lookup::<F, EF, _, _>(
            &airs,
            &[&tall, &short],
            &[None, None],
            &publics,
            &mut prover_challenger,
        );
        let lookup_proof = lookup_proof.unwrap();
        let sumcheck = AirZerocheck::new(&airs, 0);
        let (proof, prover_point) = sumcheck.prove_with_lookup(
            &[None, None],
            &[&tall, &short],
            &publics,
            lookup,
            &mut prover_challenger,
        );

        let mut verifier_challenger = challenger();
        let verifier_lookup = verify_lookup::<F, EF, _, _>(
            &airs,
            &[tall.num_variables(), short.num_variables()],
            Some(&lookup_proof),
            &mut verifier_challenger,
        )
        .unwrap()
        .unwrap();
        let verifier_point = sumcheck
            .verify_with_lookup(
                &proof,
                &[tall.num_variables(), short.num_variables()],
                &publics,
                Some(&verifier_lookup),
                &mut verifier_challenger,
            )
            .unwrap();

        assert_eq!(verifier_point, prover_point);
        assert!(prover_point.num_variables() > tall.num_variables());
        let prover_final: EF = prover_challenger.sample_algebra_element();
        let verifier_final: EF = verifier_challenger.sample_algebra_element();
        assert_eq!(prover_final, verifier_final);
    }
}
