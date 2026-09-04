//! Planning and reduction of the multilinear lookup argument.
//!
//! Every lookup in the batch becomes one fraction `multiplicity / denominator` per trace row.
//! Fractional GKR proves that all of those fractions sum to zero, which is the LogUp identity.
//!
//! ```text
//!     sum over every declared tuple b, every row x of
//!         m_b(x) / (bus_prefix_b - sum_k beta^k * payload_bk(x))   =   0
//! ```
//!
//! The reduction ends at one random point, where the two sides are re-linked.
//! GKR opens the fractions there.
//! The zerocheck sumcheck then rebuilds the same value from the AIR's own columns.

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
use p3_util::log2_ceil_usize;
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
/// Two declarations balance against each other only when they land on the same bus.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum LookupBus {
    /// A bus shared across AIRs by name.
    Global {
        /// Name every participating declaration agrees on.
        name: String,
    },
    /// A bus private to one declaration, which must therefore balance on its own.
    Local {
        /// Position of the declaring AIR in caller order.
        instance_index: usize,
        /// Position of the declaration within that AIR, in emission order.
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
    /// Bus identifier for each declaration, in the same order.
    pub(crate) bus_ids: Vec<usize>,
}

/// Where every declared tuple lives inside the padded fraction tables.
///
/// Prover and verifier derive this independently from the AIRs and the trace heights.
/// No layout metadata is ever read out of a proof.
///
/// Nothing here holds trace data.
/// A block's position follows from its instance's offset and height, plus its declaration rank.
///
/// ```text
///     tallest AIR    | tuple 0 | tuple 1 | tuple 2 |
///     shorter AIR                                  | t0 | t1 |
///     padding                                                 | 0 / 1 ... |
/// ```
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
    /// Returns nothing when no AIR in the batch declares a lookup.
    /// That is the signal that this proof carries no lookup section at all.
    ///
    /// # Arguments
    ///
    /// - `airs`: every AIR in the batch, in caller order.
    /// - `num_variables`: base-two logarithm of each AIR's trace height, in the same order.
    ///
    /// # Errors
    ///
    /// Returns an error if the worst-case multiplicity sum reaches the field characteristic.
    /// A multiplicity could otherwise wrap around and forge a balanced bus.
    ///
    /// # Panics
    ///
    /// Panics if an AIR declares mutually-exclusive interactions.
    /// Panics if a lookup expression reads a periodic column.
    /// Panics if two declarations share a bus name but disagree on payload width.
    /// Panics if no declaration carries a payload element.
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

        // Evaluate each AIR symbolically once.
        // Its declared weights and its trace height fix the multiplicity soundness bound.
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

        // Keep only the AIRs that emit at least one tuple.
        // A local declaration with no tuple is inert.
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

        // Tallest traces come first.
        // Every height is a power of two.
        // The blocks placed before an instance therefore leave its offset aligned to its height.
        active.sort_by_key(|instance| Reverse(instance.num_variables));

        // No active lookup means this batch has no lookup transcript or proof section.
        if active.is_empty() {
            return Ok(None);
        }

        // Bus identifiers keep unrelated lookup arguments from cancelling against each other.
        // The widest payload fixes how many beta powers exist.
        // The running height counts the leaves in use before padding.
        let mut bus_to_id = BTreeMap::new();
        let mut max_width = 0;
        let mut active_height = 0;

        for instance in &mut active {
            let trace_height = 1usize << instance.num_variables;

            // Every tuple owns one contiguous block as tall as this instance's trace.
            // An instance's blocks are adjacent, in declaration order.
            instance.base_offset = active_height;
            let contribution_count = instance
                .lookups
                .iter()
                .map(|lookup| lookup.elements.len())
                .sum::<usize>();
            let instance_height = contribution_count * trace_height;
            active_height += instance_height;

            // Record one bus identifier per declaration.
            // A local bus is scoped to its declaring AIR.
            // Global buses with the same name share one identifier.
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

                    // The widest payload fixes which beta power is left free for the bus offset.
                    let width = lookup.elements.first().map_or(0, Vec::len);
                    let next_id = bus_to_id.len();

                    // The first declaration on a bus fixes both its identifier and its width.
                    // A second width would let a short tuple alias a longer one ending in zeroes.
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

        // The bus offset sits one beta power above every payload coordinate.
        // An all-empty payload leaves no power free for it.
        assert!(
            max_width > 0,
            "lookup declarations must carry at least one payload element"
        );

        Ok(Some(Self {
            instances: active,
            max_width,
            num_buses: bus_to_id.len(),
            num_variables: log2_ceil_usize(active_height),
        }))
    }
}

/// The claim that hands the lookup reduction over to the AIR sumcheck.
///
/// Fractional GKR ends by opening the numerator and denominator tables at one random point `q`.
/// A fresh scalar `theta` folds those two openings into a single value:
///
/// ```text
///     claimed_sum = N(q) + theta * (D(q) - 1)
/// ```
///
/// The AIR sumcheck then proves the same value from the trace columns:
///
/// ```text
///     claimed_sum = sum_b w_b * sum_x eq(q_b, x) * (m_b(x) + theta * (D_b(x) - 1))
/// ```
///
/// Here `b` ranges over declared tuples.
/// `w_b` selects tuple `b`'s block.
/// `q_b` is the row-coordinate suffix of `q`.
///
/// Shifting each denominator by one is what makes the padding free.
/// An unused leaf materializes as `0 / 1`, whose shifted denominator is zero.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct AirLinkClaim<EF> {
    /// Point at which the reduction opened both tables.
    ///
    /// The zerocheck reuses it as the suffix of its own point.
    /// That is what ties the two arguments to the same random coordinates.
    pub(crate) point: Point<EF>,
    /// The folded opening the AIR sumcheck must reproduce.
    pub(crate) claimed_sum: EF,
    /// Coefficient `theta * beta^k` applied to payload coordinate `k`.
    ///
    /// A narrower payload simply leaves the tail of this vector unused.
    pub(crate) theta_beta_powers: Vec<EF>,
    /// Per-AIR link data, keyed by position in caller order.
    ///
    /// Only lookup-declaring AIRs appear.
    pub(crate) links_by_air: BTreeMap<usize, AirLinkInstance<EF>>,
}

/// Lookup-link coefficients for the declarations of one AIR.
///
/// Declarations are stored locals first, then globals, matching the order the AIR emits them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct AirLinkInstance<EF> {
    /// How many of the stored declarations are local.
    ///
    /// The global declarations start right after them.
    pub(crate) num_local_lookups: usize,
    /// One entry per lookup declaration, locals before globals.
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

    /// Rebuild the same coefficients in another representation.
    ///
    /// The packed round kernels lift every scalar into a SIMD lane group once with this.
    /// The row loop then never broadcasts a scalar again.
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

/// AIR-side coefficients tying one lookup declaration to the blocks that materialized it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct AirLinkLookup<EF> {
    /// The constant part of `theta * (D - 1)`, shared by every tuple on this bus.
    ///
    /// Only the payload fingerprint is left for the folder to subtract.
    pub(crate) theta_bus_offset: EF,
    /// Equality weight selecting each tuple's own block, in declaration order.
    pub(crate) block_weights: Vec<EF>,
}

impl<EF: Field> AirLinkClaim<EF> {
    /// Turn the reduction's output into the claim and coefficients the AIR sumcheck needs.
    ///
    /// Both sides run this identically.
    /// It reads only the plan, the challenges, and the reduction output, never the trace.
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

                // The instance's offset is aligned to its own height.
                // Shifting it therefore yields the index of its first block directly.
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
                            // Keep only what the AIR folder still has to rebuild.
                            theta_bus_offset: theta * (challenges.bus_prefix[bus_id] - EF::ONE),
                            block_weights,
                        }
                    })
                    .collect();

                (
                    planned.air_index,
                    AirLinkInstance {
                        // Declarations arrive locals first, then globals.
                        // The folder uses this boundary to address each kind in emission order.
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

    /// Split the single folded claim into one private per-AIR share.
    ///
    /// The verifier only ever sees the total.
    /// The prover needs the shares because each AIR joins the sumcheck at its own trace height.
    /// A stage's round polynomial must start from the part of the claim it owns.
    ///
    /// ```text
    ///     point = [ block selectors | row coordinates ]
    ///                                ^ shared by every AIR of one height
    /// ```
    ///
    /// Grouping instances by height lets one equality table serve every block of a group.
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

/// What the prover carries from the lookup reduction into the zerocheck.
pub(crate) struct ActiveLookupRuntime<EF> {
    /// Each AIR's share of the folded claim, keyed by position in caller order.
    ///
    /// These are prover-private: the verifier reconstructs only their sum.
    pub(crate) claims_by_air: BTreeMap<usize, EF>,
    /// The claim and coefficients the verifier derives for itself.
    pub(crate) air_link: AirLinkClaim<EF>,
}

/// Whether this batch has a lookup argument to couple into the zerocheck.
pub(crate) enum LookupRuntime<EF> {
    /// No AIR declares a lookup, so the zerocheck runs on ordinary constraints alone.
    Inactive,
    /// The reduction ran and produced state to couple in.
    Active(ActiveLookupRuntime<EF>),
}

impl<EF: Field> LookupRuntime<EF> {
    /// Check the coupling state before the zerocheck starts consuming it.
    ///
    /// # Panics
    ///
    /// Panics if the claims and the link data disagree, or name an AIR outside the batch.
    pub(crate) fn validate(&self, num_airs: usize) {
        if let Self::Active(runtime) = self {
            runtime.validate(num_airs);
        }
    }
}

impl<EF: Field> ActiveLookupRuntime<EF> {
    /// Check the coupling state before the zerocheck starts consuming it.
    ///
    /// # Panics
    ///
    /// Panics if the claims and the link data disagree, or name an AIR outside the batch.
    pub(crate) fn validate(&self, num_airs: usize) {
        let air_link = &self.air_link;

        // An active reduction always opens at a point with at least one coordinate.
        assert!(air_link.point.num_variables() > 0);
        assert!(!air_link.links_by_air.is_empty());

        // The private shares and the public link data must cover the same AIRs.
        assert!(self.claims_by_air.keys().eq(air_link.links_by_air.keys()));

        // Every linked AIR is in range, contributes something, and has a valid local prefix.
        assert!(air_link.links_by_air.iter().all(|(&air_index, link)| {
            air_index < num_airs && !link.is_empty() && link.num_local_lookups <= link.lookups.len()
        }));
    }
}

/// Reasons the lookup phase rejects a proof.
#[derive(Debug, Error)]
pub enum LookupError {
    /// An AIR declares a lookup, but the proof carries no reduction for it.
    #[error("lookup proof expected but absent")]
    MissingProof,
    /// The proof carries a lookup reduction, but no AIR declares a lookup.
    #[error("lookup proof present but no AIR declares interactions")]
    UnexpectedProof,
    /// The worst-case multiplicity sum could wrap around the field characteristic.
    #[error(transparent)]
    MultiplicityHeightBound(#[from] p3_lookup::LookupError),
    /// The fractional reduction failed its own consistency checks.
    #[error("fractional GKR: {0}")]
    FractionGkr(#[from] FractionGkrError),
}

/// Materialize the lookup fractions and prove that they sum to zero.
///
/// The four input slices are all keyed by the AIR's position in caller order.
///
/// # Returns
///
/// - the reduction proof, absent when no AIR declares a lookup;
/// - the state coupling the reduction into the zerocheck.
///
/// # Panics
///
/// Panics if the input slices disagree on length.
/// Panics if the worst-case multiplicity sum reaches the field characteristic.
/// Panics if a lookup-active trace is shorter than the prover's SIMD packing width.
pub(crate) fn prove_lookup<F, EF, A, Challenger>(
    airs: &[&A],
    main: &[&Table<F>],
    preprocessed: &[Option<&Table<F>>],
    public_values: &[&[F]],
    challenger: &mut Challenger,
) -> (Option<FractionGkrProof<EF>>, LookupRuntime<EF>)
where
    F: PrimeField,
    EF: ExtensionField<F>,
    A: BaseAir<F> + Air<InteractionSymbolicBuilder<F, EF>>,
    Challenger: FieldChallenger<F>,
{
    // The four input slices use the original AIR index as their shared key.
    assert_eq!(airs.len(), main.len());
    assert_eq!(airs.len(), preprocessed.len());
    assert_eq!(airs.len(), public_values.len());

    // Phase 1: plan.
    // Extract the declarations, order the AIRs by height, and give every tuple a block.
    // A lookup-free batch touches no transcript and returns inactive state.
    let num_variables = main
        .iter()
        .map(|table| table.num_variables())
        .collect::<Vec<_>>();
    let Some(plan) = LookupPlan::build::<EF, A>(airs, &num_variables)
        .expect("lookup multiplicity height bound must hold")
    else {
        return (None, LookupRuntime::Inactive);
    };

    // Phase 2: sample the fingerprint challenges.
    // Beta combines the payload coordinates.
    // Alpha plus the reserved beta power give each bus its own prefix.
    let alpha: EF = challenger.sample_algebra_element();
    let beta: EF = challenger.sample_algebra_element();

    // Phase 3: reduce.
    // Materialize every `multiplicity / denominator` fraction.
    // Prove their padded sum is zero and open both tables at one output point.
    let fraction = plan.materialize_fraction(main, preprocessed, public_values, alpha, beta);
    let (fractional_gkr, output) = prove_fractional_gkr(&fraction, challenger);

    // Theta is drawn only after the reduction has fixed its point and openings.
    // It folds the two openings into the one claim the zerocheck carries:
    //
    //     N(point) + theta * (D(point) - 1)
    let theta: EF = challenger.sample_algebra_element();
    let air_link = AirLinkClaim::new(&plan, alpha, beta, output, theta);

    // Split that claim into the private per-AIR shares each zerocheck stage starts from.
    // The shares are prover bookkeeping only, since the sumcheck starts from their sum.
    let claims_by_air = air_link.evaluate_instance_claims(&plan, &fraction, theta);
    debug_assert_eq!(
        claims_by_air.values().copied().sum::<EF>(),
        air_link.claimed_sum
    );

    // The proof material and the private coupling state travel separately.
    let runtime = ActiveLookupRuntime {
        claims_by_air,
        air_link,
    };

    (Some(fractional_gkr), LookupRuntime::Active(runtime))
}

/// Verify the fractional reduction and rebuild the claim the zerocheck must reproduce.
///
/// Returns nothing when no AIR declares a lookup and the proof agrees.
///
/// # Errors
///
/// Returns an error when the proof and the AIRs disagree on whether a lookup exists.
/// Returns an error when the worst-case multiplicity sum reaches the field characteristic.
/// Returns an error when the reduction fails its own consistency checks.
///
/// # Panics
///
/// Panics if the AIR and trace-height slices disagree on length.
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

    // Rebuild the layout from the verifier's own AIRs and trace heights.
    // No layout metadata from the proof is trusted.
    let plan = LookupPlan::build::<EF, A>(airs, num_variables)?;

    // A lookup proof exists exactly when at least one AIR emits a tuple.
    let (plan, proof) = match (plan, proof) {
        (None, None) => return Ok(None),
        (None, Some(_)) => return Err(LookupError::UnexpectedProof),
        (Some(_), None) => return Err(LookupError::MissingProof),
        (Some(plan), Some(proof)) => (plan, proof),
    };

    // Replay the lookup challenges from the statement-bound transcript.
    let alpha: EF = challenger.sample_algebra_element();
    let beta: EF = challenger.sample_algebra_element();

    // Check the reduction over the padded layout.
    // It yields the numerator and denominator openings at its output point.
    let output = verify_fractional_gkr::<F, EF, _>(proof, plan.num_variables, challenger)?;

    // As on the prover side, theta is drawn only after the reduction fixes its output.
    // It folds the two openings into the one claim the zerocheck carries.
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
    use p3_sumcheck::generic_degree::GenericDegreeProof;
    use p3_util::log2_strict_usize;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::zerocheck::{AirZerocheck, ZerocheckError};

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
            // A degree-one ordinary constraint beside the higher-degree lookup expression.
            // It is identically satisfied, so it only exercises the two-family degree split.
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
    fn lookup_plan_is_independent_of_the_build_target_packing_width() {
        // Invariant: planning is what the verifier replays.
        // Nothing in it may depend on how wide this build's SIMD lanes happen to be.
        //
        // Fixture state: one lookup AIR over a 2-row trace.
        // That is shorter than every packing width above one.
        //
        //     scalar build  (width 1)  -> plan built
        //     avx2   build  (width 8)  -> plan built, identically
        //     avx512 build  (width 16) -> plan built, identically
        let air = BalancedLookupAir;
        let plan = LookupPlan::build::<EF, _>(&[&air], &[1])
            .expect("multiplicity bound holds")
            .expect("the AIR declares a lookup");

        // Two tuples of a 2-row trace pad up to 4 leaves, whatever the packing width is.
        assert_eq!(plan.num_variables, 2);
    }

    #[test]
    fn materialization_rejects_a_trace_shorter_than_the_packing_width() {
        let packing_variables = log2_strict_usize(<F as Field>::Packing::WIDTH);
        if packing_variables == 0 {
            // Scalar packing has width one, so no nonempty trace can be shorter.
            return;
        }

        // Mutation: give the prover a trace one variable below its own lane group.
        //
        //     block rows    : 2^(packing_variables - 1)
        //     rows per lane : 2^packing_variables
        //     -----> zero packed entries per block, so the block would stay unwritten
        let air = BalancedLookupAir;
        let mut rng = SmallRng::seed_from_u64(0x5170_2ACE);
        let main = Table::<F>::rand(&mut rng, 1, packing_variables - 1);
        let plan = LookupPlan::build::<EF, _>(&[&air], &[main.num_variables()])
            .unwrap()
            .unwrap();

        let payload = std::panic::catch_unwind(|| {
            plan.materialize_fraction(&[&main], &[None], &[&[]], EF::ONE, EF::ONE)
        })
        .expect_err("materialization must refuse a trace below its lane group");

        // A panic payload is an owned string only when the message was formatted.
        // A plain message arrives as a static string slice, so accept both shapes.
        let message = payload
            .downcast_ref::<std::string::String>()
            .map(std::string::String::as_str)
            .or_else(|| payload.downcast_ref::<&'static str>().copied())
            .expect("the panic carries a message");
        assert!(message.contains("SIMD packing width"), "{message}");
    }

    #[test]
    #[should_panic(expected = "multi-STARK lookups do not support exclusive interactions")]
    fn lookup_plan_rejects_exclusive_interactions() {
        // A mutually-exclusive interaction multiplexes several branches into one denominator,
        // which the block layout here has no place for. Reject it rather than mislay it.
        let air = ExclusiveLookupAir;
        let _ = LookupPlan::build::<EF, _>(&[&air], &[test_lookup_num_variables()]);
    }

    #[test]
    #[should_panic(expected = "multi-STARK lookup expressions do not support periodic columns")]
    fn lookup_plan_rejects_periodic_expressions() {
        // Materialization never expands periodic columns, so a lookup expression that reads
        // one would resolve against values that are not there.
        let air = PeriodicLookupAir;
        let _ = LookupPlan::build::<EF, _>(&[&air], &[test_lookup_num_variables()]);
    }

    #[test]
    fn lookup_verifier_rejects_multiplicity_bound_at_field_characteristic() {
        // Invariant: the worst-case multiplicity sum must stay below the characteristic,
        // or a bus could wrap around to zero and look balanced while it is not.
        //
        // Fixture state: a per-row count bound of 2^32 - 1 over a 16-row trace.
        //
        //     2^32 - 1 rows worth of weight >> the BabyBear characteristic
        //     -----> rejected before any transcript is replayed
        let air = ExcessiveMultiplicityAir;
        let mut challenger = challenger();
        assert!(matches!(
            verify_lookup::<F, EF, _, _>(
                &[&air],
                &[test_lookup_num_variables()],
                Some(&FractionGkrProof {
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
        // Invariant: prover and verifier derive the same claim from the same transcript,
        // and the prover's private shares add up to that claim.
        //
        //     prover  : materializes the fractions, proves the reduction
        //     verifier: replays the transcript, rebuilds the claim from the AIRs alone
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
        // Everything the verifier rebuilt matches what the prover carries forward.
        assert_eq!(verifier, prover.air_link);
        // The private per-AIR shares are exactly a decomposition of the public claim.
        assert_eq!(
            prover.claims_by_air.values().copied().sum::<EF>(),
            prover.air_link.claimed_sum
        );

        // Both transcripts absorbed and drew the same values, so they stay in lockstep.
        let prover_final: EF = prover_challenger.sample_algebra_element();
        let verifier_final: EF = verifier_challenger.sample_algebra_element();
        assert_eq!(prover_final, verifier_final);
    }

    #[test]
    fn air_link_claim_decomposes_gkr_openings_into_lookup_blocks() {
        // Invariant: the folded claim equals the weighted sum over the individual blocks.
        //
        // Fixture state: an AIR declaring 3 tuples, so 3 blocks plus one of padding.
        //
        //     leaves : | tuple 0 | tuple 1 | tuple 2 | 0 / 1 padding |
        let air = PaddedLookupAir;
        let mut rng = SmallRng::seed_from_u64(0xA17_11A5);
        let main = Table::<F>::rand(&mut rng, 1, test_lookup_num_variables());
        let plan = LookupPlan::build::<EF, _>(&[&air], &[main.num_variables()])
            .unwrap()
            .unwrap();
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
        // Rebuild the claim block by block, the long way round.
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
                // Each block contributes its own opening, shifted and weighted.
                expected += weight * (block_numer + theta * (block_denom - EF::ONE));
                contribution_index += 1;
            }
        }

        // The single opening at the reduction point already equals that whole sum.
        assert_eq!(air_link.claimed_sum, expected);
    }

    #[test]
    fn fractional_gkr_is_coupled_to_air_interactions() {
        // Invariant: the zerocheck reproduces the reduction's claim from the AIR's columns,
        // and both sides end on the same bound point with matching transcripts.
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
        // Invariant: AIRs of different lookup degrees share one transmitted round polynomial,
        // each contributing only up to its own degree.
        //
        //     degrees : 2, 3, 3, 5  ->  four AIRs, three lookup degree groups
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
        let preprocessed = vec![None; airs.len()];
        let public_values = vec![&[][..]; airs.len()];

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
        // Invariant: a shorter AIR's lookup claim stays dormant until the cube reaches its
        // height, and the global claim is exactly the sum of the two shares throughout.
        //
        //     rounds : | block selectors | 64-row stage | 16-row stage |
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
        // Block selectors make the shared cube taller than even the tallest trace.
        assert!(prover_point.num_variables() > tall.num_variables());
        let prover_final: EF = prover_challenger.sample_algebra_element();
        let verifier_final: EF = verifier_challenger.sample_algebra_element();
        assert_eq!(prover_final, verifier_final);
    }

    #[test]
    #[should_panic(expected = "declares lookups but the batch carries no lookup reduction")]
    fn lookup_free_prover_entry_point_refuses_a_lookup_air() {
        // Invariant: a lookup AIR still runs through the ordinary folder.
        // That folder accepts lookup declarations and drops them.
        //
        // Mutation: prove a lookup AIR through the entry point that carries no reduction.
        //
        //     AIR declares  : 2 tuples on a local bus
        //     batch carries : no lookup link
        //     -----> rejected rather than proved with the lookups dropped
        let air = BalancedLookupAir;
        let mut rng = SmallRng::seed_from_u64(0xD1_5CA2D);
        let main = Table::<F>::rand(&mut rng, 1, 6);
        let airs = [&air];

        let _ = AirZerocheck::new(&airs, 0).prove::<F, EF, _>(
            &[None],
            &[&main],
            &[&[]],
            &mut challenger(),
        );
    }

    #[test]
    fn lookup_free_verifier_entry_point_refuses_a_lookup_air() {
        // The prover refuses to build such a proof at all.
        // The verifier is therefore handed a transcript that cannot exist.
        // It must still reject, rather than accept a proof whose lookups went unchecked.
        //
        // Fixture state: an empty transcript replayed against a lookup-declaring AIR.
        let air = BalancedLookupAir;
        let airs = [&air];
        let sumcheck = AirZerocheck::new(&airs, 0);
        let proof = GenericDegreeProof::<F, EF> {
            claimed_sum: EF::ZERO,
            round_polys: Vec::new(),
            pow_witnesses: Vec::new(),
        };

        // The AIR at position zero declares lookups that no link accompanies.
        assert!(matches!(
            sumcheck.verify_reduction::<F, EF, _>(&proof, &[6], &[&[]], &mut challenger()),
            Err(ZerocheckError::LookupLinkMismatch { air: 0 })
        ));
    }
}
