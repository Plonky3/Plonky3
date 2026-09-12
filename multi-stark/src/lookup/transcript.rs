//! Fiat-Shamir transcript of the multilinear lookup argument.
//!
//! # Overview
//!
//! One statement of what the lookup argument draws, consumed by both sides.
//!
//! It is built from the plan, and the plan from the AIRs and their trace heights.
//! Neither side ever reads a count out of a proof.
//!
//! # Shape
//!
//! ```text
//!     alpha                          one extension element
//!     beta                           one extension element
//!     Begin  fraction reduction      bracket around the delegated run
//!     End    fraction reduction
//!     theta                          one extension element
//! ```
//!
//! # What is bound
//!
//! - Shape: nothing. Every batch draws the same three challenges around one bracket.
//! - Instance label: the whole plan, down to the bus each declaration speaks on.
//! - Nothing here: the reduction's own numbers, which its seed binds inside the bracket.
//!
//! # Soundness
//!
//! The two fingerprint challenges precede everything the fractions are built from.
//!
//! ```text
//!     draw alpha, beta  ->  materialize m_b / (prefix_b - sum_k beta^k * payload_bk)
//! ```
//!
//! A prover who learned them afterwards could choose payloads that cancel.
//!
//! Theta comes last for the mirror reason.
//! It folds the reduction's two openings into one claim, so it is drawn only once both are fixed.

use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::fs::{
    DomainSeparator, FieldToFieldCodec, FieldUnit, Hierarchy, Interaction, InteractionPattern,
    Kind, Length, ProverState, TranscriptField, VerifierState,
};
use p3_challenger::{CanObserve, CanSample};
use p3_field::{ExtensionField, Field};

use super::LookupPlan;

/// Version byte bound into the lookup transcript seed.
const VERSION: u8 = 1;

/// Protocol name bound into the lookup transcript seed.
const NAME: &[u8] = b"p3-multi-stark-lookup";

/// Step label of the challenge offsetting each bus.
const ALPHA: &str = "alpha";

/// Step label of the challenge combining the payload coordinates.
const BETA: &str = "beta";

/// Step label of the challenge folding the two reduction openings into one claim.
const THETA: &str = "theta";

/// Step label of the bracket around the delegated fractional reduction.
const FRACTION_REDUCTION: &str = "fraction_reduction";

/// Sponge alphabet of a challenger that speaks the base field natively.
type Alphabet<F> = FieldUnit<F>;

/// Type-level name of the sub-protocol the lookup argument delegates its reduction to.
///
/// Recorded on the bracket markers as a local diagnostic.
/// It does not reach the pattern fingerprint.
struct FractionReduction;

/// Placement of one lookup-active AIR inside the padded fraction tables.
///
/// Every number here is read off the plan, which both sides derive from their own AIRs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LookupInstanceShape {
    /// Position of this AIR in caller order.
    pub air_index: usize,
    /// Base-two logarithm of this AIR's trace height.
    pub num_variables: usize,
    /// First scalar leaf this AIR owns in the materialized fraction tables.
    pub base_offset: usize,
    /// Bus identifier of each nonempty declaration, in emission order.
    pub bus_ids: Vec<usize>,
}

/// Numbers that fix the transcript of one lookup argument.
///
/// Both sides build this from the plan, and the plan from the AIRs and the trace heights.
/// No number here ever comes from a proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LookupShape {
    /// Variable count of the padded fraction tables the reduction consumes.
    pub num_variables: usize,
    /// Largest payload tuple width among all planned declarations.
    pub max_width: usize,
    /// Number of distinct local and named global buses the plan assigns.
    pub num_buses: usize,
    /// One entry per lookup-active AIR, in the plan's descending-height order.
    pub instances: Vec<LookupInstanceShape>,
}

impl LookupShape {
    /// Read the shape off a plan.
    ///
    /// # Arguments
    ///
    /// - `plan`: where every declared tuple lives inside the padded fraction tables.
    #[must_use]
    pub fn new<F: Field>(plan: &LookupPlan<F>) -> Self {
        Self {
            num_variables: plan.num_variables,
            max_width: plan.max_width,
            num_buses: plan.num_buses,
            instances: plan
                .instances
                .iter()
                .map(|instance| LookupInstanceShape {
                    air_index: instance.air_index,
                    num_variables: instance.num_variables,
                    base_offset: instance.base_offset,
                    bus_ids: instance
                        .lookups
                        .iter()
                        .zip(&instance.bus_ids)
                        .filter_map(|(lookup, &id)| (!lookup.elements.is_empty()).then_some(id))
                        .collect(),
                })
                .collect(),
        }
    }

    /// Describe the transcript this shape fixes.
    ///
    /// Three challenges and one bracket, whatever the batch looks like.
    ///
    /// # Panics
    ///
    /// Never in practice.
    /// One matched bracket always passes structural validation.
    #[must_use]
    pub fn pattern<F, EF>() -> InteractionPattern
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
    {
        let challenge = |label| {
            Interaction::algebra::<F, EF>(Hierarchy::Atomic, Kind::Challenge, label, Length::Scalar)
        };

        InteractionPattern::new(vec![
            // Beta combines the payload coordinates.
            // Alpha and a reserved beta power give each bus its own prefix.
            // Both are drawn before any fraction is materialized.
            challenge(ALPHA),
            challenge(BETA),
            // The bracket records that a sub-protocol runs here.
            //
            // Its steps live in the callee's own pattern, under the callee's own seed.
            // What this pattern states is that the delegation happens, and where.
            Interaction::marker::<FractionReduction>(
                Hierarchy::Begin,
                Kind::Protocol,
                FRACTION_REDUCTION,
            ),
            Interaction::marker::<FractionReduction>(
                Hierarchy::End,
                Kind::Protocol,
                FRACTION_REDUCTION,
            ),
            // Theta is drawn only once the reduction has fixed its point and its openings.
            challenge(THETA),
        ])
        .expect("one matched bracket is always well formed")
    }

    /// Bind the protocol identity and the whole plan.
    ///
    /// The step sequence is the same for every batch, so nothing rides in the fingerprint.
    /// Every number that tells two batches apart therefore goes in the instance label.
    #[must_use]
    pub fn domain_separator<F, EF>(&self) -> DomainSeparator<Alphabet<F>>
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
    {
        let mut separator = DomainSeparator::new(VERSION, NAME, Self::pattern::<F, EF>());

        // The padded table width, the beta power reserved for the bus offset,
        // and how many distinct buses share the argument.
        separator
            .instance(&(self.num_variables as u64).to_be_bytes())
            .instance(&(self.max_width as u64).to_be_bytes())
            .instance(&(self.num_buses as u64).to_be_bytes())
            .instance(&(self.instances.len() as u64).to_be_bytes());

        // Where each AIR's blocks sit, and which bus each of its declarations speaks on.
        //
        // Two batches can agree on every total above and still place their blocks
        // differently, or route one declaration onto another bus.
        for instance in &self.instances {
            separator
                .instance(&(instance.air_index as u64).to_be_bytes())
                .instance(&(instance.num_variables as u64).to_be_bytes())
                .instance(&(instance.base_offset as u64).to_be_bytes())
                .instance(&(instance.bus_ids.len() as u64).to_be_bytes());
            for &bus_id in &instance.bus_ids {
                separator.instance(&(bus_id as u64).to_be_bytes());
            }
        }

        separator
    }
}

/// Prover-side transcript of one lookup argument.
///
/// Holds the only definition of what a prover draws around the reduction.
///
/// The challenger is borrowed, not consumed.
/// The lookup argument runs inside a STARK whose transcript continues afterwards.
pub struct LookupProverTranscript<'a, C, F: TranscriptField, EF> {
    /// Driver walking the description and holding the borrowed sponge.
    state: ProverState<&'a mut C, Alphabet<F>>,
    /// Marker for the extension field the challenges live in.
    _ef: PhantomData<EF>,
}

impl<'a, C, F, EF> LookupProverTranscript<'a, C, F, EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F>,
{
    /// Seed the transcript from the shape.
    pub fn new(challenger: &'a mut C, shape: &LookupShape) -> Self {
        let separator = shape.domain_separator::<F, EF>();
        Self {
            state: ProverState::new(challenger, &separator),
            _ef: PhantomData,
        }
    }

    /// Draw the two challenges that fingerprint every declared tuple.
    ///
    /// # Returns
    ///
    /// The bus prefix base, then the payload combiner.
    pub fn fingerprint_challenges(&mut self) -> (EF, EF) {
        let alpha = self
            .state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(ALPHA)
            .into_inner();
        let beta = self
            .state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(BETA)
            .into_inner();
        (alpha, beta)
    }

    /// Lend the sponge to the fractional reduction, bracketed as a sub-protocol.
    ///
    /// The callee seeds its own driver from the state this one has reached.
    ///
    /// # Returns
    ///
    /// Whatever the delegated run produced.
    pub fn reduction<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        self.state
            .begin_protocol::<FractionReduction>(FRACTION_REDUCTION);
        let output = run(self.state.challenger_mut());
        self.state
            .end_protocol::<FractionReduction>(FRACTION_REDUCTION);
        output
    }

    /// Draw the challenge folding the reduction's two openings into one claim.
    pub fn link_challenge(&mut self) -> EF {
        self.state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(THETA)
            .into_inner()
    }

    /// Close the transcript once every described step has been played.
    ///
    /// # Panics
    ///
    /// When the run played fewer steps than it was described with.
    pub fn finish(self) {
        assert!(
            self.state.finalize().is_empty(),
            "the lookup argument carries every value in its own proof",
        );
    }
}

/// Verifier-side transcript of one lookup argument.
///
/// Mirrors the prover side call for call, over the same description.
pub struct LookupVerifierTranscript<'a, C, F: TranscriptField, EF> {
    /// Driver walking the description and holding the borrowed sponge.
    ///
    /// The proof carries every value, so the driver reads an empty wire.
    state: VerifierState<'static, &'a mut C, Alphabet<F>>,
    /// Marker for the extension field the challenges live in.
    _ef: PhantomData<EF>,
}

impl<'a, C, F, EF> LookupVerifierTranscript<'a, C, F, EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F>,
{
    /// Seed the transcript from the shape.
    pub fn new(challenger: &'a mut C, shape: &LookupShape) -> Self {
        let separator = shape.domain_separator::<F, EF>();
        Self {
            state: VerifierState::new(challenger, &separator, &[]),
            _ef: PhantomData,
        }
    }

    /// Redraw the two challenges that fingerprint every declared tuple.
    ///
    /// # Returns
    ///
    /// The bus prefix base, then the payload combiner.
    pub fn fingerprint_challenges(&mut self) -> (EF, EF) {
        let alpha = self
            .state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(ALPHA)
            .into_inner();
        let beta = self
            .state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(BETA)
            .into_inner();
        (alpha, beta)
    }

    /// Lend the sponge to the fractional reduction, bracketed as a sub-protocol.
    ///
    /// The bracket closes whatever the delegated run returned.
    /// A rejection therefore leaves this transcript replayable to the end.
    ///
    /// # Returns
    ///
    /// Whatever the delegated run produced.
    pub fn reduction<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        self.state
            .begin_protocol::<FractionReduction>(FRACTION_REDUCTION);
        let output = run(self.state.challenger_mut());
        self.state
            .end_protocol::<FractionReduction>(FRACTION_REDUCTION);
        output
    }

    /// Redraw the challenge folding the reduction's two openings into one claim.
    pub fn link_challenge(&mut self) -> EF {
        self.state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(THETA)
            .into_inner()
    }

    /// Close the transcript once every described step has been replayed.
    ///
    /// # Panics
    ///
    /// When the run replayed fewer steps than it was described with.
    pub fn finish(self) {
        self.state
            .finalize()
            .expect("the lookup argument reads an empty wire, so no bytes can remain");
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_field::extension::BinomialExtensionField;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type Ch = DuplexChallenger<F, Perm, 16, 8>;

    fn challenger() -> Ch {
        // Fixed seed so two runs differ only where the transcript makes them differ.
        let mut rng = SmallRng::seed_from_u64(0x100C_A11E);
        Ch::new(Perm::new_from_rng_128(&mut rng))
    }

    /// Baseline shape every walk below perturbs exactly one field of.
    fn base_lookup_shape() -> LookupShape {
        LookupShape {
            num_variables: 7,
            max_width: 3,
            num_buses: 2,
            instances: vec![
                LookupInstanceShape {
                    air_index: 0,
                    num_variables: 5,
                    base_offset: 0,
                    bus_ids: vec![0, 1],
                },
                LookupInstanceShape {
                    air_index: 2,
                    num_variables: 4,
                    base_offset: 64,
                    bus_ids: vec![1],
                },
            ],
        }
    }

    /// First challenge a shape's seed produces on a fresh sponge.
    fn first_lookup_challenge(shape: &LookupShape) -> F {
        let mut sponge = challenger();
        shape.domain_separator::<F, EF>().seed(&mut sponge);
        sponge.sample()
    }

    /// Assert that perturbing one field of the shape moves the seed.
    fn lookup_field_moves_the_seed(name: &str, perturb: impl FnOnce(&mut LookupShape)) {
        let mut shape = base_lookup_shape();
        perturb(&mut shape);
        assert_ne!(
            first_lookup_challenge(&base_lookup_shape()),
            first_lookup_challenge(&shape),
            "changing {name} left the seed where it was",
        );
    }

    #[test]
    fn the_same_plan_seeds_the_same_stream_twice() {
        // Completeness: the seed is a pure function of the shape.
        let shape = base_lookup_shape();
        assert_eq!(
            first_lookup_challenge(&shape),
            first_lookup_challenge(&shape)
        );
    }

    #[test]
    fn every_field_of_the_plan_reaches_the_lookup_seed() {
        // Each row below perturbs the shape and asserts the seed moved with it.
        //
        // Every batch describes one step sequence, so none of these can ride in the fingerprint.
        // This walk is the check that the label carries them all.
        lookup_field_moves_the_seed("num_variables", |s| s.num_variables += 1);
        lookup_field_moves_the_seed("max_width", |s| s.max_width += 1);
        lookup_field_moves_the_seed("num_buses", |s| s.num_buses += 1);
        lookup_field_moves_the_seed("instances.len", |s| {
            s.instances.pop();
        });
        lookup_field_moves_the_seed("instance.air_index", |s| s.instances[0].air_index += 1);
        lookup_field_moves_the_seed("instance.num_variables", |s| {
            s.instances[1].num_variables += 1;
        });
        lookup_field_moves_the_seed("instance.base_offset", |s| s.instances[1].base_offset += 1);
        lookup_field_moves_the_seed("instance.bus_ids", |s| s.instances[0].bus_ids[1] = 0);
        lookup_field_moves_the_seed("instance.bus_ids.len", |s| {
            s.instances[0].bus_ids.pop();
        });
    }

    #[test]
    fn two_plans_agreeing_on_every_total_still_split_the_seed() {
        // Both plans hold two AIRs, two buses, the same widths and the same heights.
        //
        //     plan A: AIR 0 speaks on bus 0, AIR 2 speaks on bus 1
        //     plan B: AIR 0 speaks on bus 1, AIR 2 speaks on bus 0
        //
        // Every count agrees, so only the routing tells them apart.
        // A shared seed would let a proof for one be replayed against the other.
        let mut swapped = base_lookup_shape();
        swapped.instances[0].bus_ids = vec![1, 0];

        assert_ne!(
            first_lookup_challenge(&base_lookup_shape()),
            first_lookup_challenge(&swapped)
        );
    }
}
