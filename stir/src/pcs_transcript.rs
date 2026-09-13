//! Fiat-Shamir transcript of the STIR polynomial commitment scheme.
//!
//! # Overview
//!
//! Three descriptions, one per phase the commitment scheme owns for itself.
//!
//! - Absorbing one commitment: a list of Merkle roots.
//! - Binding the claimed evaluations, before anything batches them.
//! - Merging the height classes, delegating the proximity test, pinning the input rows.
//!
//! The batching grind and its challenge sit between the second phase and the third.
//!
//! That phase carries its own description.
//!
//! # Shape
//!
//! ```text
//!     commitment phase
//!       one opaque root per shared-domain group
//!
//!     claim phase
//!       Begin commitment
//!         Begin matrix
//!           claimed values         one step per opening point, of that matrix's width
//!         End   matrix
//!       End   commitment
//!
//!     opening phase
//!       Begin bucket combine
//!         combination challenge    only when the bucket merges several heights
//!       End   bucket combine
//!       Begin proximity test       bracket around the delegated run
//!       End   proximity test
//!       Begin bucket lanes
//!         lane indices             one uniform draw per first-round query
//!       End   bucket lanes
//! ```
//!
//! # Why the claim phase nests
//!
//! The claimed evaluations arrive grouped three levels deep.
//!
//! ```text
//!     commitment  ->  matrix  ->  opening point  ->  one value per column
//! ```
//!
//! Flattened to a bare run of steps, two groupings collapse onto one description.
//!
//! ```text
//!     two matrices of width 3, one point each   ->  3, 3
//!     one matrix of width 3, opened twice       ->  3, 3
//! ```
//!
//! A container per level keeps them apart inside the fingerprint itself.
//!
//! ```text
//!     Begin commitment  Begin matrix  3  End  Begin matrix  3  End  End
//!     Begin commitment  Begin matrix  3  3  End  End
//! ```
//!
//! Nothing about the grouping then has to travel in a separate label.
//!
//! # What is bound
//!
//! - Commitment phase: how many roots one commitment holds, through the step count.
//! - Claim phase: every claimed width, and the grouping the widths sit in.
//! - Opening phase: which buckets merge, every lane width, every lane count.
//! - Opening phase label: each bucket's shared domain and the native heights it merges.
//!
//! # What is not bound
//!
//! The parameters of the proximity test itself are absent from the opening phase.
//!
//! That run seeds its own transcript inside the bracket, under its own description.
//!
//! The bracket records that the delegation happens, and where.
//!
//! # Where the lengths come from
//!
//! Every count in every description is read off an input the caller already holds.
//!
//! ```text
//!     root count      the commitment being absorbed
//!     claim widths    the matrices a prover holds, the claims a verifier was handed
//!     bucket layout   the shared domains the claimed heights imply
//!     lane geometry   the schedule solved for each bucket
//! ```
//!
//! None of them is read out of an opening proof.

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::fs::{
    DomainSeparator, FieldToFieldCodec, FieldUnit, Hierarchy, Interaction, InteractionPattern,
    Kind, Length, ProverState, TranscriptBound, VerifierState,
};
use p3_challenger::{
    CanObserve, CanSample, CanSampleUniformBits, FieldChallenger, GrindingChallenger,
};
use p3_commit::{CommitmentOpening, MatrixOpening, Mmcs, OpenedValues, PointOpening};
use p3_field::{ExtensionField, PrimeField64, TwoAdicField};

use crate::config::StirConfig;

/// Version byte bound into every seed this module builds.
///
/// Bumping it separates two revisions of the commitment scheme.
///
/// It does so even when their step sequences agree.
const VERSION: u8 = 1;

/// Protocol name bound into the commitment-phase seed.
const COMMITMENT_NAME: &[u8] = b"p3-stir-pcs-commitment";

/// Protocol name bound into the claim-phase seed.
const CLAIM_NAME: &[u8] = b"p3-stir-pcs-claims";

/// Protocol name bound into the opening-phase seed.
const OPENING_NAME: &[u8] = b"p3-stir-pcs-opening";

/// Step label of one shared-domain group's Merkle root.
const GROUP_ROOT: &str = "group_root";

/// Container label of one commitment's claims.
const COMMITMENT_BLOCK: &str = "commitment";

/// Container label of one matrix's claims.
const MATRIX_BLOCK: &str = "matrix";

/// Step label of one opening's claimed column evaluations.
const CLAIMED_VALUES: &str = "claimed_values";

/// Container label of one bucket's class-merging block.
const COMBINE_BLOCK: &str = "bucket_combine";

/// Step label of a bucket's class-merging challenge.
const COMBINATION_CHALLENGE: &str = "combination_challenge";

/// Container label of the delegated proximity test.
const PROXIMITY_TEST: &str = "proximity_test";

/// Container label of one bucket's lane block.
const LANE_BLOCK: &str = "bucket_lanes";

/// Step label of a bucket's fiber-lane indices.
const FIBER_LANES: &str = "fiber_lanes";

/// Sponge alphabet of a challenger that speaks the base field natively.
type Alphabet<F> = FieldUnit<F>;

/// Type naming every per-commitment, per-matrix and per-bucket container.
///
/// The name is compared locally when a closer meets its opener.
///
/// It never reaches the pattern fingerprint.
type Block = ();

/// Type-level name of the delegated proximity test.
///
/// Recorded on the bracket markers as a local diagnostic.
struct ProximityTest;

/// Append `value` as eight big-endian bytes.
fn push_u64(out: &mut Vec<u8>, value: u64) {
    // A fixed width keeps a run of packed numbers self-delimiting.
    out.extend_from_slice(&value.to_be_bytes());
}

/// Append the opener of one block.
fn open(steps: &mut Vec<Interaction>, label: &'static str) {
    // A mixed container accepts nested steps of any kind, including further containers.
    steps.push(Interaction::marker::<Block>(
        Hierarchy::Begin,
        Kind::Protocol,
        label,
    ));
}

/// Append the closer of one block.
fn close(steps: &mut Vec<Interaction>, label: &'static str) {
    // The closer repeats the opener's kind, label and type.
    //
    // The validator checks that it does.
    steps.push(Interaction::marker::<Block>(
        Hierarchy::End,
        Kind::Protocol,
        label,
    ));
}

/// Number of first-round query draws one instance makes.
///
/// # Overview
///
/// The queries that read an instance's initial oracle belong to its first round.
///
/// A schedule with no intermediate round reads that oracle in the final round instead.
///
/// ```text
///     rounds > 0   ->  round zero's query count
///     rounds == 0  ->  the final round's query count
/// ```
///
/// # Returns
///
/// The count, taken from the schedule and never from a proof.
fn first_round_draw_count<F, EF, M, C>(config: &StirConfig<F, EF, M, C>) -> usize
where
    F: TwoAdicField + PrimeField64,
    EF: ExtensionField<F>,
    M: Mmcs<EF>,
    C: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    // An instance that folds at least once reads its initial oracle in round zero.
    config
        .round_configs
        .first()
        .map_or(config.final_queries, |round| round.num_queries)
}

/// Numbers that fix the transcript of absorbing one commitment.
///
/// Both sides build this from the commitment in front of them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StirPcsCommitmentShape {
    /// Number of Merkle roots the commitment holds, one per shared-domain group.
    pub num_roots: usize,
}

impl StirPcsCommitmentShape {
    /// Collect the numbers that fix one commitment absorption.
    ///
    /// # Arguments
    ///
    /// - `num_roots`: number of Merkle roots the commitment holds.
    #[must_use]
    pub const fn new(num_roots: usize) -> Self {
        Self { num_roots }
    }

    /// Describe the transcript this shape fixes.
    ///
    /// # Panics
    ///
    /// Never in practice.
    ///
    /// A flat sequence of leaf steps always passes structural validation.
    #[must_use]
    pub fn pattern(&self) -> InteractionPattern {
        // One step per root.
        //
        // The root count is the step count.
        //
        //     two roots   ->  a description of two steps
        //     three roots ->  a description of three steps
        //
        // The count therefore reaches the seed.
        //
        // It needs no absorbed length of its own.
        let steps = (0..self.num_roots)
            .map(|_| {
                Interaction::opaque(Hierarchy::Atomic, Kind::Message, GROUP_ROOT, Length::Scalar)
            })
            .collect();

        InteractionPattern::new(steps).expect("a flat sequence of leaf steps is always well formed")
    }

    /// Bind the protocol identity and this shape into a seed.
    #[must_use]
    pub fn domain_separator<F: PrimeField64>(&self) -> DomainSeparator<Alphabet<F>> {
        // The root count is the only number here.
        //
        // The fingerprint already covers it.
        DomainSeparator::new(VERSION, COMMITMENT_NAME, self.pattern())
    }
}

/// Absorb one commitment's Merkle roots.
///
/// # Overview
///
/// A commitment holds one root per shared-domain group of the matrices it covers.
///
/// The roots enter the sponge in order.
///
/// The seed already fixes how many there are.
///
/// ```text
///     one commitment of two roots   ->  seed(2 steps), root_a, root_b
///     two commitments of one root   ->  seed(1 step), root_a, then seed(1 step), root_b
/// ```
///
/// Two commitments that split a given total of roots differently therefore stay apart.
///
/// So do two commitments over the same roots in the opposite order.
///
/// # Arguments
///
/// - `challenger`: sponge of the surrounding protocol, borrowed for the absorption.
/// - `roots`: the commitment's Merkle roots, in group order.
///
/// # Panics
///
/// Never in practice.
///
/// Every described step is played before the driver is closed.
pub(crate) fn observe_commitment<Ch, F, C>(challenger: &mut Ch, roots: Vec<C>)
where
    F: PrimeField64,
    C: Clone,
    Ch: CanObserve<F> + CanObserve<C>,
{
    // Seeding folds the root count into the sponge before any root is absorbed.
    let shape = StirPcsCommitmentShape::new(roots.len());
    let separator = shape.domain_separator::<F>();
    let mut state: ProverState<&mut Ch, Alphabet<F>> = ProverState::new(challenger, &separator);

    // Each root travels as an opaque value.
    //
    // The challenger owns its encoding, not this layer.
    for root in roots {
        state.observe_opaque(GROUP_ROOT, root);
    }

    // Nothing was written to the driver's own buffer.
    //
    // Closing is purely the shape check.
    assert!(
        state.finalize().is_empty(),
        "a commitment absorption carries no wire bytes",
    );
}

/// Numbers that fix the transcript of one batch of opening claims.
///
/// Both sides build this from their own inputs, never from a proof.
///
/// The prover's inputs are the matrices it holds.
///
/// The verifier's are the claims it was asked to check.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StirPcsClaimShape {
    /// Claimed-value counts, nested by commitment, then matrix, then opening point.
    ///
    /// Each innermost entry is one matrix's column count.
    ///
    /// That is how many values one opening of it carries.
    pub claim_widths: Vec<Vec<Vec<usize>>>,
}

impl StirPcsClaimShape {
    /// Derive the shape from the values a prover is about to claim.
    ///
    /// # Arguments
    ///
    /// - `opened_values`: the claimed evaluations, nested three levels deep.
    #[must_use]
    pub fn from_opened_values<EF>(opened_values: &OpenedValues<EF>) -> Self {
        // Walk the same three levels the absorption walks, keeping only the counts.
        let claim_widths = opened_values
            .iter()
            .map(|commitment| {
                commitment
                    .iter()
                    .map(|matrix| matrix.iter().map(Vec::len).collect())
                    .collect()
            })
            .collect();

        Self { claim_widths }
    }

    /// Derive the shape from the claims a verifier was asked to check.
    ///
    /// # Arguments
    ///
    /// - `claims`: one entry per commitment, holding each matrix's claims.
    #[must_use]
    pub fn from_claims<EF, Com, Domain>(claims: &[CommitmentOpening<EF, Com, Domain>]) -> Self {
        // The claim tree has exactly the nesting the prover's value tree has.
        let claim_widths = claims
            .iter()
            .map(|claim| {
                claim
                    .matrices
                    .iter()
                    .map(|matrix| {
                        matrix
                            .points
                            .iter()
                            .map(|PointOpening { values, .. }| values.len())
                            .collect()
                    })
                    .collect()
            })
            .collect();

        Self { claim_widths }
    }

    /// Describe the transcript this shape fixes.
    ///
    /// # Panics
    ///
    /// Never in practice.
    ///
    /// Every container opened below is closed in the same call.
    #[must_use]
    pub fn pattern<F, EF>(&self) -> InteractionPattern
    where
        F: PrimeField64,
        EF: ExtensionField<F>,
    {
        let mut steps = Vec::new();

        for commitment in &self.claim_widths {
            // One commitment is one block.
            //
            // Its matrix count cannot be reshuffled.
            open(&mut steps, COMMITMENT_BLOCK);
            for matrix in commitment {
                // One matrix is one block too.
                //
                // Its opening count cannot be reshuffled either.
                open(&mut steps, MATRIX_BLOCK);
                for &width in matrix {
                    // One opening is one step declaring that matrix's column count.
                    //
                    // The width belongs to the description.
                    //
                    // It does not come from the values as they arrive.
                    steps.push(Interaction::algebra::<F, EF>(
                        Hierarchy::Atomic,
                        Kind::Message,
                        CLAIMED_VALUES,
                        Length::Fixed(width),
                    ));
                }
                close(&mut steps, MATRIX_BLOCK);
            }
            close(&mut steps, COMMITMENT_BLOCK);
        }

        InteractionPattern::new(steps).expect("every container opened here is closed here")
    }

    /// Bind the protocol identity and this shape into a seed.
    ///
    /// Every number of this phase shapes the step sequence.
    ///
    /// The fingerprint of that sequence covers all of them.
    #[must_use]
    pub fn domain_separator<F, EF>(&self) -> DomainSeparator<Alphabet<F>>
    where
        F: PrimeField64,
        EF: ExtensionField<F>,
    {
        DomainSeparator::new(VERSION, CLAIM_NAME, self.pattern::<F, EF>())
    }
}

/// Bind one batch of claimed evaluations, prover side.
///
/// # Overview
///
/// The description is derived from the very values being bound.
///
/// So the widths and the grouping reach the seed.
///
/// No disagreement between the two is representable.
///
/// # Arguments
///
/// - `challenger`: sponge of the surrounding protocol, borrowed for the phase.
/// - `opened_values`: the claimed evaluations, nested by commitment, matrix and point.
///
/// # Panics
///
/// Never in practice.
///
/// Every described step is played before the driver is closed.
pub(crate) fn observe_opened_values<Ch, F, EF>(
    challenger: &mut Ch,
    opened_values: &OpenedValues<EF>,
) where
    F: PrimeField64,
    EF: ExtensionField<F>,
    Ch: CanObserve<F> + CanSample<F>,
{
    // Seeding folds the claim shape into the sponge before any value is absorbed.
    let shape = StirPcsClaimShape::from_opened_values(opened_values);
    let separator = shape.domain_separator::<F, EF>();
    let mut state: ProverState<&mut Ch, Alphabet<F>> = ProverState::new(challenger, &separator);

    for commitment in opened_values {
        // Open the commitment's block before any of its matrices.
        state.begin_protocol::<Block>(COMMITMENT_BLOCK);
        for matrix in commitment {
            // Open the matrix's block before any of its openings.
            state.begin_protocol::<Block>(MATRIX_BLOCK);
            for values in matrix {
                // One opening is one step carrying that matrix's whole row of claims.
                let _bound =
                    state.observe_extensions::<F, EF, FieldToFieldCodec<F>>(CLAIMED_VALUES, values);
            }
            state.end_protocol::<Block>(MATRIX_BLOCK);
        }
        state.end_protocol::<Block>(COMMITMENT_BLOCK);
    }

    // The claims travel in the caller's own proof.
    //
    // The driver buffered nothing.
    assert!(
        state.finalize().is_empty(),
        "the commitment scheme carries every claim in its own proof",
    );
}

/// Bind one batch of opening claims, verifier side.
///
/// # Overview
///
/// Mirrors the prover side value for value, over the same description.
///
/// The claims are the statement this call was handed, not anything read out of a proof.
///
/// So the description derived from them agrees with them by construction.
///
/// # Arguments
///
/// - `challenger`: sponge of the surrounding protocol, borrowed for the phase.
/// - `claims`: one entry per commitment, holding the points and values of each matrix.
///
/// # Panics
///
/// Never in practice.
///
/// Every described step is played before the driver is closed.
pub(crate) fn observe_claims<Ch, F, EF, Com, Domain>(
    challenger: &mut Ch,
    claims: &[CommitmentOpening<EF, Com, Domain>],
) where
    F: PrimeField64,
    EF: ExtensionField<F>,
    Ch: CanObserve<F> + CanSample<F>,
{
    // Seeding folds the claim shape into the sponge before any value is absorbed.
    let shape = StirPcsClaimShape::from_claims(claims);
    let separator = shape.domain_separator::<F, EF>();
    let mut state: VerifierState<'static, &mut Ch, Alphabet<F>> =
        VerifierState::new(challenger, &separator, &[]);

    for claim in claims {
        // Open the commitment's block before any of its matrices.
        state.begin_protocol::<Block>(COMMITMENT_BLOCK);
        for MatrixOpening { points, .. } in &claim.matrices {
            // Open the matrix's block before any of its openings.
            state.begin_protocol::<Block>(MATRIX_BLOCK);
            for PointOpening { values, .. } in points {
                // The step's own width came from this slice.
                //
                // The count check cannot fire.
                let _bound =
                    state.observe_extensions::<F, EF, FieldToFieldCodec<F>>(CLAIMED_VALUES, values);
            }
            state.end_protocol::<Block>(MATRIX_BLOCK);
        }
        state.end_protocol::<Block>(COMMITMENT_BLOCK);
    }

    // The claims are held by the caller.
    //
    // No wire bytes can remain unread.
    state
        .finalize()
        .expect("the commitment scheme reads an empty wire");
}

/// Numbers that fix the transcript of one shared-domain bucket.
///
/// Both sides build this from their own configuration, never from a proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StirPcsBucketShape {
    /// Log of the shared evaluation domain this bucket's instance runs on.
    ///
    /// Reaches the seed through the instance label.
    pub log_lde_height: usize,
    /// Log of every native height merged into this bucket's codeword, descending.
    ///
    /// A single entry means nothing is merged.
    ///
    /// No merging challenge is then drawn.
    ///
    /// Reaches the seed through the instance label.
    pub log_native_heights: Vec<usize>,
    /// Log of the arity of the first fold.
    ///
    /// This is the bit width of one lane index.
    pub log_first_fold_arity: usize,
    /// Number of first-round query draws.
    ///
    /// One lane is drawn per query draw.
    pub num_query_draws: usize,
}

impl StirPcsBucketShape {
    /// Derive one bucket's shape from its configuration.
    ///
    /// # Arguments
    ///
    /// - `log_lde_height`: log of the shared evaluation domain the bucket runs on.
    /// - `log_native_heights`: log of every native height the bucket merges, descending.
    /// - `config`: the schedule the bucket's proximity test was solved for.
    #[must_use]
    pub fn new<F, EF, M, C>(
        log_lde_height: usize,
        log_native_heights: Vec<usize>,
        config: &StirConfig<F, EF, M, C>,
    ) -> Self
    where
        F: TwoAdicField + PrimeField64,
        EF: ExtensionField<F>,
        M: Mmcs<EF>,
        C: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        Self {
            log_lde_height,
            log_native_heights,
            // A fiber of the first fold domain holds one point per lane.
            //
            // So a lane index spans exactly the arity's worth of bits.
            log_first_fold_arity: config.log_starting_folding_factor,
            // One lane is drawn per first-round query draw, repeats included.
            num_query_draws: first_round_draw_count(config),
        }
    }

    /// Whether this bucket merges more than one native height.
    ///
    /// A bucket of a single height needs no merging challenge.
    #[must_use]
    pub const fn merges_classes(&self) -> bool {
        self.log_native_heights.len() > 1
    }

    /// Every number of this bucket, packed for the instance label.
    ///
    /// Each value is eight big-endian bytes.
    ///
    /// The packing is therefore self-delimiting.
    fn label_bytes(&self) -> Vec<u8> {
        // Two bucket-wide values, then one per merged height.
        let mut out = Vec::with_capacity(8 * (2 + self.log_native_heights.len()));
        push_u64(&mut out, self.log_lde_height as u64);
        push_u64(&mut out, self.log_native_heights.len() as u64);
        for &log_native_height in &self.log_native_heights {
            push_u64(&mut out, log_native_height as u64);
        }
        out
    }
}

/// Numbers that fix the transcript of one opening argument's bucket phase.
///
/// Both sides build this from their own configuration, never from a proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StirPcsOpeningShape {
    /// One entry per bucket, in the order the buckets are played.
    pub buckets: Vec<StirPcsBucketShape>,
}

impl StirPcsOpeningShape {
    /// Collect the buckets one opening argument plays.
    ///
    /// # Arguments
    ///
    /// - `buckets`: one entry per bucket, in play order.
    #[must_use]
    pub const fn new(buckets: Vec<StirPcsBucketShape>) -> Self {
        Self { buckets }
    }

    /// Number of buckets this argument plays.
    #[must_use]
    pub const fn num_buckets(&self) -> usize {
        self.buckets.len()
    }

    /// Describe the transcript this shape fixes.
    ///
    /// # Panics
    ///
    /// Never in practice.
    ///
    /// Every container opened below is closed in the same call.
    #[must_use]
    pub fn pattern<F, EF>(&self) -> InteractionPattern
    where
        F: PrimeField64,
        EF: ExtensionField<F>,
    {
        // Three steps per merging block, three per lane block, two for the bracket.
        let mut steps = Vec::with_capacity(6 * self.buckets.len() + 2);

        for bucket in &self.buckets {
            // Every bucket opens a merging block, whether or not it draws inside it.
            //
            // The block says where one bucket's merging stops.
            open(&mut steps, COMBINE_BLOCK);
            if bucket.merges_classes() {
                // One challenge merges the bucket's classes into one codeword.
                steps.push(Interaction::algebra::<F, EF>(
                    Hierarchy::Atomic,
                    Kind::Challenge,
                    COMBINATION_CHALLENGE,
                    Length::Scalar,
                ));
            }
            close(&mut steps, COMBINE_BLOCK);
        }

        // The bracket records that a sub-protocol runs here.
        //
        // Its steps live in the callee's own description.
        //
        // This description states only that the delegation happens, and where.
        steps.push(Interaction::marker::<ProximityTest>(
            Hierarchy::Begin,
            Kind::Protocol,
            PROXIMITY_TEST,
        ));
        steps.push(Interaction::marker::<ProximityTest>(
            Hierarchy::End,
            Kind::Protocol,
            PROXIMITY_TEST,
        ));

        for bucket in &self.buckets {
            // Lanes are drawn only once the whole proximity test is in the sponge.
            open(&mut steps, LANE_BLOCK);
            // Lane indices are drawn without modular bias.
            //
            // The tag records that.
            steps.push(Interaction::uniform_bits(
                Hierarchy::Atomic,
                Kind::Challenge,
                FIBER_LANES,
                bucket.log_first_fold_arity,
                Length::Fixed(bucket.num_query_draws),
            ));
            close(&mut steps, LANE_BLOCK);
        }

        InteractionPattern::new(steps).expect("every container opened here is closed here")
    }

    /// Bind the protocol identity, this shape, and the remaining parameters.
    ///
    /// A number that changes the step sequence is covered by the fingerprint.
    ///
    /// The rest go in the instance label.
    #[must_use]
    pub fn domain_separator<F, EF>(&self) -> DomainSeparator<Alphabet<F>>
    where
        F: PrimeField64,
        EF: ExtensionField<F>,
    {
        let mut separator = DomainSeparator::new(VERSION, OPENING_NAME, self.pattern::<F, EF>());

        // Header: how many buckets share the argument.
        let mut header = Vec::with_capacity(8);
        push_u64(&mut header, self.buckets.len() as u64);
        separator.instance(&header);

        // One delimited chunk per bucket.
        //
        // Two bucketings never collapse into one.
        for bucket in &self.buckets {
            separator.instance(&bucket.label_bytes());
        }

        separator
    }
}

/// Prover-side transcript of one opening argument's bucket phase.
///
/// # Overview
///
/// Holds the only definition of what a prover draws around the proximity test.
///
/// Three things happen here, in this order.
///
/// - Each bucket merges its native-height classes into one codeword.
/// - The proximity test runs on every bucket, inside a bracket.
/// - Each bucket picks one fiber lane per first-round query draw.
///
/// # Why the lanes come last
///
/// A lane check reads the input rows at one position inside a queried fiber.
///
/// A prover who learns that position first can place its disagreement elsewhere.
///
/// No lane would then ever land on it.
///
/// So every lane is drawn once the whole proximity test is already in the sponge.
///
/// # Borrowing
///
/// The challenger is borrowed, not consumed.
///
/// The commitment scheme runs inside a larger protocol.
///
/// That protocol's own transcript continues where this one stops.
pub struct OpeningProverTranscript<'a, C, F: PrimeField64, EF> {
    /// Driver walking the description and holding the borrowed sponge.
    state: ProverState<&'a mut C, Alphabet<F>>,
    /// The numbers this run was described with.
    shape: StirPcsOpeningShape,
    /// Marker for the extension field the challenges live in.
    _ef: PhantomData<EF>,
}

impl<'a, C, F, EF> OpeningProverTranscript<'a, C, F, EF>
where
    F: PrimeField64,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F> + CanSampleUniformBits<F>,
{
    /// Seed the transcript from the shape.
    ///
    /// # Arguments
    ///
    /// - `challenger`: sponge of the surrounding protocol, borrowed for the phase.
    /// - `shape`: the numbers that fix this phase's transcript.
    pub fn new(challenger: &'a mut C, shape: StirPcsOpeningShape) -> Self {
        // Seeding folds the shape fingerprint into the sponge before any step.
        let separator = shape.domain_separator::<F, EF>();
        Self {
            state: ProverState::new(challenger, &separator),
            shape,
            _ef: PhantomData,
        }
    }

    /// Play one bucket's merging block.
    ///
    /// # Returns
    ///
    /// The merging challenge, or nothing when the bucket holds a single native height.
    ///
    /// # Panics
    ///
    /// When the bucket index lies outside the described run.
    pub fn combination_challenge(&mut self, bucket: usize) -> Option<EF> {
        let merges = self.shape.buckets[bucket].merges_classes();
        // The block opens either way.
        //
        // It marks this bucket's slot in the sequence.
        self.state.begin_protocol::<Block>(COMBINE_BLOCK);
        let challenge = merges.then(|| {
            self.state
                .challenge_extension::<F, EF, FieldToFieldCodec<F>>(COMBINATION_CHALLENGE)
                .into_inner()
        });
        self.state.end_protocol::<Block>(COMBINE_BLOCK);
        challenge
    }

    /// Lend the sponge to the proximity test, bracketed as a sub-protocol.
    ///
    /// The callee seeds its own driver from the state this one has reached.
    ///
    /// # Returns
    ///
    /// Whatever the delegated run produced.
    pub fn delegate<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        self.state.begin_protocol::<ProximityTest>(PROXIMITY_TEST);
        let output = run(self.state.challenger_mut());
        self.state.end_protocol::<ProximityTest>(PROXIMITY_TEST);
        output
    }

    /// Draw one bucket's fiber lanes, one per first-round query draw.
    ///
    /// # Returns
    ///
    /// Every lane index, in draw order, repeats included.
    ///
    /// # Panics
    ///
    /// When the bucket index lies outside the described run.
    pub fn lanes(&mut self, bucket: usize) -> Vec<usize> {
        let shape = &self.shape.buckets[bucket];
        let (width, count) = (shape.log_first_fold_arity, shape.num_query_draws);
        self.state.begin_protocol::<Block>(LANE_BLOCK);
        let lanes = self
            .state
            .challenge_uniform_bits::<F>(FIBER_LANES, width, count)
            .into_iter()
            .map(TranscriptBound::into_inner)
            .collect();
        self.state.end_protocol::<Block>(LANE_BLOCK);
        lanes
    }

    /// Close the transcript once every described step has been played.
    ///
    /// # Panics
    ///
    /// When the run played fewer steps than it was described with.
    pub fn finish(self) {
        // Every value of this phase is a challenge.
        //
        // The driver buffered nothing.
        assert!(
            self.state.finalize().is_empty(),
            "the bucket phase draws challenges and sends nothing",
        );
    }
}

/// Verifier-side transcript of one opening argument's bucket phase.
///
/// Mirrors the prover side call for call, over the same description.
///
/// Every value of this phase is drawn from the sponge.
///
/// No step of it can be rejected.
pub struct OpeningVerifierTranscript<'a, C, F: PrimeField64, EF> {
    /// Driver walking the description and holding the borrowed sponge.
    ///
    /// The phase sends nothing.
    ///
    /// The driver reads an empty wire.
    state: VerifierState<'static, &'a mut C, Alphabet<F>>,
    /// The numbers this run was described with.
    shape: StirPcsOpeningShape,
    /// Marker for the extension field the challenges live in.
    _ef: PhantomData<EF>,
}

impl<'a, C, F, EF> OpeningVerifierTranscript<'a, C, F, EF>
where
    F: PrimeField64,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F> + CanSampleUniformBits<F>,
{
    /// Seed the transcript from the shape.
    ///
    /// The arguments match the prover's.
    ///
    /// Both sides seed identically.
    pub fn new(challenger: &'a mut C, shape: StirPcsOpeningShape) -> Self {
        // Seeding folds the shape fingerprint into the sponge before any step.
        let separator = shape.domain_separator::<F, EF>();
        Self {
            state: VerifierState::new(challenger, &separator, &[]),
            shape,
            _ef: PhantomData,
        }
    }

    /// Replay one bucket's merging block.
    ///
    /// # Returns
    ///
    /// The merging challenge, or nothing when the bucket holds a single native height.
    ///
    /// # Panics
    ///
    /// When the bucket index lies outside the described run.
    pub fn combination_challenge(&mut self, bucket: usize) -> Option<EF> {
        let merges = self.shape.buckets[bucket].merges_classes();
        // The block opens either way.
        //
        // It marks this bucket's slot in the sequence.
        self.state.begin_protocol::<Block>(COMBINE_BLOCK);
        let challenge = merges.then(|| {
            self.state
                .challenge_extension::<F, EF, FieldToFieldCodec<F>>(COMBINATION_CHALLENGE)
                .into_inner()
        });
        self.state.end_protocol::<Block>(COMBINE_BLOCK);
        challenge
    }

    /// Lend the sponge to the proximity test, bracketed as a sub-protocol.
    ///
    /// The bracket closes whatever the delegated run returned.
    ///
    /// A rejection therefore leaves this transcript replayable to the end.
    ///
    /// # Returns
    ///
    /// Whatever the delegated run produced.
    pub fn delegate<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        self.state.begin_protocol::<ProximityTest>(PROXIMITY_TEST);
        let output = run(self.state.challenger_mut());
        self.state.end_protocol::<ProximityTest>(PROXIMITY_TEST);
        output
    }

    /// Redraw one bucket's fiber lanes, one per first-round query draw.
    ///
    /// # Returns
    ///
    /// Every lane index, in draw order, repeats included.
    ///
    /// # Panics
    ///
    /// When the bucket index lies outside the described run.
    pub fn lanes(&mut self, bucket: usize) -> Vec<usize> {
        let shape = &self.shape.buckets[bucket];
        let (width, count) = (shape.log_first_fold_arity, shape.num_query_draws);
        self.state.begin_protocol::<Block>(LANE_BLOCK);
        let lanes = self
            .state
            .challenge_uniform_bits::<F>(FIBER_LANES, width, count)
            .into_iter()
            .map(TranscriptBound::into_inner)
            .collect();
        self.state.end_protocol::<Block>(LANE_BLOCK);
        lanes
    }

    /// Release the completeness check because the proof is being rejected.
    ///
    /// Every path that leaves the transcript early goes through this.
    ///
    /// Dropping an unfinished driver otherwise panics.
    ///
    /// That panic would land on top of an error already travelling to the caller.
    pub fn abort(&mut self) {
        self.state.abort();
    }

    /// Close the transcript once every described step has been replayed.
    ///
    /// # Panics
    ///
    /// When the run replayed fewer steps than it was described with.
    pub fn finish(self) {
        self.state
            .finalize()
            .expect("the bucket phase reads an empty wire, so no bytes can remain");
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::testing::{SeedDigest, assert_seeds_pairwise_distinct, seed_digest};
    use p3_challenger::{CanSampleBits, DuplexChallenger};
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type Ch = DuplexChallenger<F, Perm, 16, 8>;

    /// One commitment's claims, shaped the way a caller hands them over.
    type Claim = CommitmentOpening<EF, (), ()>;

    fn fresh_challenger() -> Ch {
        // Fixed seed so two runs differ only where the transcript makes them differ.
        let mut rng = SmallRng::seed_from_u64(0x5717_09C5);
        Ch::new(Perm::new_from_rng_128(&mut rng))
    }

    /// A verifier-side claim tree matching `widths`, with every claimed value set to one.
    fn claims_of(widths: &[Vec<Vec<usize>>]) -> Vec<Claim> {
        widths
            .iter()
            .map(|commitment| CommitmentOpening {
                commitment: (),
                matrices: commitment
                    .iter()
                    .map(|matrix| MatrixOpening {
                        domain: (),
                        // Each opening point carries as many values as its width says.
                        points: matrix
                            .iter()
                            .map(|&width| PointOpening {
                                point: EF::ONE,
                                values: vec![EF::ONE; width],
                            })
                            .collect(),
                    })
                    .collect(),
            })
            .collect()
    }

    /// A prover-side value tree matching `widths`, with every claimed value set to one.
    fn opened_values_of(widths: &[Vec<Vec<usize>>]) -> OpenedValues<EF> {
        widths
            .iter()
            .map(|commitment| {
                commitment
                    .iter()
                    .map(|matrix| matrix.iter().map(|&w| vec![EF::ONE; w]).collect())
                    .collect()
            })
            .collect()
    }

    /// The digest of the byte stream a claim grouping seeds its sponge with.
    fn claim_seed(widths: Vec<Vec<Vec<usize>>>) -> SeedDigest {
        let shape = StirPcsClaimShape {
            claim_widths: widths,
        };
        seed_digest(&shape.domain_separator::<F, EF>())
    }

    /// One bucket over the given merged heights, with fixed lane geometry.
    fn bucket(log_native_heights: Vec<usize>) -> StirPcsBucketShape {
        StirPcsBucketShape {
            log_lde_height: 10,
            log_native_heights,
            log_first_fold_arity: 2,
            num_query_draws: 3,
        }
    }

    /// The digest of the byte stream a bucket-phase shape seeds its sponge with.
    fn opening_seed(shape: &StirPcsOpeningShape) -> SeedDigest {
        seed_digest(&shape.domain_separator::<F, EF>())
    }

    #[test]
    fn the_root_count_of_a_commitment_reaches_its_seed() {
        // Invariant: the root count is the step count.
        //
        // It cannot move without the seed moving too.
        //
        // Fixture state: three absorptions, of one, two and three roots.
        //
        //     1 root  -> a description of 1 step
        //     2 roots -> a description of 2 steps
        //     3 roots -> a description of 3 steps
        let seeds: Vec<(usize, SeedDigest)> = (1..=3)
            .map(|num_roots| {
                let shape = StirPcsCommitmentShape::new(num_roots);
                (num_roots, seed_digest(&shape.domain_separator::<F>()))
            })
            .collect();

        assert_seeds_pairwise_distinct(&seeds);
    }

    #[test]
    fn splitting_a_commitment_in_two_moves_the_sponge() {
        // Invariant: two roots absorbed as one commitment differ from two absorbed apart.
        //
        // Fixture state: roots 7 and 9, absorbed once together and once separately.
        //
        // Mutation: split the two-root commitment into two one-root commitments.
        //
        //     merged: seed(2 steps), 7, 9
        //     split : seed(1 step), 7, then seed(1 step), 9
        let (root_a, root_b) = (F::from_u64(7), F::from_u64(9));

        let mut merged = fresh_challenger();
        observe_commitment::<_, F, _>(&mut merged, vec![root_a, root_b]);

        let mut split = fresh_challenger();
        observe_commitment::<_, F, _>(&mut split, vec![root_a]);
        observe_commitment::<_, F, _>(&mut split, vec![root_b]);

        assert_ne!(merged.sample_bits(24), split.sample_bits(24));
    }

    #[test]
    fn the_root_order_of_a_commitment_reaches_the_sponge() {
        // Invariant: roots are absorbed in sequence, never as an order-insensitive set.
        //
        // Fixture state: the same two roots, in the two possible orders.
        //
        // Mutation: swap the two roots.
        //
        //     forward : 7, 9
        //     reversed: 9, 7
        let (root_a, root_b) = (F::from_u64(7), F::from_u64(9));

        let mut forward = fresh_challenger();
        observe_commitment::<_, F, _>(&mut forward, vec![root_a, root_b]);

        let mut reversed = fresh_challenger();
        observe_commitment::<_, F, _>(&mut reversed, vec![root_b, root_a]);

        assert_ne!(forward.sample_bits(24), reversed.sample_bits(24));
    }

    #[test]
    fn a_root_that_moves_moves_the_sponge() {
        // Invariant: every root is absorbed.
        //
        // Perturbing one parts the two sponges.
        //
        // Fixture state: a commitment of three roots, 7, 9 and 11.
        //
        // Mutation: bump the middle root by one.
        //
        //     bound:     7,  9, 11
        //     perturbed: 7, 10, 11
        let roots = vec![F::from_u64(7), F::from_u64(9), F::from_u64(11)];

        let mut bound = fresh_challenger();
        observe_commitment::<_, F, _>(&mut bound, roots.clone());

        let mut perturbed_roots = roots;
        perturbed_roots[1] += F::ONE;
        let mut perturbed = fresh_challenger();
        observe_commitment::<_, F, _>(&mut perturbed, perturbed_roots);

        assert_ne!(bound.sample_bits(24), perturbed.sample_bits(24));
    }

    #[test]
    fn no_two_claim_groupings_share_a_seed() {
        // Invariant: every knob of the claim description is pairwise separated.
        //
        // Fixture state: one commitment, one matrix, two openings of widths 3 and 1.
        //
        // Mutation: one step away from that baseline, six different ways.
        //
        //     baseline         [[[3, 1]]]
        //     wider            [[[4, 1]]]        one more value in the first opening
        //     longer           [[[3, 1, 2]]]     one more opening, so one more step
        //     reordered        [[[1, 3]]]        the same widths, absorbed the other way
        //     two matrices     [[[3], [1]]]      the same widths, split across two blocks
        //     two commitments  [[[3]], [[1]]]    the same widths, split across two blocks
        let seeds = [
            ("baseline", claim_seed(vec![vec![vec![3, 1]]])),
            ("wider", claim_seed(vec![vec![vec![4, 1]]])),
            ("longer", claim_seed(vec![vec![vec![3, 1, 2]]])),
            ("reordered", claim_seed(vec![vec![vec![1, 3]]])),
            ("two matrices", claim_seed(vec![vec![vec![3], vec![1]]])),
            (
                "two commitments",
                claim_seed(vec![vec![vec![3]], vec![vec![1]]]),
            ),
        ];

        assert_seeds_pairwise_distinct(&seeds);
    }

    #[test]
    fn the_claim_grouping_reaches_the_pattern_itself() {
        // Invariant: the containers separate two groupings.
        //
        // No separate label does that work.
        //
        // Fixture state: three groupings whose widths all flatten to `3, 3`.
        //
        //     one matrix, two points   Begin c  Begin m  3  3  End  End
        //     two matrices, one point  Begin c  Begin m  3  End  Begin m  3  End  End
        //     two commitments          Begin c  Begin m  3  End  End  Begin c  ...
        //
        // Fingerprints are compared here rather than seeds.
        //
        // A seed would also move if a label carried the grouping instead.
        let two_points = StirPcsClaimShape {
            claim_widths: vec![vec![vec![3, 3]]],
        };
        let two_matrices = StirPcsClaimShape {
            claim_widths: vec![vec![vec![3], vec![3]]],
        };
        let two_commitments = StirPcsClaimShape {
            claim_widths: vec![vec![vec![3]], vec![vec![3]]],
        };

        let hashes = [
            two_points.pattern::<F, EF>().pattern_hash(),
            two_matrices.pattern::<F, EF>().pattern_hash(),
            two_commitments.pattern::<F, EF>().pattern_hash(),
        ];

        // Mutation: strip the containers.
        //
        // The three then collapse onto one description.
        let flattened: Vec<[u8; 32]> = [&two_points, &two_matrices, &two_commitments]
            .iter()
            .map(|shape| {
                let leaves: Vec<Interaction> = shape
                    .pattern::<F, EF>()
                    .interactions()
                    .iter()
                    .copied()
                    .filter(|step| step.hierarchy() == Hierarchy::Atomic)
                    .collect();
                InteractionPattern::new(leaves)
                    .expect("leaves alone are well formed")
                    .pattern_hash()
            })
            .collect();

        // Nested: all three descriptions differ.
        assert_ne!(hashes[0], hashes[1]);
        assert_ne!(hashes[0], hashes[2]);
        assert_ne!(hashes[1], hashes[2]);
        // Flat: all three descriptions coincide.
        //
        // That is what the containers rule out.
        assert_eq!(flattened[0], flattened[1]);
        assert_eq!(flattened[0], flattened[2]);
    }

    #[test]
    fn a_claimed_value_that_moves_moves_the_sponge() {
        // Invariant: every claimed value is absorbed.
        //
        // Perturbing one parts the two sponges.
        //
        // Fixture state: two commitments of widths [[3, 1], [2]] and [[4]].
        //
        // Mutation: bump one claimed value of the second commitment's only opening.
        //
        //     bound:     ..., [1, 1, 1, 1]
        //     perturbed: ..., [1, 1, 2, 1]
        let widths = vec![vec![vec![3, 1], vec![2]], vec![vec![4]]];
        let values = opened_values_of(&widths);

        let mut bound = fresh_challenger();
        observe_opened_values::<_, F, EF>(&mut bound, &values);

        let mut perturbed_values = values;
        perturbed_values[1][0][0][2] += EF::ONE;
        let mut perturbed = fresh_challenger();
        observe_opened_values::<_, F, EF>(&mut perturbed, &perturbed_values);

        assert_ne!(bound.sample_bits(24), perturbed.sample_bits(24));
    }

    #[test]
    fn both_sides_absorb_one_batch_of_claims_identically() {
        // Invariant: each side drives the claim phase from its own input shape.
        //
        // Both land on the same sponge state.
        //
        // Fixture state: two commitments of widths [[3, 1], [2]] and [[4]].
        //
        //     prover  : walks the value tree it computed
        //     verifier: walks the claim tree it was handed
        let widths = vec![vec![vec![3, 1], vec![2]], vec![vec![4]]];

        let mut prover = fresh_challenger();
        observe_opened_values::<_, F, EF>(&mut prover, &opened_values_of(&widths));

        let mut verifier = fresh_challenger();
        observe_claims::<_, F, EF, (), ()>(&mut verifier, &claims_of(&widths));

        assert_eq!(prover.sample_bits(24), verifier.sample_bits(24));

        // Both sides derive the same description from their own tree.
        assert_eq!(
            StirPcsClaimShape::from_opened_values(&opened_values_of(&widths)),
            StirPcsClaimShape::from_claims(&claims_of(&widths)),
        );
    }

    #[test]
    fn no_two_configurations_of_the_bucket_shape_share_a_seed() {
        // Invariant: every knob of the bucket-phase shape reaches the seed, pairwise.
        //
        // Fixture state: one bucket on the domain of size 2^10.
        //
        //     merged heights   2^8 and 2^6
        //     lane draws       3 indices of 2 bits each
        //
        // Mutation: one field moved by one step, seven different ways.
        let baseline = StirPcsOpeningShape::new(vec![bucket(vec![8, 6])]);

        let mut wider_domain = baseline.clone();
        wider_domain.buckets[0].log_lde_height += 1;

        let mut one_class = baseline.clone();
        one_class.buckets[0].log_native_heights = vec![8];

        let mut other_classes = baseline.clone();
        other_classes.buckets[0].log_native_heights = vec![8, 5];

        let mut three_classes = baseline.clone();
        three_classes.buckets[0].log_native_heights = vec![8, 6, 4];

        let mut wider_lane = baseline.clone();
        wider_lane.buckets[0].log_first_fold_arity += 1;

        let mut more_draws = baseline.clone();
        more_draws.buckets[0].num_query_draws += 1;

        let mut two_buckets = baseline.clone();
        two_buckets.buckets.push(bucket(vec![8, 6]));

        let seeds = [
            ("baseline", opening_seed(&baseline)),
            ("log_lde_height", opening_seed(&wider_domain)),
            ("one merged class", opening_seed(&one_class)),
            ("other merged classes", opening_seed(&other_classes)),
            ("three merged classes", opening_seed(&three_classes)),
            ("log_first_fold_arity", opening_seed(&wider_lane)),
            ("num_query_draws", opening_seed(&more_draws)),
            ("bucket count", opening_seed(&two_buckets)),
        ];

        assert_seeds_pairwise_distinct(&seeds);
    }

    #[test]
    fn a_single_class_bucket_draws_no_merging_challenge() {
        // Invariant: the merging block is present either way.
        //
        // It is empty for a bucket of one class.
        //
        // Fixture state: two buckets, the first merging two heights and the second one.
        //
        //     bucket 0   Begin combine  challenge  End
        //     bucket 1   Begin combine             End
        let shape = StirPcsOpeningShape::new(vec![bucket(vec![8, 6]), bucket(vec![8])]);

        let mut challenger = fresh_challenger();
        let mut transcript =
            OpeningProverTranscript::<Ch, F, EF>::new(&mut challenger, shape.clone());

        // The merging bucket yields a challenge.
        //
        // The single-class one yields nothing.
        assert!(transcript.combination_challenge(0).is_some());
        assert!(transcript.combination_challenge(1).is_none());
        transcript.delegate(|_| ());
        let _lanes_0 = transcript.lanes(0);
        let _lanes_1 = transcript.lanes(1);
        transcript.finish();

        // Mutation: let the second bucket merge too.
        //
        // That adds a step.
        //
        // The seed moves with it.
        let both_merge = StirPcsOpeningShape::new(vec![bucket(vec![8, 6]), bucket(vec![8, 4])]);
        assert_ne!(opening_seed(&shape), opening_seed(&both_merge));
    }

    #[test]
    fn both_sides_draw_the_same_bucket_challenges() {
        // Invariant: the two sides walk the bucket phase in lockstep, bracket included.
        //
        // They leave the sponge in one state.
        //
        // Fixture state: two buckets, the first merging two heights and the second one.
        let shape = StirPcsOpeningShape::new(vec![bucket(vec![8, 6]), bucket(vec![7])]);

        // Prover side: merge, delegate, then draw the lanes of both buckets.
        let mut prover_challenger = fresh_challenger();
        let mut prover =
            OpeningProverTranscript::<Ch, F, EF>::new(&mut prover_challenger, shape.clone());
        let prover_r_comb = prover.combination_challenge(0);
        assert!(prover.combination_challenge(1).is_none());
        prover.delegate(|_| ());
        let prover_lanes: Vec<Vec<usize>> = (0..2).map(|b| prover.lanes(b)).collect();
        prover.finish();

        // Verifier side: the same calls, in the same order.
        let mut verifier_challenger = fresh_challenger();
        let mut verifier =
            OpeningVerifierTranscript::<Ch, F, EF>::new(&mut verifier_challenger, shape);
        let verifier_r_comb = verifier.combination_challenge(0);
        assert!(verifier.combination_challenge(1).is_none());
        verifier.delegate(|_| ());
        let verifier_lanes: Vec<Vec<usize>> = (0..2).map(|b| verifier.lanes(b)).collect();
        verifier.finish();

        // Every drawn value agrees across the two sides.
        assert_eq!(prover_r_comb, verifier_r_comb);
        assert_eq!(prover_lanes, verifier_lanes);

        // Lane widths follow the described arity.
        //
        // Each index stays inside its fiber.
        assert!(prover_lanes.iter().flatten().all(|&lane| lane < 4));

        // Both sponges land on the same state.
        let prover_next: F = prover_challenger.sample();
        let verifier_next: F = verifier_challenger.sample();
        assert_eq!(prover_next, verifier_next);
    }

    #[test]
    fn a_rejected_delegation_leaves_the_bucket_transcript_droppable() {
        // Invariant: a sub-protocol that rejects must not turn into a drop-time panic.
        //
        // Fixture state: one bucket, whose delegation returns a rejection.
        //
        //     combine  ->  delegate (rejects)  ->  abort  ->  drop
        //
        // Without the release, dropping a half-played driver panics on top of the error.
        let shape = StirPcsOpeningShape::new(vec![bucket(vec![8, 6])]);

        let mut challenger = fresh_challenger();
        let mut transcript = OpeningVerifierTranscript::<Ch, F, EF>::new(&mut challenger, shape);
        let _r_comb = transcript.combination_challenge(0);

        // Mutation: the delegated run reports a failure instead of an output.
        let outcome: Result<(), ()> = transcript.delegate(|_| Err(()));
        assert!(outcome.is_err());

        // Releasing the completeness check is what keeps the rejection the only failure.
        transcript.abort();
        drop(transcript);
    }
}
