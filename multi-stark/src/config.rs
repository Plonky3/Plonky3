//! Configuration tying together the commitment scheme, challenge field, and transcript.

use alloc::vec::Vec;

use p3_challenger::fs::TranscriptField;
use p3_commit::MultilinearPcs;
use p3_field::ExtensionField;
use p3_sumcheck::layout::Table;

pub use crate::rounds::sliced::MAX_SLICED_ROUNDS;

/// Zerocheck rounds a stage evaluates on its bit-sliced planes by default.
///
/// A sliced round evaluates the AIR once per sixty-four residual rows instead of once per row,
/// so each extra round doubles the plane work while halving the dense tail that follows it.
/// Where the crossover falls depends on how much constraint work the AIR does per column, not
/// on the trace height, so no single count suits every AIR.
///
/// Three is the count that does not lose on either of the two hash AIRs measured.
pub const DEFAULT_SLICED_ROUNDS: usize = 3;

/// The wiring a multilinear AIR proof depends on.
///
/// One implementation fixes a commitment scheme, a challenge field, and a transcript type.
///
/// The prover and verifier are generic over any implementation.
pub trait MultiStarkConfig {
    /// Base field the trace and committed columns live in.
    ///
    /// The transcript requires an injective byte encoding for its domain separator.
    /// Both supported prime fields and binary tower fields provide one.
    type Val: TranscriptField;

    /// Extension field that challenges are drawn from.
    ///
    /// Its cardinality must cover the complete reduction's security budget.
    /// `ExtensionField` also admits the base field itself; that is appropriate
    /// for sufficiently large fields, but does not enforce a security minimum.
    /// Use [`crate::verify_with_security`] to enforce an explicit target.
    type Challenge: ExtensionField<Self::Val>;

    /// Fiat-Shamir transcript used to derive challenges and absorb commitments.
    type Challenger;

    /// Multilinear commitment scheme over the base field.
    type Pcs: MultilinearPcs<Self::Challenge, Self::Challenger, Val = Self::Val>;

    /// Borrow the commitment scheme for the main trace.
    fn pcs(&self) -> &Self::Pcs;

    /// Trusted collision-security cap for the transcript and all commitment hashes.
    ///
    /// Return the minimum supported by these primitives and their parameters.
    /// This is separate from the PCS's algebraic soundness assessment. The
    /// default supplies no evidence, so security-checked entry points fail closed.
    fn collision_resistance_bits(&self) -> Option<usize> {
        None
    }

    /// Zerocheck rounds a stage may evaluate on its bit-sliced planes.
    ///
    /// Only the sliced kernel reads this, and only for a stage whose cells all fit its
    /// subfield. Every other stage ignores it, and every count produces the same proof.
    ///
    /// Raising it trades the dense rounds that follow for more plane work. On a c8a.16xlarge
    /// at 2^20 rows, moving from three to four took the Blake3 zerocheck from 1.30 s to 1.22 s
    /// and the Keccak-f zerocheck from 423 ms to 657 ms: the wider trace amortizes the AIR
    /// evaluation over its sixty-four rows, the narrower one does not.
    ///
    /// A count above [`MAX_SLICED_ROUNDS`] is capped by it, as is one above the row variables
    /// a stage keeps once a word's lanes are spent. Both caps are silent: the proof is the
    /// same either way, so only the timing of a stage tells them apart. With its
    /// late-materialization parameter set, [`ReprBackend`](crate::ReprBackend) serves a stage
    /// that qualifies one further round from its planes, which this count does not grant.
    fn sliced_rounds(&self) -> usize {
        DEFAULT_SLICED_ROUNDS
    }

    /// Borrow the commitment scheme for the preprocessed trace.
    ///
    /// The preprocessed trace stacks a different column count than the main trace.
    /// Its stacked polynomial therefore has a different arity.
    /// That arity needs its own scheme instance.
    ///
    /// Called only during setup, and only when the AIR declares preprocessed columns.
    /// Configurations for AIRs without a preprocessed trace need not override it.
    ///
    /// # Panics
    ///
    /// The default panics.
    /// A configuration proving an AIR with preprocessed columns must override it.
    fn preprocessed_pcs(&self) -> &Self::Pcs {
        unimplemented!("this configuration does not provide a preprocessed commitment scheme")
    }

    /// Smallest table arity the commitment scheme accepts without padding.
    ///
    /// A table below this floor is zero-padded before commitment, which breaks the successor view:
    /// - padding moves the repeated boundary row into the pad,
    /// - so the repeat-last successor view no longer reads itself,
    /// - which disagrees with the zerocheck successor convention.
    ///
    /// The committed prover therefore requires the trace arity to meet this floor.
    /// The opening point and the committed table then share one frame.
    ///
    /// For a WHIR-backed config this is the first-round folding factor.
    fn min_num_variables(&self) -> usize;

    /// Pack committed trace tables into the commitment scheme's witness form.
    ///
    /// The witness representation is private to the commitment scheme:
    /// - it folds in the slot layout and preprocessing depth that scheme expects,
    /// - the generic commitment interface exposes neither,
    /// - so the columns-to-witness bridge lives here, where those details are known.
    ///
    /// # Arguments
    ///
    /// - `tables`: source tables whose rows are trace-column polynomials.
    fn build_witness(
        &self,
        tables: Vec<Table<Self::Val>>,
    ) -> <Self::Pcs as MultilinearPcs<Self::Challenge, Self::Challenger>>::Witness;

    /// Borrows one committed trace table retained between commit and open.
    fn committed_table<'a>(
        &self,
        prover_data: &'a ProverData<Self>,
        table_index: usize,
    ) -> &'a Table<Self::Val>;
}

/// Commitment scheme selected by a configuration.
pub type Pcs<C> = <C as MultiStarkConfig>::Pcs;

/// Succinct commitment produced by a configuration's commitment scheme.
pub type Commitment<C> = <Pcs<C> as MultilinearPcs<
    <C as MultiStarkConfig>::Challenge,
    <C as MultiStarkConfig>::Challenger,
>>::Commitment;

/// Prover-only data retained between commit and open.
pub type ProverData<C> = <Pcs<C> as MultilinearPcs<
    <C as MultiStarkConfig>::Challenge,
    <C as MultiStarkConfig>::Challenger,
>>::ProverData;

/// Opening proof produced by a configuration's commitment scheme.
pub type PcsProof<C> = <Pcs<C> as MultilinearPcs<
    <C as MultiStarkConfig>::Challenge,
    <C as MultiStarkConfig>::Challenger,
>>::Proof;

/// Opening-verification error produced by a configuration's commitment scheme.
pub type PcsError<C> = <Pcs<C> as MultilinearPcs<
    <C as MultiStarkConfig>::Challenge,
    <C as MultiStarkConfig>::Challenger,
>>::Error;

/// Configuration or budget failure reported by the proving commitment scheme.
pub type PcsProverError<C> = <Pcs<C> as MultilinearPcs<
    <C as MultiStarkConfig>::Challenge,
    <C as MultiStarkConfig>::Challenger,
>>::ProverError;
