//! Proving and verifying keys carrying the reusable preprocessed commitment.
//!
//! The preprocessed trace is fixed by the AIR, not the witness.
//! It is committed once here and reused across every proof for that AIR and trace height.
//!
//! The prover key keeps the committed data so each proof opens it without re-encoding.
//! The verifier key keeps only the commitment.
//! Both sides absorb that commitment before sampling any challenge.

use alloc::vec::Vec;

use p3_air::symbolic::AirLayout;
use p3_air::{Air, BaseAir};
use p3_bus::BusSymbolicBuilder;
use p3_commit::MultilinearPcs;
use p3_field::{ExtensionField, Field};
use p3_lookup::InteractionSymbolicBuilder;
use p3_matrix::Matrix;
use p3_sumcheck::layout::Table;
use p3_util::log2_strict_usize;

use crate::ProvingError;
use crate::config::{Commitment, MultiStarkConfig, PcsProverError, ProverData};
use crate::rounds::AirProfile;
use crate::zerocheck::get_air_profile;

/// Batched preprocessed data the prover reuses across proofs.
///
/// All preprocessed tables are stacked into one commitment in AIR-instance
/// order, skipping AIRs that have no preprocessed columns.
pub(crate) struct PreprocessedProverData<C: MultiStarkConfig> {
    /// Commitment to the stacked preprocessed tables.
    pub(crate) commitment: Commitment<C>,
    /// Committed prover data behind the batched commitment, cloned per proof to open.
    pub(crate) prover_data: ProverData<C>,
}

/// The prover's key for an ordered AIR batch and its fixed trace heights.
///
/// The proof must use the same AIRs in the same order as setup.
/// This remains required when no AIR has preprocessed columns.
pub struct ProvingKey<C: MultiStarkConfig> {
    /// Batched preprocessed data, present only when at least one AIR declares it.
    pub(crate) preprocessed: Option<PreprocessedProverData<C>>,
    /// Zerocheck metadata fixed by the AIRs at setup.
    pub(crate) air_profiles: Vec<AirProfile>,
    /// Whether any AIR declares a binary-bus interaction, read off one bus pass at setup.
    pub(crate) declares_bus: bool,
    /// Every AIR's widths and public-value count, in setup order.
    air_shapes: Vec<AirShape>,
}

/// The widths and public-value count of one AIR, which cost nothing to read again.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AirShape {
    /// Main columns.
    width: usize,
    /// Preprocessed columns.
    preprocessed_width: usize,
    /// Public values the AIR reads.
    num_public_values: usize,
}

impl AirShape {
    /// The shape `air` declares.
    fn of<F, A: BaseAir<F>>(air: &A) -> Self {
        Self {
            width: air.width(),
            preprocessed_width: air.preprocessed_width(),
            num_public_values: air.num_public_values(),
        }
    }
}

impl<C: MultiStarkConfig> ProvingKey<C> {
    /// Whether this key can have come from `setup` over `airs`, in this order.
    ///
    /// The count, every width and every public-value count are compared on each call.
    ///
    /// A debug build also reruns setup's symbolic passes and compares what they record, so a
    /// key whose AIRs declare other lookups or buses under the same shapes is caught there too.
    pub(crate) fn describes<A>(&self, airs: &[&A]) -> bool
    where
        A: BaseAir<C::Val>
            + Air<InteractionSymbolicBuilder<C::Val, C::Challenge>>
            + Air<BusSymbolicBuilder<C::Val, C::Challenge>>,
    {
        let shapes = airs.len() == self.air_shapes.len()
            && airs
                .iter()
                .zip(&self.air_shapes)
                .all(|(&air, &shape)| AirShape::of::<C::Val, A>(air) == shape);
        #[cfg(debug_assertions)]
        let shapes = shapes
            && airs
                .iter()
                .zip(&self.air_profiles)
                .all(|(&air, &profile)| get_air_profile::<C::Val, C::Challenge, A>(air) == profile)
            && declares_bus::<C::Val, C::Challenge, A>(airs) == self.declares_bus;
        shapes
    }
}

/// Whether any AIR declares a binary-bus interaction.
///
/// Those declarations are recorded by their own builder, so they take their own pass.
fn declares_bus<F, EF, A>(airs: &[&A]) -> bool
where
    F: Field,
    EF: ExtensionField<F>,
    A: Air<BusSymbolicBuilder<F, EF>>,
{
    airs.iter().any(|&air| {
        let profile = BusSymbolicBuilder::<F, EF>::from_air(air, AirLayout::from_air::<F>(air));
        !profile.interactions().is_empty()
    })
}

/// The verifier's key for an ordered AIR batch and its fixed trace heights.
///
/// Verification must use the same AIRs in the same order as setup.
/// This remains required when no AIR has preprocessed columns.
pub struct VerifyingKey<C: MultiStarkConfig> {
    /// Batched preprocessed commitment, present only when at least one AIR declares it.
    pub(crate) preprocessed: Option<Commitment<C>>,
    /// Row variables of each committed preprocessed table, in batch order.
    ///
    /// The commitment alone does not fix them.
    ///
    /// Stacking places tables tallest first, whichever AIR owns each one.
    ///
    /// Tables of 2^8 and 2^7 rows therefore commit to the same polynomial in either assignment.
    pub(crate) preprocessed_log_heights: Vec<usize>,
    /// Zerocheck metadata fixed by the AIRs at setup.
    pub(crate) air_profiles: Vec<AirProfile>,
}

/// Commit all AIR preprocessed traces once, returning matched prover and verifier keys.
///
/// When the AIR declares no preprocessed trace, both keys carry no preprocessed data.
/// The proof then runs exactly as the main-only flow does.
///
/// The AIR order fixed here is the batch order.
/// The prover-side and verifier-side batches must list their instances in this same order.
/// The preprocessed tables are stacked in this order, so a mismatch pairs each instance with the wrong table.
///
/// The commitment is deterministic in the trace.
/// The throwaway challenger therefore only satisfies the commit signature.
/// Its post-state is discarded.
/// Prover and verifier both re-absorb the stored commitment into the real transcript before sampling.
///
/// # Arguments
///
/// - `config`: the proof configuration selecting the preprocessed commitment scheme.
/// - `airs`: AIRs in proof order.
/// - `challenger`: a throwaway transcript used only for its commit side effect.
///
/// # Panics
///
/// Panics if an AIR declares preprocessed columns but does not return a preprocessed trace.
#[allow(clippy::type_complexity)]
pub fn setup<C, A>(
    config: &C,
    airs: &[&A],
    challenger: &mut C::Challenger,
) -> Result<(ProvingKey<C>, VerifyingKey<C>), ProvingError<PcsProverError<C>>>
where
    C: MultiStarkConfig,
    A: BaseAir<C::Val>
        + Air<InteractionSymbolicBuilder<C::Val, C::Challenge>>
        + Air<BusSymbolicBuilder<C::Val, C::Challenge>>,
    Commitment<C>: Clone,
{
    let air_profiles = airs
        .iter()
        .map(|&air| get_air_profile::<C::Val, C::Challenge, A>(air))
        .collect::<Vec<_>>();
    let declares_bus = declares_bus::<C::Val, C::Challenge, A>(airs);
    let air_shapes = airs
        .iter()
        .map(|&air| AirShape::of::<C::Val, A>(air))
        .collect::<Vec<_>>();
    let mut tables = Vec::new();
    let mut preprocessed_log_heights = Vec::new();

    for air in airs.iter().filter(|air| air.preprocessed_width() != 0) {
        let trace = air
            .preprocessed_trace()
            .expect("AIR with preprocessed columns must return a preprocessed trace");

        preprocessed_log_heights.push(log2_strict_usize(trace.height()));
        tables.push(Table::new(trace.transpose()));
    }

    if tables.is_empty() {
        return Ok((
            ProvingKey {
                preprocessed: None,
                air_profiles: air_profiles.clone(),
                declares_bus,
                air_shapes,
            },
            VerifyingKey {
                preprocessed: None,
                preprocessed_log_heights,
                air_profiles,
            },
        ));
    }

    // Turn traces into one multilinear per column, then commit the stacked tables once.
    let witness = config.build_witness(tables);
    let (commitment, prover_data) = config
        .preprocessed_pcs()
        .commit(witness, challenger)
        .map_err(|source| ProvingError::Pcs {
            phase: "preprocessing commitment",
            source,
        })?;

    // The prover key keeps committed data to open each proof.
    let proving = ProvingKey {
        preprocessed: Some(PreprocessedProverData {
            commitment: commitment.clone(),
            prover_data,
        }),
        air_profiles: air_profiles.clone(),
        declares_bus,
        air_shapes,
    };
    // The verifier key keeps only the commitment; shape facts come from AIR metadata.
    let verifying = VerifyingKey {
        preprocessed: Some(commitment),
        preprocessed_log_heights,
        air_profiles,
    };
    Ok((proving, verifying))
}
