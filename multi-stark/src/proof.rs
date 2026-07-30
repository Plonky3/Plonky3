//! Proof data and opening shapes for multilinear AIR verification.

use p3_sumcheck::generic_degree::GenericDegreeProof;
use serde::{Deserialize, Serialize};

use crate::config::{Commitment, MultiStarkConfig, PcsProof};
use crate::fractional_gkr::FractionGkrProof;

/// A complete proof for AIR instances sharing one zerocheck.
///
/// The parts are checked in order against one shared transcript:
/// - the commitment binds all main trace tables.
/// - the optional lookup proof reduces the materialized fractions with GKR.
/// - the sumcheck reduces the AIR constraint to one bound-point claim.
/// - the main opening proves all main trace tables at that point.
/// - the preprocessed opening, when present, proves all preprocessed tables at that point.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct MultiStarkProof<C: MultiStarkConfig> {
    /// Commitment to all main trace tables in input-instance order.
    pub commitment: Commitment<C>,
    /// Fractional-GKR lookup proof, absent when no AIR declares interactions.
    pub lookup: Option<FractionGkrProof<C::Challenge>>,
    /// Zerocheck sumcheck transcript for the beta-batched AIR constraints.
    pub sumcheck: GenericDegreeProof<C::Val, C::Challenge>,
    /// Main-trace opening for every committed main table.
    pub opening: PcsProof<C>,
    /// Batched preprocessed-trace opening.
    ///
    /// `None` when no AIR in the batch declares preprocessed columns.
    pub preprocessed_opening: Option<PcsProof<C>>,
}

impl<C: MultiStarkConfig> core::fmt::Debug for MultiStarkProof<C>
where
    Commitment<C>: core::fmt::Debug,
    C::Val: core::fmt::Debug,
    C::Challenge: core::fmt::Debug,
    PcsProof<C>: core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MultiStarkProof")
            .field("commitment", &self.commitment)
            .field("lookup", &self.lookup)
            .field("sumcheck", &self.sumcheck)
            .field("opening", &self.opening)
            .finish()
    }
}
