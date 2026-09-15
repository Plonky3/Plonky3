//! Proof data and opening shapes for multilinear AIR verification.

use alloc::vec::Vec;

use p3_sumcheck::generic_degree::GenericDegreeProof;
use serde::{Deserialize, Serialize};

use crate::config::{Commitment, MultiStarkConfig, PcsProof};
use crate::fractional_gkr::FractionGkrProof;
use crate::logup_star::LogupStarProof;

/// One batch's indexed-lookup round.
///
/// # Soundness
///
/// The claims travel in the proof because the reduction runs before the opening.
///
/// A verifier cannot have them earlier.
///
/// They are values of committed columns, and only an opening supplies those.
///
/// Two things then stand between a prover and a forged claim.
///
/// The reduction binds every claim before drawing any challenge of its own.
///
/// The closing check compares each claim against the opening of its payload column.
///
/// So a claim is fixed before it can be tuned, and authenticated before it is believed.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize, EF: Serialize"))]
#[serde(bound(deserialize = "F: Deserialize<'de>, EF: Deserialize<'de>"))]
pub struct IndexedLookupProof<F, EF> {
    /// What each reader claims it pulled, at the bound point, in plan order.
    ///
    /// One inner vector per reader, holding one value per column of the table it reads.
    pub reader_claims: Vec<Vec<EF>>,
    /// The reduction tying those claims to the tables and the position columns.
    pub reduction: LogupStarProof<F, EF>,
}

/// A complete proof for AIR instances sharing one zerocheck.
///
/// The parts are checked in order against one shared transcript:
/// - the commitment binds all main trace tables.
/// - the optional lookup proof reduces the materialized fractions with GKR.
/// - the sumcheck reduces the AIR constraint to one bound-point claim.
/// - the optional indexed round reduces every indexed read against that point.
/// - the main opening proves all main trace tables at every point a claim was left at.
/// - the preprocessed opening, when present, proves all preprocessed tables the same way.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct MultiStarkProof<C: MultiStarkConfig> {
    /// Commitment to all main trace tables in input-instance order.
    pub commitment: Commitment<C>,
    /// Fractional-GKR lookup proof, absent when no AIR declares interactions.
    pub lookup: Option<FractionGkrProof<C::Challenge>>,
    /// Indexed-lookup round, absent when no AIR declares an indexed read.
    pub indexed: Option<IndexedLookupProof<C::Val, C::Challenge>>,
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
