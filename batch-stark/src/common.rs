//! Shared data between batch-STARK prover and verifier.
//!
//! This module is intended to store per-instance data that is common to both
//! proving and verification, such as lookup tables and preprocessed traces.
//!
//! The preprocessed support integrates with `p3-uni-stark`'s transparent
//! preprocessed columns API, but batches all preprocessed traces into a single
//! global commitment (one matrix per instance that uses preprocessed columns).

use alloc::vec;
use alloc::vec::Vec;

use hashbrown::HashMap;
use p3_air::Air;
use p3_air::symbolic::{SymbolicAirBuilder, SymbolicExpression};
use p3_challenger::FieldChallenger;
use p3_commit::Pcs;
use p3_field::{Algebra, BasedVectorSpace};
use p3_lookup::lookup_traits::{Kind, Lookup, LookupGadget};
use p3_matrix::Matrix;
use p3_uni_stark::Val;
use p3_util::log2_strict_usize;

use crate::config::{Challenge, Commitment, Domain, StarkGenericConfig as SGC};
use crate::prover::StarkInstance;

/// Per-instance metadata for a preprocessed trace that lives inside a
/// global preprocessed commitment.
#[derive(Clone)]
pub struct PreprocessedInstanceMeta {
    /// Index of this instance's preprocessed matrix inside the global [`Pcs`]
    /// commitment / prover data.
    pub matrix_index: usize,
    /// Width (number of columns) of the preprocessed trace.
    pub width: usize,
    /// Log2 of the base trace degree for this instance's preprocessed trace.
    ///
    /// This matches the log2 of the main trace degree (without ZK padding)
    /// for that instance.
    pub degree_bits: usize,
}

/// Global preprocessed data shared by all batch-STARK instances.
///
/// This batches all per-instance preprocessed traces into a single [`Pcs`]
/// commitment, while keeping a mapping from instance index to matrix index
/// and per-matrix metadata.
pub struct GlobalPreprocessed<SC: SGC> {
    /// Single [`Pcs`] commitment to all preprocessed traces (one matrix per
    /// instance that defines preprocessed columns).
    pub commitment: Commitment<SC>,
    /// For each STARK instance, optional metadata describing its preprocessed
    /// trace inside the global commitment.
    ///
    /// `instances[i] == None` means instance `i` has no preprocessed columns.
    pub instances: Vec<Option<PreprocessedInstanceMeta>>,
    /// Mapping from preprocessed matrix index to the corresponding instance index.
    ///
    /// This allows building per-matrix opening schedules and routing opened
    /// values back to instances.
    pub matrix_to_instance: Vec<usize>,
}

/// Struct storing data common to both the prover and verifier.
// TODO: Optionally cache a single challenger seed for transparent
//       preprocessed data (per-instance widths + global root), so
//       prover and verifier don't have to recompute/rehash it each run.
pub struct CommonData<SC: SGC> {
    /// Optional global preprocessed commitment shared by all instances.
    ///
    /// When `None`, no instance uses preprocessed columns.
    pub preprocessed: Option<GlobalPreprocessed<SC>>,
    /// The lookups used by each STARK instance.
    /// There is one `Vec<Lookup<Val<SC>>>` per STARK instance.
    /// They are stored in the same order as the STARK instance inputs provided to `new`.
    pub lookups: Vec<Vec<Lookup<Val<SC>>>>,
}

/// Prover-exclusive data not shared with the verifier.
///
/// This contains the PCS prover data for preprocessed traces, which is only
/// needed during proving.
pub struct ProverOnlyData<SC: SGC> {
    /// PCS prover data for preprocessed traces.
    ///
    /// Present only when at least one instance has preprocessed columns.
    pub preprocessed_prover_data:
        Option<<SC::Pcs as Pcs<Challenge<SC>, SC::Challenger>>::ProverData>,
}

/// Combined prover data containing both common and prover-only data.
///
/// This is a convenience struct that bundles [`CommonData`] (shared with verifier)
/// and [`ProverOnlyData`] (prover-exclusive) together.
pub struct ProverData<SC: SGC> {
    /// Data shared between prover and verifier.
    pub common: CommonData<SC>,
    /// Prover-exclusive data.
    pub prover_only: ProverOnlyData<SC>,
}

impl<SC: SGC> CommonData<SC> {
    pub const fn new(
        preprocessed: Option<GlobalPreprocessed<SC>>,
        lookups: Vec<Vec<Lookup<Val<SC>>>>,
    ) -> Self {
        Self {
            preprocessed,
            lookups,
        }
    }

    /// Create [`CommonData`] with no preprocessed columns or lookups.
    ///
    /// Use this when none of your [`Air`] implementations have preprocessed columns or lookups.
    pub fn empty(num_instances: usize) -> Self {
        let lookups = vec![Vec::new(); num_instances];
        Self {
            preprocessed: None,
            lookups,
        }
    }
}

impl<SC: SGC> ProverOnlyData<SC> {
    /// Create empty [`ProverOnlyData`] with no preprocessed prover data.
    pub const fn empty() -> Self {
        Self {
            preprocessed_prover_data: None,
        }
    }
}

impl<SC: SGC> ProverData<SC> {
    /// Create [`ProverData`] with no preprocessed columns or lookups.
    ///
    /// Use this when none of your [`Air`] implementations have preprocessed columns or lookups.
    pub fn empty(num_instances: usize) -> Self {
        Self {
            common: CommonData::empty(num_instances),
            prover_only: ProverOnlyData::empty(),
        }
    }
}

impl<SC> ProverData<SC>
where
    SC: SGC,
    Challenge<SC>: BasedVectorSpace<Val<SC>>,
{
    /// Build [`ProverData`] directly from STARK instances.
    ///
    /// This automatically:
    /// - Derives trace degrees from trace heights
    /// - Computes extended degrees (base + ZK padding)
    /// - Sets up preprocessed columns for [`Air`] implementations that define them, committing
    ///   to them in a single global [`Pcs`] commitment.
    /// - Deduces symbolic lookups from the STARKs
    ///
    /// This is a convenience function mainly used for tests.
    pub fn from_instances<A>(config: &SC, instances: &[StarkInstance<'_, SC, A>]) -> Self
    where
        SymbolicExpression<SC::Challenge>: Algebra<SymbolicExpression<Val<SC>>>,
        A: Air<SymbolicAirBuilder<Val<SC>, SC::Challenge>> + Clone,
    {
        let degrees: Vec<usize> = instances.iter().map(|i| i.trace.height()).collect();
        let log_ext_degrees: Vec<usize> = degrees
            .iter()
            .map(|&d| log2_strict_usize(d) + config.is_zk())
            .collect();
        let mut airs: Vec<A> = instances.iter().map(|i| i.air.clone()).collect();
        Self::from_airs_and_degrees(config, &mut airs, &log_ext_degrees)
    }

    /// Build [`ProverData`] from [`Air`] implementations and their extended trace degree bits.
    ///
    /// # Arguments
    ///
    /// * `trace_ext_degree_bits` - Log2 of extended trace degrees (including ZK padding)
    ///
    /// # Returns
    ///
    /// Prover data containing the global preprocessed commitment (if at least
    /// one [`Air`] defines preprocessed columns) and the PCS prover data.
    pub fn from_airs_and_degrees<A>(
        config: &SC,
        airs: &mut [A],
        trace_ext_degree_bits: &[usize],
    ) -> Self
    where
        SymbolicExpression<SC::Challenge>: Algebra<SymbolicExpression<Val<SC>>>,
        A: Air<SymbolicAirBuilder<Val<SC>, SC::Challenge>>,
    {
        assert_eq!(
            airs.len(),
            trace_ext_degree_bits.len(),
            "airs and trace_ext_degree_bits must have the same length"
        );

        let pcs = config.pcs();
        let is_zk = config.is_zk();

        let mut instances_meta: Vec<Option<PreprocessedInstanceMeta>> =
            Vec::with_capacity(airs.len());
        let mut matrix_to_instance: Vec<usize> = Vec::new();
        let mut domains_and_traces: Vec<(Domain<SC>, _)> = Vec::new();

        for (i, (air, &ext_db)) in airs.iter().zip(trace_ext_degree_bits.iter()).enumerate() {
            // Derive base trace degree bits from extended degree bits.
            let base_db = ext_db - is_zk;
            let maybe_prep = air.preprocessed_trace();

            let Some(preprocessed) = maybe_prep else {
                instances_meta.push(None);
                continue;
            };

            let width = preprocessed.width();
            if width == 0 {
                instances_meta.push(None);
                continue;
            }

            let degree = 1 << base_db;
            let ext_degree = 1 << ext_db;
            assert_eq!(
                preprocessed.height(),
                degree,
                "preprocessed trace height must equal trace degree for instance {}",
                i
            );

            let domain = pcs.natural_domain_for_degree(ext_degree);
            let matrix_index = domains_and_traces.len();

            domains_and_traces.push((domain, preprocessed));
            matrix_to_instance.push(i);

            instances_meta.push(Some(PreprocessedInstanceMeta {
                matrix_index,
                width,
                degree_bits: ext_db,
            }));
        }

        let (preprocessed, preprocessed_prover_data) = if domains_and_traces.is_empty() {
            (None, None)
        } else {
            let (commitment, prover_data) = pcs.commit_preprocessing(domains_and_traces);
            (
                Some(GlobalPreprocessed {
                    commitment,
                    instances: instances_meta,
                    matrix_to_instance,
                }),
                Some(prover_data),
            )
        };

        let lookups = airs.iter_mut().map(|air| air.get_lookups()).collect();

        Self {
            common: CommonData {
                preprocessed,
                lookups,
            },
            prover_only: ProverOnlyData {
                preprocessed_prover_data,
            },
        }
    }
}

pub fn get_perm_challenges<SC: SGC, LG: LookupGadget>(
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

            for context in contexts {
                match &context.kind {
                    Kind::Global(name) => {
                        // Get or create the global challenges.
                        let challenges: &mut Vec<SC::Challenge> =
                            global_perm_challenges.entry(name).or_insert_with(|| {
                                (0..num_challenges_per_lookup)
                                    .map(|_| challenger.sample_algebra_element())
                                    .collect()
                            });
                        instance_challenges.extend_from_slice(challenges);
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
