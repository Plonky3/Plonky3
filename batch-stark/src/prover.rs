//! Batch-STARK prover: commits traces, computes quotient polynomials, and
//! produces opening proofs for multiple AIR instances in a single FRI batch.

use alloc::vec;
use alloc::vec::Vec;

#[cfg(debug_assertions)]
use p3_air::DebugConstraintBuilder;
use p3_air::symbolic::{AirLayout, SymbolicExpressionExt};
use p3_air::{Air, RowWindow};
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::{
    Algebra, BasedVectorSpace, PackedFieldExtension, PackedValue, PrimeCharacteristicRing,
};
use p3_lookup::folder::ProverConstraintFolderWithLookups;
use p3_lookup::logup::LogUpGadget;
use p3_lookup::{InteractionSymbolicBuilder, Kind, Lookup, LookupData, LookupProtocol};
use p3_matrix::Matrix;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_maybe_rayon::DisjointMutPtr;
use p3_maybe_rayon::prelude::*;
use p3_uni_stark::{OpenedValues, PackedChallenge, PackedVal, ProverConstraintFolder};
use p3_util::log2_strict_usize;
use tracing::{debug_span, info_span, instrument};

use crate::common::ProverData;
use crate::config::{Challenge, Domain, StarkGenericConfig as SGC, Val};
use crate::proof::{BatchCommitments, BatchOpenedValues, BatchProof, OpenedValuesWithLookups};
use crate::symbolic::{
    get_constraint_layout, get_log_num_quotient_chunks, get_symbolic_constraints,
};
use crate::transcript::BatchTranscript;

/// A single AIR instance bundled with its execution trace, public inputs,
/// and lookup declarations.
///
/// One or more of these are passed to the batch prover.
#[derive(Debug)]
pub struct StarkInstance<'a, SC: SGC, A> {
    /// The AIR (constraint system) for this instance.
    pub air: &'a A,
    /// Execution trace as a row-major matrix over the base field.
    pub trace: &'a RowMajorMatrix<Val<SC>>,
    /// Public input values exposed by this instance.
    pub public_values: Vec<Val<SC>>,
}

impl<'a, SC: SGC, A> StarkInstance<'a, SC, A> {
    /// Build instances from parallel slices of AIRs, traces, and public values.
    pub fn new_multiple(
        airs: &'a [A],
        traces: &'a [&'a RowMajorMatrix<Val<SC>>],
        public_values: &[Vec<Val<SC>>],
    ) -> Vec<Self> {
        airs.iter()
            .zip(traces.iter())
            .zip(public_values.iter())
            .map(|((air, trace), public_values)| Self {
                air,
                trace,
                public_values: public_values.clone(),
            })
            .collect()
    }
}

/// Generate a batch STARK proof for all provided instances.
///
/// # Overview
///
/// Runs the full prover pipeline:
/// - Commit to execution traces (single batched commitment).
/// - Generate and commit permutation traces for lookup arguments.
/// - Compute quotient polynomials per instance.
/// - Commit to quotient chunks.
/// - Open all commitments at a random out-of-domain point.
///
/// # Arguments
///
/// - `config`: STARK configuration (PCS, challenger, ZK flag).
/// - `instances`: one entry per AIR instance with its trace and public values.
/// - `prover_data`: precomputed common data and preprocessed commitments.
///
/// # Returns
///
/// A self-contained batch proof that can be verified with `verify_batch`.
#[instrument(skip_all)]
pub fn prove_batch<
    SC,
    #[cfg(debug_assertions)] A: for<'a> Air<DebugConstraintBuilder<'a, Val<SC>, SC::Challenge>>
        + Air<InteractionSymbolicBuilder<Val<SC>, SC::Challenge>>
        + for<'a> Air<ProverConstraintFolderWithLookups<'a, SC>>
        + Clone,
    #[cfg(not(debug_assertions))] A: for<'a> Air<InteractionSymbolicBuilder<Val<SC>, SC::Challenge>>
        + for<'a> Air<ProverConstraintFolderWithLookups<'a, SC>>
        + Clone,
>(
    config: &SC,
    instances: &[StarkInstance<'_, SC, A>],
    prover_data: &ProverData<SC>,
) -> BatchProof<SC>
where
    SC: SGC,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>: Algebra<SC::Challenge>,
{
    let common = &prover_data.common;
    // TODO: Extend if additional lookup gadgets are added.
    let lookup_gadget = LogUpGadget::new();

    let pcs = config.pcs();
    let mut transcript = BatchTranscript::<SC>::new(config.initialise_challenger());

    // Collect per-instance degree information.
    let degrees: Vec<usize> = instances.iter().map(|i| i.trace.height()).collect();
    let log_degrees: Vec<usize> = degrees.iter().copied().map(log2_strict_usize).collect();
    // Extended degree accounts for the ZK blinding factor (2x when ZK is enabled).
    let log_ext_degrees: Vec<usize> = log_degrees.iter().map(|&d| d + config.is_zk()).collect();

    // Read lookups from the keygen-cached CommonData (not from instances).
    let all_lookups: Vec<&[Lookup<Val<SC>>]> = common.lookups.iter().map(|l| &**l).collect();
    let mut lookup_data: Vec<Vec<_>> = all_lookups
        .iter()
        .map(|lookups| {
            // Only global lookups produce cumulated values that enter the transcript.
            lookups
                .iter()
                .filter_map(|lookup| match &lookup.kind {
                    Kind::Global(name) => Some(LookupData {
                        name: name.clone(),
                        aux_column: lookup.column,
                        cumulative_sum: SC::Challenge::ZERO,
                    }),
                    _ => None,
                })
                .collect::<Vec<_>>()
        })
        .collect();

    // Base and extended domains for every instance.
    let (trace_domains, ext_trace_domains): (Vec<Domain<SC>>, Vec<Domain<SC>>) = degrees
        .iter()
        .map(|&deg| {
            (
                pcs.natural_domain_for_degree(deg),
                pcs.natural_domain_for_degree(deg * (config.is_zk() + 1)),
            )
        })
        .unzip();

    // Extract AIRs and borrow public values; consume traces later without cloning.
    let airs: Vec<&A> = instances.iter().map(|i| i.air).collect();
    let pub_vals: Vec<&[Val<SC>]> = instances
        .iter()
        .map(|i| i.public_values.as_slice())
        .collect();

    // Determine preprocessed widths and quotient chunk counts per instance.
    let mut preprocessed_widths = Vec::with_capacity(airs.len());
    let (log_num_quotient_chunks, num_quotient_chunks): (Vec<usize>, Vec<usize>) = airs
        .iter()
        .zip(pub_vals.iter())
        .enumerate()
        .map(|(i, (air, _pv))| {
            // Width of the preprocessed trace for this instance (0 if absent).
            let pre_w = common
                .preprocessed
                .as_ref()
                .and_then(|g| g.instances[i].as_ref().map(|m| m.width))
                .unwrap_or(0);
            preprocessed_widths.push(pre_w);

            let layout = AirLayout {
                preprocessed_width: pre_w,
                main_width: air.width(),
                num_public_values: air.num_public_values(),
                num_periodic_columns: air.num_periodic_columns(),
                ..Default::default()
            };

            // Infer the log of the quotient polynomial degree from symbolic analysis.
            let lq_chunks =
                info_span!("infer log of constraint degree", air_idx = i).in_scope(|| {
                    get_log_num_quotient_chunks::<Val<SC>, SC::Challenge, A, LogUpGadget>(
                        air,
                        layout,
                        all_lookups[i],
                        config.is_zk(),
                        &lookup_gadget,
                    )
                });
            // Actual number of quotient chunks (doubled when ZK is enabled).
            let n_chunks = 1 << (lq_chunks + config.is_zk());
            (lq_chunks, n_chunks)
        })
        .unzip();

    let n_instances = airs.len();
    let widths: Vec<usize> = airs.iter().map(|a| A::width(a)).collect();

    // Transcript: Observe instance count and per-instance bindings.
    transcript.observe_instance_count(n_instances);
    for i in 0..n_instances {
        transcript.observe_instance_binding(
            log_ext_degrees[i],
            log_degrees[i],
            widths[i],
            num_quotient_chunks[i],
        );
    }

    // Transcript: Main trace commitment

    // Build PCS inputs for every instance and commit in a single batch.
    let main_commit_inputs = instances
        .iter()
        .zip(ext_trace_domains.iter().cloned())
        .map(|(inst, dom)| (dom, inst.trace.clone()))
        .collect::<Vec<_>>();
    let (main_commit, main_data) = pcs.commit(main_commit_inputs);

    transcript.observe_main(&main_commit, &pub_vals);
    transcript.observe_preprocessed(&preprocessed_widths, common.preprocessed.as_ref());

    // Transcript: Lookup challenges and permutation traces

    // Draw per-instance challenges for the lookup argument.
    let challenges_per_instance = transcript.sample_perm_challenges(&all_lookups, &lookup_gadget);

    // Generate permutation traces for instances that have lookups.
    let mut permutation_commit_inputs = Vec::with_capacity(n_instances);
    instances
        .iter()
        .enumerate()
        .zip(ext_trace_domains.iter().cloned())
        .for_each(|((i, inst), ext_domain)| {
            if !all_lookups[i].is_empty() {
                // Compute the permutation argument trace from lookups and challenges.
                let generated_perm = lookup_gadget.generate_permutation::<SC>(
                    inst.trace,
                    &inst.air.preprocessed_trace(),
                    &inst.public_values,
                    all_lookups[i],
                    &mut lookup_data[i],
                    &challenges_per_instance[i],
                );

                #[cfg(debug_assertions)]
                {
                    use crate::check_constraints::check_constraints;

                    let preprocessed_trace = inst.air.preprocessed_trace();

                    let perm_vals: Vec<SC::Challenge> =
                        lookup_data[i].iter().map(|ld| ld.cumulative_sum).collect();
                    let lookup_constraints_inputs = (all_lookups[i], &lookup_gadget);
                    check_constraints(
                        inst.air,
                        inst.trace,
                        &preprocessed_trace,
                        &generated_perm,
                        &challenges_per_instance[i],
                        &perm_vals,
                        &inst.public_values,
                        lookup_constraints_inputs,
                    );
                }

                // Consume the generated matrix directly; no extra clone before flattening.
                permutation_commit_inputs.push((ext_domain, generated_perm.flatten_to_base()));
            }
        });

    // Debug-only: verify that all lookup sums balance across instances.
    #[cfg(debug_assertions)]
    {
        use p3_lookup::debug_util::{LookupDebugInstance, check_lookups};

        let preprocessed_traces: Vec<_> = instances
            .iter()
            .map(|inst| inst.air.preprocessed_trace())
            .collect();
        let debug_instances: Vec<_> = instances
            .iter()
            .zip(preprocessed_traces.iter())
            .zip(all_lookups.iter())
            .map(|((inst, prep), lookups)| LookupDebugInstance {
                main_trace: inst.trace,
                preprocessed_trace: prep,
                public_values: &inst.public_values,
                lookups,
                permutation_challenges: &[],
            })
            .collect();
        check_lookups(&debug_instances);
    }

    // Commit all permutation traces (if any).
    let permutation_commit_and_data = if !permutation_commit_inputs.is_empty() {
        Some(pcs.commit(permutation_commit_inputs))
    } else {
        None
    };

    // Transcript: observe permutation commitment + lookup data, sample alpha.
    let alpha: Challenge<SC> = transcript.observe_perm_and_sample_alpha(
        permutation_commit_and_data.as_ref().map(|(c, _)| c),
        &lookup_data,
    );

    // Accumulators for quotient chunk domains / matrices / per-instance ranges.
    let mut quotient_chunk_domains = Vec::new();
    let mut quotient_chunk_mats = Vec::new();
    let mut quotient_chunk_ranges = Vec::with_capacity(n_instances);

    // Tracks which permutation matrix index corresponds to each instance.
    let mut perm_counter = 0;

    // TODO: Parallelize this loop for better performance with many instances.
    for (i, trace_domain) in trace_domains.iter().enumerate() {
        let _air_span = info_span!("compute quotient", air_idx = i).entered();

        let log_chunks = log_num_quotient_chunks[i];
        let n_chunks = num_quotient_chunks[i];
        // Build the quotient domain: disjoint from the trace domain,
        // with size = ext_degree * num_quotient_chunks.
        let quotient_domain =
            ext_trace_domains[i].create_disjoint_domain(1 << (log_ext_degrees[i] + log_chunks));

        let sym_layout = AirLayout {
            preprocessed_width: preprocessed_widths[i],
            main_width: airs[i].width(),
            num_public_values: airs[i].num_public_values(),
            num_periodic_columns: airs[i].num_periodic_columns(),
            ..Default::default()
        };

        // Debug-only: verify the static constraint-count hint matches symbolic analysis.
        debug_assert!(
            airs[i].num_constraints().is_none_or(|n| {
                n == get_symbolic_constraints(airs[i], sym_layout, all_lookups[i], &lookup_gadget)
                    .0
                    .len()
            }),
            "num_constraints() = {} but symbolic evaluation found {} base constraints",
            airs[i].num_constraints().unwrap(),
            get_symbolic_constraints(airs[i], sym_layout, all_lookups[i], &lookup_gadget,)
                .0
                .len(),
        );

        // Evaluate the committed main trace on the quotient domain via LDE.
        let trace_on_quotient_domain =
            pcs.get_evaluations_on_domain(&main_data, i, quotient_domain);

        // Evaluate the permutation trace on the quotient domain (if lookups exist).
        let permutation_on_quotient_domain = permutation_commit_and_data
            .as_ref()
            .filter(|_| !all_lookups[i].is_empty())
            .map(|(_, perm_data)| {
                let evals = pcs.get_evaluations_on_domain(perm_data, perm_counter, quotient_domain);
                perm_counter += 1;
                evals
            });

        // Evaluate preprocessed columns on the quotient domain (if present).
        let preprocessed_on_quotient_domain = common
            .preprocessed
            .as_ref()
            .and_then(|g| g.instances[i].as_ref())
            .map(|meta| {
                let preprocessed_prover_data = prover_data
                    .prover_only
                    .preprocessed_prover_data
                    .as_ref()
                    .expect("preprocessed_prover_data must exist when preprocessed columns exist");
                pcs.get_evaluations_on_domain_no_random(
                    preprocessed_prover_data,
                    meta.matrix_index,
                    quotient_domain,
                )
            });

        // Compute quotient(x) = constraints(x) / Z_H(x) on the quotient domain.
        let perm_vals: Vec<_> = lookup_data[i].iter().map(|ld| ld.cumulative_sum).collect();
        let q_values = quotient_values(
            pcs,
            airs[i],
            pub_vals[i],
            sym_layout,
            *trace_domain,
            quotient_domain,
            &trace_on_quotient_domain,
            permutation_on_quotient_domain.as_ref(),
            all_lookups[i],
            &perm_vals,
            &lookup_gadget,
            &challenges_per_instance[i],
            preprocessed_on_quotient_domain.as_ref(),
            alpha,
        );

        // Flatten extension values to base field and split into degree-bounded chunks.
        let q_flat = RowMajorMatrix::new_col(q_values).flatten_to_base();
        let chunk_mats = quotient_domain.split_evals(n_chunks, q_flat);
        let chunk_domains = quotient_domain.split_domains(n_chunks);

        // Compute low-degree extensions of each chunk for commitment.
        let evals = chunk_domains.iter().zip(chunk_mats).map(|(d, m)| (*d, m));
        let ldes = pcs.get_quotient_ldes(evals, n_chunks);

        // Record the range of chunks belonging to this instance.
        let start = quotient_chunk_domains.len();
        quotient_chunk_domains.extend(chunk_domains);
        quotient_chunk_mats.extend(ldes);
        let end = quotient_chunk_domains.len();
        quotient_chunk_ranges.push((start, end));
    }

    // Commit all quotient chunks in a single batch.
    let (quotient_commit, quotient_data) = pcs.commit_ldes(quotient_chunk_mats);
    transcript.observe_quotient_commitment(&quotient_commit);

    // Transcript: Optional ZK randomization polynomial
    //
    // When ZK is enabled, commit to a random extension-field polynomial of
    // degree 2n. The PCS later adds (R(X) - R(z)) / (X - z) to the batch,
    // hiding the trace values at the query points.
    //
    // TODO: This approach is only statistically ZK.
    // A perfectly-ZK version would use a true extension-field polynomial.
    let (opt_r_commit, opt_r_data) = if SC::Pcs::ZK {
        let (r_commit, r_data) = pcs
            .get_opt_randomization_poly_commitment(ext_trace_domains.iter().copied())
            .expect("ZK is enabled, so we should have randomization commitments");
        (Some(r_commit), Some(r_data))
    } else {
        (None, None)
    };

    if let Some(r_commit) = &opt_r_commit {
        transcript.observe_random_commitment(r_commit);
    }

    // Transcript: OOD opening

    // Sample the out-of-domain evaluation point.
    let zeta: Challenge<SC> = transcript.sample_zeta();

    // Build the opening rounds and produce the FRI opening proof.
    let (opened_values, opening_proof) = {
        let mut rounds = Vec::new();

        // Round 0 (optional): randomization polynomial opened at zeta per instance.
        let round0 = opt_r_data.as_ref().map(|r_data| {
            let round0_points = trace_domains.iter().map(|_| vec![zeta]).collect();
            (r_data, round0_points)
        });
        rounds.extend(round0);

        // Round 1: main trace. Open at zeta; also at the next domain point
        // if the AIR accesses the next row.
        let round1_points = trace_domains
            .iter()
            .enumerate()
            .map(|(i, dom)| {
                if !airs[i].main_next_row_columns().is_empty() {
                    vec![
                        zeta,
                        dom.next_point(zeta)
                            .expect("domain should support next_point operation"),
                    ]
                } else {
                    vec![zeta]
                }
            })
            .collect::<Vec<_>>();
        rounds.push((&main_data, round1_points));

        // Round 2: quotient chunks, each opened at zeta only.
        let round2_points = quotient_chunk_ranges
            .iter()
            .cloned()
            .flat_map(|(s, e)| (s..e).map(|_| vec![zeta]))
            .collect::<Vec<_>>();
        rounds.push((&quotient_data, round2_points));

        // Round 3 (optional): preprocessed columns. Open at zeta, and also
        // at the next-row point if the AIR reads preprocessed next-row columns.
        if let Some(global) = &common.preprocessed {
            let preprocessed_prover_data = prover_data
                .prover_only
                .preprocessed_prover_data
                .as_ref()
                .expect("preprocessed_prover_data must exist when preprocessed columns exist");
            let pre_points = global
                .matrix_to_instance
                .iter()
                .map(|&inst_idx| {
                    if !airs[inst_idx].preprocessed_next_row_columns().is_empty() {
                        let zeta_next_i = trace_domains[inst_idx]
                            .next_point(zeta)
                            .expect("domain should support next_point operation");
                        vec![zeta, zeta_next_i]
                    } else {
                        vec![zeta]
                    }
                })
                .collect();
            rounds.push((preprocessed_prover_data, pre_points));
        }

        // Round 4 (optional): permutation traces for instances with lookups.
        // Always opened at both zeta and the next-row point.
        let lookup_points: Vec<_> = trace_domains
            .iter()
            .zip(&all_lookups)
            .filter(|&(_, lookups)| !lookups.is_empty())
            .map(|(dom, _)| {
                vec![
                    zeta,
                    dom.next_point(zeta)
                        .expect("domain should support next_point operation"),
                ]
            })
            .collect();

        if let Some((_, perm_data)) = &permutation_commit_and_data {
            let lookup_round = (perm_data, lookup_points);
            rounds.push(lookup_round);
        }

        pcs.open_with_preprocessing(
            rounds,
            &mut transcript.challenger,
            common.preprocessed.is_some(),
        )
    };

    // Parse opened values into per-instance structures

    // Permutation round follows preprocessed (if present), else takes its slot.
    let permutation_idx = if common.preprocessed.is_some() {
        SC::Pcs::PREPROCESSED_TRACE_IDX + 1
    } else {
        SC::Pcs::PREPROCESSED_TRACE_IDX
    };

    // Main trace opened values: one entry per instance.
    let trace_values_for_mats = &opened_values[SC::Pcs::TRACE_IDX];
    assert_eq!(trace_values_for_mats.len(), n_instances);

    let mut per_instance = Vec::with_capacity(n_instances);

    // Preprocessed openings (if a global preprocessed commitment exists).
    let preprocessed_openings = common
        .preprocessed
        .as_ref()
        .map(|_| &opened_values[SC::Pcs::PREPROCESSED_TRACE_IDX]);

    // Iterator over permutation opened values (one per instance with lookups).
    let is_lookup = permutation_commit_and_data.is_some();
    let permutation_values_for_mats = if is_lookup {
        &opened_values[permutation_idx]
    } else {
        &vec![]
    };
    let mut permutation_values_for_mats = permutation_values_for_mats.iter();

    // Iterate over quotient chunk ranges to assemble per-instance opened values.
    let mut quotient_openings_iter = opened_values[SC::Pcs::QUOTIENT_IDX].iter();
    for (i, (s, e)) in quotient_chunk_ranges.iter().copied().enumerate() {
        // Optional randomization polynomial opening.
        let random = if opt_r_data.is_some() {
            Some(opened_values[0][i][0].clone())
        } else {
            None
        };

        // Main trace: local row always present; next row only if AIR uses it.
        let tv = &trace_values_for_mats[i];
        let trace_local = tv[0].clone();
        let trace_next = if !airs[i].main_next_row_columns().is_empty() {
            Some(tv[1].clone())
        } else {
            None
        };

        // Quotient chunks: collect the zeta-point opening of each chunk.
        let mut qcs = Vec::with_capacity(e - s);
        for _ in s..e {
            let mat_vals = quotient_openings_iter
                .next()
                .expect("chunk index in bounds");
            qcs.push(mat_vals[0].clone());
        }

        // Preprocessed openings: local and optionally next row.
        let (preprocessed_local, preprocessed_next) = if let (Some(global), Some(pre_round)) =
            (&common.preprocessed, preprocessed_openings)
        {
            global.instances[i].as_ref().map_or((None, None), |meta| {
                let vals = &pre_round[meta.matrix_index];
                if !airs[i].preprocessed_next_row_columns().is_empty() {
                    assert_eq!(
                        vals.len(),
                        2,
                        "expected two opening points (zeta, zeta_next) for preprocessed trace"
                    );
                    (Some(vals[0].clone()), Some(vals[1].clone()))
                } else {
                    assert_eq!(
                        vals.len(),
                        1,
                        "expected one opening point (zeta) for preprocessed trace"
                    );
                    (Some(vals[0].clone()), None)
                }
            })
        } else {
            (None, None)
        };

        // Permutation openings: present only for instances with lookups.
        let (permutation_local, permutation_next) = if !all_lookups[i].is_empty() {
            let perm_v = permutation_values_for_mats
                .next()
                .expect("instance should have permutation openings");
            (perm_v[0].clone(), perm_v[1].clone())
        } else {
            (vec![], vec![])
        };

        // Assemble the complete opened values for this instance.
        let base_opened = OpenedValues {
            trace_local,
            trace_next,
            preprocessed_local,
            preprocessed_next,
            quotient_chunks: qcs,
            random,
        };

        per_instance.push(OpenedValuesWithLookups {
            base_opened_values: base_opened,
            permutation_local,
            permutation_next,
        });
    }

    // Extract the permutation commitment (if any) for inclusion in the proof.
    let permutation = permutation_commit_and_data
        .as_ref()
        .map(|(comm, _)| comm.clone());

    // Assemble the final proof structure.
    BatchProof {
        commitments: BatchCommitments {
            main: main_commit,
            quotient_chunks: quotient_commit,
            random: opt_r_commit,
            permutation,
        },
        opened_values: BatchOpenedValues {
            instances: per_instance,
        },
        opening_proof,
        global_lookup_data: lookup_data,
        degree_bits: log_ext_degrees,
    }
}

/// Evaluate the quotient polynomial q(x) = C(x) / Z_H(x) over the quotient
/// domain using packed SIMD arithmetic.
///
/// # Overview
///
/// For each packed chunk of domain points, this function:
/// - Loads the main, preprocessed, and permutation traces at the current and
///   next rows.
/// - Evaluates all AIR constraints (including lookup constraints) into a
///   constraint folder.
/// - Combines constraints using random powers of alpha.
/// - Divides by the vanishing polynomial to get the quotient value.
///
/// # Performance
///
/// Constraint and permutation buffers are allocated once per rayon task and
/// reused across chunk iterations via `for_each_init`, reducing allocator
/// pressure in the hot loop.
///
/// The output is written directly into a pre-allocated buffer through
/// disjoint mutable pointer access, avoiding an intermediate `collect`.
#[instrument(name = "compute quotient polynomial", skip_all)]
#[allow(clippy::too_many_arguments)]
pub fn quotient_values<SC, A, Mat, LG>(
    pcs: &SC::Pcs,
    air: &A,
    public_values: &[Val<SC>],
    layout: AirLayout,
    trace_domain: Domain<SC>,
    quotient_domain: Domain<SC>,
    trace_on_quotient_domain: &Mat,
    opt_permutation_on_quotient_domain: Option<&Mat>,
    lookups: &[Lookup<Val<SC>>],
    permutation_vals: &[SC::Challenge],
    lookup_gadget: &LG,
    permutation_challenges: &[SC::Challenge],
    preprocessed_on_quotient_domain: Option<&Mat>,
    alpha: SC::Challenge,
) -> Vec<SC::Challenge>
where
    SC: SGC,
    A: Air<InteractionSymbolicBuilder<Val<SC>, SC::Challenge>>
        + for<'a> Air<ProverConstraintFolderWithLookups<'a, SC>>,
    Mat: Matrix<Val<SC>> + Sync,
    LG: LookupProtocol + Sync,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>: Algebra<SC::Challenge>,
{
    let quotient_size = quotient_domain.size();
    let main_width = trace_on_quotient_domain.width();
    let (perm_width, perm_height) = opt_permutation_on_quotient_domain
        .as_ref()
        .map_or((0, 0), |mat| (mat.width(), mat.height()));

    // Extension field dimension (number of base-field coordinates per extension element).
    let ext_degree = SC::Challenge::DIMENSION;

    // Compute selector polynomials (is_first_row, is_last_row, etc.) over the quotient domain.
    let mut sels = debug_span!("Compute Selectors")
        .in_scope(|| trace_domain.selectors_on_coset(quotient_domain));

    // Distance between the "current" and "next" row in the quotient domain.
    let qdb = log2_strict_usize(quotient_domain.size()) - log2_strict_usize(trace_domain.size());
    let next_step = 1 << qdb;

    // Pad selectors so every packed chunk can load `pack_width` scalars. When
    // `quotient_size` is not a multiple of `pack_width`, extend by repeating
    // coset values modulo `quotient_size` (matching wrapped trace indexing).
    let pack_width = PackedVal::<SC>::WIDTH;
    let padded_sel_len = quotient_size.next_multiple_of(pack_width);
    let pad_selectors = |v: &mut Vec<Val<SC>>| {
        debug_assert_eq!(v.len(), quotient_size);
        for idx in quotient_size..padded_sel_len {
            v.push(v[idx % quotient_size]);
        }
    };
    pad_selectors(&mut sels.is_first_row);
    pad_selectors(&mut sels.is_last_row);
    pad_selectors(&mut sels.is_transition);
    pad_selectors(&mut sels.inv_vanishing);

    // Decompose alpha into per-constraint powers, split by base vs extension constraints.
    let constraint_layout = get_constraint_layout(air, layout, lookups, lookup_gadget);
    let (base_alpha_powers, ext_alpha_powers) = constraint_layout.decompose_alpha(alpha);

    let periodic_cols = air.periodic_columns();
    let periodic_table =
        pcs.build_periodic_lde_table(&periodic_cols, trace_domain, quotient_domain);

    let periodic_packed: Vec<Vec<PackedVal<SC>>> = if periodic_table.is_empty() {
        Vec::new()
    } else {
        let ncols = periodic_table.width();
        (0..quotient_size)
            .step_by(pack_width)
            .map(|i_start| {
                (0..ncols)
                    .map(|col_idx| {
                        PackedVal::<SC>::from_fn(|offset| {
                            *periodic_table.get(i_start + offset, col_idx)
                        })
                    })
                    .collect()
            })
            .collect()
    };

    // Broadcast scalar challenges to packed representations for SIMD evaluation.
    let packed_perm_challenges: Vec<PackedChallenge<SC>> = permutation_challenges
        .iter()
        .map(|&p_c| PackedChallenge::<SC>::from(p_c))
        .collect();
    let permutation_vals_packed: Vec<PackedChallenge<SC>> = permutation_vals
        .iter()
        .map(|&v| PackedChallenge::<SC>::from(v))
        .collect();

    // Capacities for the reusable per-task buffers.
    let n_base = constraint_layout.base_indices.len();
    let n_ext = constraint_layout.ext_indices.len();
    let constraint_count = constraint_layout.total_constraints();
    let perm_cols = if perm_width > 0 {
        perm_width / ext_degree
    } else {
        0
    };

    // Pre-allocate the output buffer and obtain a disjoint-write pointer.
    // SAFETY: Each chunk writes to `result[i_start..i_start + chunk_emit]` with
    // `chunk_emit = min(pack_width, quotient_size - i_start)`; ranges are disjoint.
    let mut result = SC::Challenge::zero_vec(quotient_size);
    let result_ptr = DisjointMutPtr::new(&mut result);

    (0..quotient_size)
        .into_par_iter()
        .step_by(pack_width)
        .for_each_init(
            // Per-task initialization: allocate constraint and permutation buffers once.
            // These are cleared (without deallocating) and reused on every iteration.
            || {
                (
                    Vec::with_capacity(n_base),
                    Vec::with_capacity(n_ext),
                    Vec::with_capacity(2 * perm_cols),
                )
            },
            |(base_buf, ext_buf, perm_buf), i_start| {
                let chunk_emit = pack_width.min(quotient_size - i_start);
                // Load SIMD-packed selector values for this chunk.
                let i_range = i_start..i_start + pack_width;
                let is_first_row =
                    *PackedVal::<SC>::from_slice(&sels.is_first_row[i_range.clone()]);
                let is_last_row = *PackedVal::<SC>::from_slice(&sels.is_last_row[i_range.clone()]);
                let is_transition =
                    *PackedVal::<SC>::from_slice(&sels.is_transition[i_range.clone()]);
                let inv_vanishing = *PackedVal::<SC>::from_slice(&sels.inv_vanishing[i_range]);

                // Pack the main trace rows (current + next) for this chunk.
                let main = RowMajorMatrix::new(
                    trace_on_quotient_domain.vertically_packed_row_pair(i_start, next_step),
                    main_width,
                );

                // Pack preprocessed rows if the AIR has preprocessed columns.
                let preprocessed = preprocessed_on_quotient_domain.map(|preprocessed| {
                    let preprocessed_width = preprocessed.width();
                    RowMajorMatrix::new(
                        preprocessed.vertically_packed_row_pair(i_start, next_step),
                        preprocessed_width,
                    )
                });

                // Build a packed permutation matrix from element-wise reads.
                // The buffer is cleared and refilled each iteration without reallocating.
                perm_buf.clear();
                if let Some(permutation_on_quotient_domain) =
                    opt_permutation_on_quotient_domain.as_ref()
                {
                    // Current-row permutation columns.
                    perm_buf.extend((0..perm_width).step_by(ext_degree).map(|col| {
                        PackedChallenge::<SC>::from_basis_coefficients_fn(|i| {
                            PackedVal::<SC>::from_fn(|offset| {
                                permutation_on_quotient_domain
                                    .get((i_start + offset) % perm_height, col + i)
                                    .unwrap()
                            })
                        })
                    }));
                    // Next-row permutation columns.
                    perm_buf.extend((0..perm_width).step_by(ext_degree).map(|col| {
                        PackedChallenge::<SC>::from_basis_coefficients_fn(|i| {
                            PackedVal::<SC>::from_fn(|offset| {
                                permutation_on_quotient_domain
                                    .get((i_start + next_step + offset) % perm_height, col + i)
                                    .unwrap()
                            })
                        })
                    }));
                }
                let permutation = RowMajorMatrix::new(core::mem::take(perm_buf), perm_cols);

                let preprocessed_view = preprocessed
                    .as_ref()
                    .map_or_else(|| RowMajorMatrixView::new(&[], 0), |m| m.as_view());

                // Swap in the reusable constraint buffers (already cleared).
                base_buf.clear();
                ext_buf.clear();
                let periodic_values: &[PackedVal<SC>] = if periodic_packed.is_empty() {
                    &[]
                } else {
                    &periodic_packed[i_start / pack_width]
                };
                let inner_folder = ProverConstraintFolder {
                    main: main.as_view(),
                    preprocessed: preprocessed_view,
                    preprocessed_window: RowWindow::from_view(&preprocessed_view),
                    periodic_values,
                    public_values,
                    is_first_row,
                    is_last_row,
                    is_transition,
                    base_alpha_powers: &base_alpha_powers,
                    ext_alpha_powers: &ext_alpha_powers,
                    base_constraints: core::mem::take(base_buf),
                    ext_constraints: core::mem::take(ext_buf),
                    constraint_index: 0,
                    constraint_count,
                };

                // Wrap the inner folder with lookup-specific fields and evaluate.
                let mut folder = ProverConstraintFolderWithLookups {
                    inner: inner_folder,
                    permutation: permutation.as_view(),
                    permutation_challenges: &packed_perm_challenges,
                    permutation_values: &permutation_vals_packed,
                };
                lookup_gadget.eval_air_and_lookups(air, &mut folder, lookups);

                // Combine all constraints with alpha powers and divide by the vanishing polynomial.
                let quotient = folder.inner.finalize_constraints() * inv_vanishing;

                // Reclaim buffers for reuse in the next iteration.
                *base_buf = folder.inner.base_constraints;
                *ext_buf = folder.inner.ext_constraints;
                *perm_buf = folder.permutation.to_row_major_matrix().values;

                // Unpack the SIMD quotient into individual extension-field values
                // and write them directly into the pre-allocated output buffer.
                // SAFETY: Each i_start is unique and targets a disjoint slice.
                let out = unsafe { result_ptr.slice_mut(i_start, chunk_emit) };
                for (idx_in_packing, slot) in out.iter_mut().enumerate() {
                    *slot = quotient.extract(idx_in_packing);
                }
            },
        );

    result
}
