use alloc::vec;
use alloc::vec::Vec;

use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::{BasedVectorSpace, PackedValue, PrimeCharacteristicRing};
use p3_lookup::folder::ProverConstraintFolderWithLookups;
use p3_lookup::lookup_traits::{
    AirLookupHandler, EmptyLookupGadget, Kind, Lookup, LookupData, LookupGadget,
};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use p3_uni_stark::{
    OpenedValues, PackedChallenge, PackedVal, ProverConstraintFolder, SymbolicAirBuilder,
    SymbolicExpression,
};
use p3_util::log2_strict_usize;
use tracing::{debug_span, instrument};

#[cfg(debug_assertions)]
use crate::check_constraints::DebugConstraintBuilderWithLookups;
use crate::common::{CommonData, get_perm_challenges};
use crate::config::{Challenge, Domain, StarkGenericConfig as SGC, Val, observe_instance_binding};
use crate::proof::{BatchCommitments, BatchOpenedValues, BatchProof, OpenedValuesWithLookups};
use crate::symbolic::{get_log_num_quotient_chunks, get_symbolic_constraints};

#[derive(Debug)]
pub struct StarkInstance<'a, SC: SGC, A> {
    pub air: &'a A,
    pub trace: RowMajorMatrix<Val<SC>>,
    pub public_values: Vec<Val<SC>>,
    pub lookups: Vec<Lookup<Val<SC>>>,
}

impl<'a, SC: SGC, A> StarkInstance<'a, SC, A> {
    pub fn new_multiple(
        airs: &'a [A],
        traces: &[RowMajorMatrix<Val<SC>>],
        public_values: &[Vec<Val<SC>>],
        common_data: &CommonData<SC>,
    ) -> Vec<Self> {
        airs.iter()
            .zip(traces.iter())
            .zip(public_values.iter())
            .zip(common_data.lookups.iter())
            .map(|(((air, trace), public_values), lookups)| Self {
                air,
                trace: trace.clone(),
                public_values: public_values.clone(),
                lookups: lookups.clone(),
            })
            .collect()
    }
}

#[instrument(skip_all)]
pub fn prove_batch<
    SC,
    #[cfg(debug_assertions)] A: for<'a> AirLookupHandler<DebugConstraintBuilderWithLookups<'a, Val<SC>, SC::Challenge>>
        + AirLookupHandler<SymbolicAirBuilder<Val<SC>, SC::Challenge>>
        + for<'a> AirLookupHandler<ProverConstraintFolderWithLookups<'a, SC>>
        + Clone,
    #[cfg(not(debug_assertions))] A: for<'a> AirLookupHandler<SymbolicAirBuilder<Val<SC>, SC::Challenge>>
        + for<'a> AirLookupHandler<ProverConstraintFolderWithLookups<'a, SC>>
        + Clone,
    LG,
>(
    config: &SC,
    instances: &[StarkInstance<'_, SC, A>],
    common: &CommonData<SC>,
    lookup_gadget: &LG,
) -> BatchProof<SC>
where
    SC: SGC,
    SymbolicExpression<SC::Challenge>: From<SymbolicExpression<Val<SC>>>,
    LG: LookupGadget + Sync,
{
    let pcs = config.pcs();
    let mut challenger = config.initialise_challenger();

    // TODO: No ZK support for batch-stark yet.
    if config.is_zk() != 0 {
        panic!("p3-batch-stark: ZK mode is not supported yet");
    }

    // Use instances in provided order.
    let degrees: Vec<usize> = instances.iter().map(|i| i.trace.height()).collect();
    let log_degrees: Vec<usize> = degrees.iter().copied().map(log2_strict_usize).collect();
    let log_ext_degrees: Vec<usize> = log_degrees.iter().map(|&d| d + config.is_zk()).collect();

    // Extract lookups and create lookup data in one pass.
    let (all_lookups, mut lookup_data): (Vec<Vec<_>>, Vec<Vec<_>>) = instances
        .iter()
        .map(|inst| {
            (
                inst.lookups.clone(),
                // We only get `LookupData` for global lookups, since we only need it for the expected cumulated value.
                inst.lookups
                    .iter()
                    .filter_map(|lookup| match &lookup.kind {
                        Kind::Global(name) => Some(LookupData {
                            name: name.clone(),
                            aux_idx: lookup.columns[0],
                            expected_cumulated: SC::Challenge::ZERO,
                        }),
                        _ => None,
                    })
                    .collect::<Vec<_>>(),
            )
        })
        .unzip();

    // Domains for each instance (base and extended) in one pass.
    let (trace_domains, ext_trace_domains): (Vec<Domain<SC>>, Vec<Domain<SC>>) = degrees
        .iter()
        .map(|&deg| {
            (
                pcs.natural_domain_for_degree(deg),
                pcs.natural_domain_for_degree(deg * (config.is_zk() + 1)),
            )
        })
        .unzip();

    // Extract AIRs and public values; consume traces later without cloning.
    let airs: Vec<&A> = instances.iter().map(|i| i.air).collect();
    let pub_vals: Vec<Vec<Val<SC>>> = instances.iter().map(|i| i.public_values.clone()).collect();

    let mut preprocessed_widths = Vec::with_capacity(airs.len());
    let (log_num_quotient_chunks, num_quotient_chunks): (Vec<usize>, Vec<usize>) = airs
        .iter()
        .zip(pub_vals.iter())
        .enumerate()
        .map(|(i, (air, pv))| {
            let pre_w = common
                .preprocessed
                .as_ref()
                .and_then(|g| g.instances[i].as_ref().map(|m| m.width))
                .unwrap_or(0);
            preprocessed_widths.push(pre_w);
            let lq_chunks = get_log_num_quotient_chunks::<Val<SC>, SC::Challenge, A, LG>(
                air,
                pre_w,
                pv.len(),
                &all_lookups[i],
                &lookup_data[i],
                config.is_zk(),
                lookup_gadget,
            );
            let n_chunks = 1 << (lq_chunks + config.is_zk());
            (lq_chunks, n_chunks)
        })
        .unzip();

    // Observe the number of instances up front so the transcript can't be reinterpreted
    // with a different partitioning.
    let n_instances = airs.len();
    challenger.observe_base_as_algebra_element::<Challenge<SC>>(Val::<SC>::from_usize(n_instances));

    // Observe per-instance binding data: (log_ext_degree, log_degree), width, num quotient chunks.
    for i in 0..n_instances {
        observe_instance_binding::<SC>(
            &mut challenger,
            log_ext_degrees[i],
            log_degrees[i],
            A::width(airs[i]),
            num_quotient_chunks[i],
        );
    }

    // Commit to all traces using a single batched commitment, preserving input order.
    let main_commit_inputs = instances
        .iter()
        .zip(ext_trace_domains.iter().cloned())
        .map(|(inst, dom)| (dom, inst.trace.clone()))
        .collect::<Vec<_>>();
    let (main_commit, main_data) = pcs.commit(main_commit_inputs);

    // Observe main commitment and all public values (as base field elements).
    challenger.observe(main_commit.clone());
    for pv in &pub_vals {
        challenger.observe_slice(pv);
    }

    // Observe preprocessed widths for each instance, to bind transparent
    // preprocessed columns into the transcript. If a global preprocessed
    // commitment exists, observe it once.
    for &pre_w in preprocessed_widths.iter() {
        challenger.observe_base_as_algebra_element::<Challenge<SC>>(Val::<SC>::from_usize(pre_w));
    }
    if let Some(global) = &common.preprocessed {
        challenger.observe(global.commitment.clone());
    }

    // Sample the lookup challenges.
    let challenges_per_instance =
        get_perm_challenges::<SC, LG>(&mut challenger, &all_lookups, lookup_gadget);

    // Get permutation matrices, if any, along with their associated trace domain
    let mut permutation_commit_inputs = Vec::with_capacity(n_instances);
    instances
        .iter()
        .enumerate()
        .zip(ext_trace_domains.iter().cloned())
        .for_each(|((i, inst), ext_domain)| {
            if !all_lookups[i].is_empty() {
                let generated_perm = lookup_gadget.generate_permutation::<SC>(
                    &inst.trace,
                    &inst.air.preprocessed_trace(),
                    &inst.public_values,
                    &all_lookups[i],
                    &mut lookup_data[i],
                    &challenges_per_instance[i],
                );
                permutation_commit_inputs
                    .push((ext_domain, generated_perm.clone().flatten_to_base()));

                #[cfg(debug_assertions)]
                {
                    use crate::check_constraints::check_constraints;

                    let preprocessed_trace = inst.air.preprocessed_trace();

                    let lookup_constraints_inputs = (
                        all_lookups[i].as_slice(),
                        lookup_data[i].as_slice(),
                        lookup_gadget,
                    );
                    check_constraints(
                        inst.air,
                        &inst.trace,
                        &preprocessed_trace,
                        &generated_perm,
                        &challenges_per_instance[i],
                        &inst.public_values,
                        lookup_constraints_inputs,
                    );
                }
            }
        });

    // Commit to all traces in one multi-matrix commitment, preserving input order.
    let permutation_commit_and_data = if !permutation_commit_inputs.is_empty() {
        let commitment = pcs.commit(permutation_commit_inputs);
        challenger.observe(commitment.0.clone());
        Some(commitment)
    } else {
        None
    };

    // Compute quotient chunk counts and domains per instance inline in the loop below.

    // Get the random alpha to fold constraints.
    let alpha: Challenge<SC> = challenger.sample_algebra_element();

    // Build per-instance quotient domains and values, and split into chunks.
    let mut quotient_chunk_domains: Vec<Domain<SC>> = Vec::new();
    let mut quotient_chunk_mats: Vec<RowMajorMatrix<Val<SC>>> = Vec::new();
    // Track ranges so we can map openings back to instances.
    let mut quotient_chunk_ranges: Vec<(usize, usize)> = Vec::with_capacity(n_instances);

    let mut perm_counter = 0;

    // TODO: Parallelize this loop for better performance with many instances.
    for (i, trace_domain) in trace_domains.iter().enumerate() {
        let log_chunks = log_num_quotient_chunks[i];
        let n_chunks = num_quotient_chunks[i];
        // Disjoint domain of size ext_degree * num_quotient_chunks
        // (log size = log_ext_degrees[i] + log_num_quotient_chunks[i]); use ext domain for shift.
        let quotient_domain =
            ext_trace_domains[i].create_disjoint_domain(1 << (log_ext_degrees[i] + log_chunks));

        // Count constraints to size alpha powers packing.
        let (base_constraints, extension_constraints) = get_symbolic_constraints(
            airs[i],
            preprocessed_widths[i],
            pub_vals[i].len(),
            &all_lookups[i],
            &lookup_data[i],
            lookup_gadget,
        );
        let constraint_len = base_constraints.len() + extension_constraints.len();

        // Get evaluations on quotient domain from the main commitment.
        let trace_on_quotient_domain =
            pcs.get_evaluations_on_domain(&main_data, i, quotient_domain);

        let permutation_on_quotient_domain = permutation_commit_and_data
            .as_ref()
            .filter(|_| !all_lookups[i].is_empty())
            .map(|(_, perm_data)| {
                let evals = pcs.get_evaluations_on_domain(perm_data, perm_counter, quotient_domain);
                perm_counter += 1;
                evals
            });

        // Get preprocessed evaluations if this instance has preprocessed columns.
        let preprocessed_on_quotient_domain = common
            .preprocessed
            .as_ref()
            .and_then(|g| g.instances[i].as_ref().map(|meta| (g, meta)))
            .map(|(g, meta)| {
                pcs.get_evaluations_on_domain(&g.prover_data, meta.matrix_index, quotient_domain)
            });

        // Compute quotient(x) = constraints(x)/Z_H(x) over quotient_domain, as extension values.
        let q_values = quotient_values::<SC, A, _, LG>(
            airs[i],
            &pub_vals[i],
            *trace_domain,
            quotient_domain,
            &trace_on_quotient_domain,
            permutation_on_quotient_domain.as_ref(),
            &all_lookups[i],
            &lookup_data[i],
            lookup_gadget,
            &challenges_per_instance[i],
            preprocessed_on_quotient_domain.as_ref(),
            alpha,
            constraint_len,
        );

        // Flatten to base field and split into chunks.
        let q_flat = RowMajorMatrix::new_col(q_values).flatten_to_base();
        let chunk_mats = quotient_domain.split_evals(n_chunks, q_flat);
        let chunk_domains = quotient_domain.split_domains(n_chunks);

        let start = quotient_chunk_domains.len();
        quotient_chunk_domains.extend(chunk_domains);
        quotient_chunk_mats.extend(chunk_mats);
        let end = quotient_chunk_domains.len();
        quotient_chunk_ranges.push((start, end));
    }

    // Commit to all quotient chunks together.
    let quotient_commit_inputs = quotient_chunk_domains
        .iter()
        .cloned()
        .zip(quotient_chunk_mats.into_iter())
        .collect::<Vec<_>>();
    let (quotient_commit, quotient_data) = pcs.commit(quotient_commit_inputs);
    challenger.observe(quotient_commit.clone());

    // ZK disabled: no randomization round.

    // Sample OOD point.
    let zeta: Challenge<SC> = challenger.sample_algebra_element();

    // Build opening rounds, including optional global preprocessed commitment.
    let (opened_values, opening_proof) = {
        let mut rounds = Vec::new();

        // Main trace round: per instance, open at zeta and its next point.
        let round1_points = ext_trace_domains
            .iter()
            .map(|dom| {
                vec![
                    zeta,
                    dom.next_point(zeta)
                        .expect("domain should support next_point operation"),
                ]
            })
            .collect::<Vec<_>>();
        rounds.push((&main_data, round1_points));

        // Quotient chunks round: one point per chunk at zeta.
        let round2_points = quotient_chunk_ranges
            .iter()
            .cloned()
            .flat_map(|(s, e)| (s..e).map(|_| vec![zeta]))
            .collect::<Vec<_>>();
        rounds.push((&quotient_data, round2_points));

        // Optional global preprocessed round: one matrix per instance that
        // has preprocessed columns.
        if let Some(global) = &common.preprocessed {
            let pre_points = global
                .matrix_to_instance
                .iter()
                .map(|&inst_idx| {
                    let zeta_next_i = ext_trace_domains[inst_idx]
                        .next_point(zeta)
                        .expect("domain should support next_point operation");
                    vec![zeta, zeta_next_i]
                })
                .collect::<Vec<_>>();
            rounds.push((&global.prover_data, pre_points));
        }

        let lookup_points: Vec<_> = ext_trace_domains
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

        pcs.open(rounds, &mut challenger)
    };

    // Rely on PCS indices for opened value groups: main trace, quotient, preprocessed.
    let trace_idx = SC::Pcs::TRACE_IDX;
    let quotient_idx = SC::Pcs::QUOTIENT_IDX;
    let preprocessed_idx = SC::Pcs::PREPROCESSED_TRACE_IDX;
    let permutation_idx = if common.preprocessed.is_some() {
        preprocessed_idx + 1
    } else {
        preprocessed_idx
    };

    // Parse trace opened values per instance.
    let trace_values_for_mats = &opened_values[trace_idx];
    assert_eq!(trace_values_for_mats.len(), n_instances);

    // Parse quotient chunk opened values and map per instance.
    let mut per_instance: Vec<OpenedValuesWithLookups<Challenge<SC>>> =
        Vec::with_capacity(n_instances);

    // Preprocessed openings, if a global preprocessed commitment exists.
    let preprocessed_openings = common
        .preprocessed
        .as_ref()
        .map(|_| &opened_values[SC::Pcs::PREPROCESSED_TRACE_IDX]);

    let is_lookup = permutation_commit_and_data.is_some();
    let permutation_values_for_mats = if is_lookup {
        &opened_values[permutation_idx]
    } else {
        &vec![]
    };
    let mut permutation_values_for_mats = permutation_values_for_mats.iter();

    let mut quotient_openings_iter = opened_values[quotient_idx].iter();
    for (i, (s, e)) in quotient_chunk_ranges.iter().copied().enumerate() {
        // Trace locals
        let tv = &trace_values_for_mats[i];
        let trace_local = tv[0].clone();
        let trace_next = tv[1].clone();

        // Quotient chunks: for each chunk matrix, take the first point (zeta) values.
        let mut qcs = Vec::with_capacity(e - s);
        for _ in s..e {
            let mat_vals = quotient_openings_iter
                .next()
                .expect("chunk index in bounds");
            qcs.push(mat_vals[0].clone());
        }

        // Preprocessed openings (if present).
        let (preprocessed_local, preprocessed_next) = if let (Some(global), Some(pre_round)) =
            (&common.preprocessed, preprocessed_openings)
        {
            global.instances[i].as_ref().map_or((None, None), |meta| {
                let vals = &pre_round[meta.matrix_index];
                assert_eq!(
                    vals.len(),
                    2,
                    "expected two opening points (zeta, zeta_next) for preprocessed trace"
                );
                (Some(vals[0].clone()), Some(vals[1].clone()))
            })
        } else {
            (None, None)
        };

        // Not all AIRs have lookups, so for each instance, we first need to check whether it has lookups.
        let (permutation_local, permutation_next) = if !all_lookups[i].is_empty() {
            let perm_v = permutation_values_for_mats
                .next()
                .expect("instance should have permutation openings");
            (perm_v[0].clone(), perm_v[1].clone())
        } else {
            (vec![], vec![])
        };

        let base_opened = OpenedValues {
            trace_local,
            trace_next,
            preprocessed_local,
            preprocessed_next,
            quotient_chunks: qcs,
            random: None, // ZK not supported in batch-stark yet
        };

        per_instance.push(OpenedValuesWithLookups {
            base_opened_values: base_opened,
            permutation_local,
            permutation_next,
        });
    }

    let permutation = permutation_commit_and_data
        .as_ref()
        .map(|(comm, _)| comm.clone());

    BatchProof {
        commitments: BatchCommitments {
            main: main_commit,
            quotient_chunks: quotient_commit,
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

#[instrument(name = "compute quotient polynomial", skip_all)]
// TODO: Group some arguments to remove the `allow`?
#[allow(clippy::too_many_arguments)]
pub fn quotient_values<SC, A, Mat, LG>(
    air: &A,
    public_values: &[Val<SC>],
    trace_domain: Domain<SC>,
    quotient_domain: Domain<SC>,
    trace_on_quotient_domain: &Mat,
    opt_permutation_on_quotient_domain: Option<&Mat>,
    lookups: &[Lookup<Val<SC>>],
    lookup_data: &[LookupData<SC::Challenge>],
    lookup_gadget: &LG,
    permutation_challenges: &[SC::Challenge],
    preprocessed_on_quotient_domain: Option<&Mat>,
    alpha: SC::Challenge,
    constraint_count: usize,
) -> Vec<SC::Challenge>
where
    SC: SGC,
    A: for<'a> AirLookupHandler<ProverConstraintFolderWithLookups<'a, SC>>,
    Mat: Matrix<Val<SC>> + Sync,
    LG: LookupGadget + Sync,
{
    let quotient_size = quotient_domain.size();
    let main_width = trace_on_quotient_domain.width();
    let (perm_width, perm_height) = opt_permutation_on_quotient_domain
        .as_ref()
        .map_or((0, 0), |mat| (mat.width(), mat.height()));

    let ext_degree = SC::Challenge::DIMENSION;

    let mut sels = debug_span!("Compute Selectors")
        .in_scope(|| trace_domain.selectors_on_coset(quotient_domain));

    let qdb = log2_strict_usize(quotient_domain.size()) - log2_strict_usize(trace_domain.size());
    let next_step = 1 << qdb;

    // Pad selectors with default values if the domain is smaller than the packing width.
    let pack_width = PackedVal::<SC>::WIDTH;
    if quotient_size < pack_width {
        let pad_len = pack_width;
        // Helper to resize a specific selector vector
        let pad = |v: &mut Vec<_>| v.resize(pad_len, Val::<SC>::default());
        pad(&mut sels.is_first_row);
        pad(&mut sels.is_last_row);
        pad(&mut sels.is_transition);
        pad(&mut sels.inv_vanishing);
    }

    let mut alpha_powers = alpha.powers().collect_n(constraint_count);
    alpha_powers.reverse();
    // alpha powers looks like Vec<EF> ~ Vec<[F; D]>
    // It's useful to also have access to the transpose of this of form [Vec<F>; D].
    let decomposed_alpha_powers: Vec<_> = (0..SC::Challenge::DIMENSION)
        .map(|i| {
            alpha_powers
                .iter()
                .map(|x| x.as_basis_coefficients_slice()[i])
                .collect()
        })
        .collect();
    (0..quotient_size)
        .into_par_iter()
        .step_by(PackedVal::<SC>::WIDTH)
        .flat_map_iter(|i_start| {
            let i_range = i_start..i_start + PackedVal::<SC>::WIDTH;

            let is_first_row = *PackedVal::<SC>::from_slice(&sels.is_first_row[i_range.clone()]);
            let is_last_row = *PackedVal::<SC>::from_slice(&sels.is_last_row[i_range.clone()]);
            let is_transition = *PackedVal::<SC>::from_slice(&sels.is_transition[i_range.clone()]);
            let inv_vanishing = *PackedVal::<SC>::from_slice(&sels.inv_vanishing[i_range]);

            // Retrieve main trace as a matrix evaluated on the quotient domain.
            let main = RowMajorMatrix::new(
                trace_on_quotient_domain.vertically_packed_row_pair(i_start, next_step),
                main_width,
            );

            let preprocessed = preprocessed_on_quotient_domain.map(|preprocessed| {
                let preprocessed_width = preprocessed.width();
                RowMajorMatrix::new(
                    preprocessed.vertically_packed_row_pair(i_start, next_step),
                    preprocessed_width,
                )
            });

            // Retrieve permutation trace as a matrix evaluated on the quotient domain.
            let permutation = opt_permutation_on_quotient_domain.as_ref().map_or_else(
                || RowMajorMatrix::new(vec![], 0),
                |permutation_on_quotient_domain| {
                    let perms = (0..perm_width)
                        .step_by(ext_degree)
                        .map(|col| {
                            PackedChallenge::<SC>::from_basis_coefficients_fn(|i| {
                                PackedVal::<SC>::from_fn(|offset| {
                                    permutation_on_quotient_domain
                                        .get((i_start + offset) % perm_height, col + i)
                                        .unwrap()
                                })
                            })
                        })
                        .chain((0..perm_width).step_by(ext_degree).map(|col| {
                            PackedChallenge::<SC>::from_basis_coefficients_fn(|i| {
                                PackedVal::<SC>::from_fn(|offset| {
                                    permutation_on_quotient_domain
                                        .get((i_start + next_step + offset) % perm_height, col + i)
                                        .unwrap()
                                })
                            })
                        }));

                    RowMajorMatrix::new(perms.collect::<Vec<_>>(), perm_width / ext_degree)
                },
            );

            let accumulator = PackedChallenge::<SC>::ZERO;
            let inner_folder = ProverConstraintFolder {
                main: main.as_view(),
                preprocessed: preprocessed.as_ref().map(|m| m.as_view()),
                public_values,
                is_first_row,
                is_last_row,
                is_transition,
                alpha_powers: &alpha_powers,
                decomposed_alpha_powers: &decomposed_alpha_powers,
                accumulator,
                constraint_index: 0,
            };
            let packed_perm_challenges = permutation_challenges
                .iter()
                .map(|p_c| PackedChallenge::<SC>::from(*p_c))
                .collect::<Vec<_>>();

            let mut folder = ProverConstraintFolderWithLookups {
                inner: inner_folder,
                permutation: permutation.as_view(),
                permutation_challenges: &packed_perm_challenges,
            };
            <A as AirLookupHandler<ProverConstraintFolderWithLookups<'_, SC>>>::eval(
                air,
                &mut folder,
                lookups,
                lookup_data,
                lookup_gadget,
            );

            // quotient(x) = constraints(x) / Z_H(x)
            let quotient = folder.inner.accumulator * inv_vanishing;

            // "Transpose" D packed base coefficients into WIDTH scalar extension coefficients.
            (0..core::cmp::min(quotient_size, PackedVal::<SC>::WIDTH)).map(move |idx_in_packing| {
                SC::Challenge::from_basis_coefficients_fn(|coeff_idx| {
                    quotient.as_basis_coefficients_slice()[coeff_idx].as_slice()[idx_in_packing]
                })
            })
        })
        .collect()
}

pub fn prove_batch_no_lookups<
    SC,
    #[cfg(debug_assertions)] A: for<'a> AirLookupHandler<DebugConstraintBuilderWithLookups<'a, Val<SC>, SC::Challenge>>
        + AirLookupHandler<SymbolicAirBuilder<Val<SC>, SC::Challenge>>
        + for<'a> AirLookupHandler<ProverConstraintFolderWithLookups<'a, SC>>
        + Clone,
    #[cfg(not(debug_assertions))] A: for<'a> AirLookupHandler<SymbolicAirBuilder<Val<SC>, SC::Challenge>>
        + for<'a> AirLookupHandler<ProverConstraintFolderWithLookups<'a, SC>>
        + Clone,
>(
    config: &SC,
    instances: &[StarkInstance<'_, SC, A>],
    common_data: &CommonData<SC>,
) -> BatchProof<SC>
where
    SC: SGC,
    SymbolicExpression<SC::Challenge>: From<SymbolicExpression<Val<SC>>>,
{
    let dummy_lookup_gadget = EmptyLookupGadget;
    assert!(common_data.lookups.iter().all(|l| l.is_empty()));
    prove_batch(config, instances, common_data, &dummy_lookup_gadget)
}
