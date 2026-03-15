use alloc::vec;
use alloc::vec::Vec;

#[cfg(debug_assertions)]
use p3_air::DebugConstraintBuilder;
use p3_air::symbolic::{AirLayout, SymbolicAirBuilder, SymbolicExpressionExt};
use p3_air::{Air, RowWindow};
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::{
    Algebra, BasedVectorSpace, PackedFieldExtension, PackedValue, PrimeCharacteristicRing,
};
use p3_lookup::AirWithLookups;
use p3_lookup::folder::ProverConstraintFolderWithLookups;
use p3_lookup::logup::LogUpGadget;
use p3_lookup::lookup_traits::{Kind, Lookup, LookupData, LookupGadget};
use p3_matrix::Matrix;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_maybe_rayon::prelude::*;
use p3_uni_stark::{OpenedValues, PackedChallenge, PackedVal, ProverConstraintFolder};
use p3_util::log2_strict_usize;
use tracing::{debug_span, info_span, instrument};

use crate::common::{CommonData, ProverData, get_perm_challenges};
use crate::config::{Challenge, Domain, StarkGenericConfig as SGC, Val, observe_instance_binding};
use crate::proof::{BatchCommitments, BatchOpenedValues, BatchProof, OpenedValuesWithLookups};
use crate::symbolic::{
    get_constraint_layout, get_log_num_quotient_chunks, get_symbolic_constraints,
};

#[derive(Debug)]
pub struct StarkInstance<'a, SC: SGC, A> {
    pub air: &'a A,
    pub trace: &'a RowMajorMatrix<Val<SC>>,
    pub public_values: Vec<Val<SC>>,
    pub lookups: Vec<Lookup<Val<SC>>>,
}

impl<'a, SC: SGC, A> StarkInstance<'a, SC, A> {
    pub fn new_multiple(
        airs: &'a [A],
        traces: &'a [&'a RowMajorMatrix<Val<SC>>],
        public_values: &[Vec<Val<SC>>],
        common_data: &CommonData<SC>,
    ) -> Vec<Self> {
        airs.iter()
            .zip(traces.iter())
            .zip(public_values.iter())
            .zip(common_data.lookups.iter())
            .map(|(((air, trace), public_values), lookups)| Self {
                air,
                trace,
                public_values: public_values.clone(),
                lookups: lookups.clone(),
            })
            .collect()
    }
}

#[instrument(skip_all)]
pub fn prove_batch<
    SC,
    #[cfg(debug_assertions)] A: for<'a> Air<DebugConstraintBuilder<'a, Val<SC>, SC::Challenge>>
        + Air<SymbolicAirBuilder<Val<SC>, SC::Challenge>>
        + for<'a> Air<ProverConstraintFolderWithLookups<'a, SC>>
        + Clone,
    #[cfg(not(debug_assertions))] A: for<'a> Air<SymbolicAirBuilder<Val<SC>, SC::Challenge>>
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
    let mut challenger = config.initialise_challenger();

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
        .map(|(i, (air, _pv))| {
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
                ..Default::default()
            };
            let lq_chunks =
                info_span!("infer log of constraint degree", air_idx = i).in_scope(|| {
                    get_log_num_quotient_chunks::<Val<SC>, SC::Challenge, A, LogUpGadget>(
                        air,
                        layout,
                        &all_lookups[i],
                        config.is_zk(),
                        &lookup_gadget,
                    )
                });
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
        get_perm_challenges::<SC, LogUpGadget>(&mut challenger, &all_lookups, &lookup_gadget);

    // Get permutation matrices, if any, along with their associated trace domain
    let mut permutation_commit_inputs = Vec::with_capacity(n_instances);
    instances
        .iter()
        .enumerate()
        .zip(ext_trace_domains.iter().cloned())
        .for_each(|((i, inst), ext_domain)| {
            if !all_lookups[i].is_empty() {
                let generated_perm = lookup_gadget.generate_permutation::<SC>(
                    inst.trace,
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

                    let perm_vals: Vec<SC::Challenge> = lookup_data[i]
                        .iter()
                        .map(|ld| ld.expected_cumulated)
                        .collect();
                    let lookup_constraints_inputs = (all_lookups[i].as_slice(), &lookup_gadget);
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
            }
        });

    // Check that all lookups are balanced.
    #[cfg(debug_assertions)]
    {
        use p3_lookup::debug_util::{LookupDebugInstance, check_lookups};

        let preprocessed_traces: Vec<Option<RowMajorMatrix<Val<SC>>>> = instances
            .iter()
            .map(|inst| inst.air.preprocessed_trace())
            .collect();
        let debug_instances: Vec<LookupDebugInstance<'_, Val<SC>>> = instances
            .iter()
            .zip(preprocessed_traces.iter())
            .map(|(inst, prep)| LookupDebugInstance {
                main_trace: inst.trace,
                preprocessed_trace: prep,
                public_values: &inst.public_values,
                lookups: &inst.lookups,
                permutation_challenges: &[],
            })
            .collect();
        check_lookups(&debug_instances);
    }

    // Commit to all traces in one multi-matrix commitment, preserving input order.
    let permutation_commit_and_data = if !permutation_commit_inputs.is_empty() {
        let commitment = pcs.commit(permutation_commit_inputs);
        challenger.observe(commitment.0.clone());
        for data in lookup_data.iter().flatten() {
            challenger.observe_algebra_element(data.expected_cumulated);
        }
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
        let _air_span = info_span!("compute quotient", air_idx = i).entered();

        let log_chunks = log_num_quotient_chunks[i];
        let n_chunks = num_quotient_chunks[i];
        // Disjoint domain of size ext_degree * num_quotient_chunks
        // (log size = log_ext_degrees[i] + log_num_quotient_chunks[i]); use ext domain for shift.
        let quotient_domain =
            ext_trace_domains[i].create_disjoint_domain(1 << (log_ext_degrees[i] + log_chunks));

        let sym_layout = AirLayout {
            preprocessed_width: preprocessed_widths[i],
            main_width: airs[i].width(),
            num_public_values: airs[i].num_public_values(),
            ..Default::default()
        };

        // In debug builds, cross-check the static hint against symbolic evaluation.
        debug_assert!(
            airs[i].num_constraints().is_none_or(|n| {
                n == get_symbolic_constraints(airs[i], sym_layout, &all_lookups[i], &lookup_gadget)
                    .0
                    .len()
            }),
            "num_constraints() = {} but symbolic evaluation found {} base constraints",
            airs[i].num_constraints().unwrap(),
            get_symbolic_constraints(airs[i], sym_layout, &all_lookups[i], &lookup_gadget,)
                .0
                .len(),
        );

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

        // Compute quotient(x) = constraints(x)/Z_H(x) over quotient_domain, as extension values.
        let perm_vals: Vec<SC::Challenge> = lookup_data[i]
            .iter()
            .map(|ld| ld.expected_cumulated)
            .collect();
        let q_values = quotient_values::<SC, A, _, LogUpGadget>(
            airs[i],
            &pub_vals[i],
            sym_layout,
            *trace_domain,
            quotient_domain,
            &trace_on_quotient_domain,
            permutation_on_quotient_domain.as_ref(),
            &all_lookups[i],
            &perm_vals,
            &lookup_gadget,
            &challenges_per_instance[i],
            preprocessed_on_quotient_domain.as_ref(),
            alpha,
        );

        // Flatten to base field and split into chunks.
        let q_flat = RowMajorMatrix::new_col(q_values).flatten_to_base();
        let chunk_mats = quotient_domain.split_evals(n_chunks, q_flat);
        let chunk_domains = quotient_domain.split_domains(n_chunks);

        let evals = chunk_domains.iter().zip(chunk_mats).map(|(d, m)| (*d, m));
        let ldes = pcs.get_quotient_ldes(evals, n_chunks);

        let start = quotient_chunk_domains.len();
        quotient_chunk_domains.extend(chunk_domains);
        quotient_chunk_mats.extend(ldes);
        let end = quotient_chunk_domains.len();
        quotient_chunk_ranges.push((start, end));
    }

    // Commit to all quotient chunks together.
    let (quotient_commit, quotient_data) = pcs.commit_ldes(quotient_chunk_mats);
    challenger.observe(quotient_commit.clone());

    // If zk is enabled, we generate random extension field values of the size of the randomized trace. If `n` is the degree of the initial trace,
    // then the randomized trace has degree `2n`. To randomize the FRI batch polynomial, we then need an extension field random polynomial of degree `2n -1`.
    // So we can generate a random polynomial of degree `2n`, and provide it to `open` as is.
    // Then the method will add `(R(X) - R(z)) / (X - z)` (which is of the desired degree `2n - 1`), to the batch of polynomials.
    // Since we need a random polynomial defined over the extension field, and the `commit` method is over the base field,
    // we actually need to commit to `SC::Challenge::D` base field random polynomials.
    // This is similar to what is done for the quotient polynomials.
    // TODO: This approach is only statistically zk. To make it perfectly zk, `R` would have to truly be an extension field polynomial.
    let (opt_r_commit, opt_r_data) = if SC::Pcs::ZK {
        let (r_commit, r_data) = pcs
            .get_opt_randomization_poly_commitment(ext_trace_domains.iter().copied())
            .expect("ZK is enabled, so we should have randomization commitments");
        (Some(r_commit), Some(r_data))
    } else {
        (None, None)
    };

    if let Some(r_commit) = &opt_r_commit {
        challenger.observe(r_commit.clone());
    }

    // Sample OOD point.
    let zeta: Challenge<SC> = challenger.sample_algebra_element();

    // Build opening rounds, including optional global preprocessed commitment.
    let (opened_values, opening_proof) = {
        let mut rounds = Vec::new();

        let round0 = opt_r_data.as_ref().map(|r_data| {
            let round0_points = trace_domains.iter().map(|_| vec![zeta]).collect::<Vec<_>>();
            (r_data, round0_points)
        });
        rounds.extend(round0);
        // Main trace round: per instance, open at zeta and (conditionally) its next point.
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

        // Quotient chunks round: one point per chunk at zeta.
        let round2_points = quotient_chunk_ranges
            .iter()
            .cloned()
            .flat_map(|(s, e)| (s..e).map(|_| vec![zeta]))
            .collect::<Vec<_>>();
        rounds.push((&quotient_data, round2_points));

        // Optional global preprocessed round: one matrix per instance that
        // has preprocessed columns. For AIRs that only use the local
        // preprocessed row, we open only at `zeta`; otherwise we open at
        // both `zeta` and `zeta_next`.
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
                .collect::<Vec<_>>();
            rounds.push((preprocessed_prover_data, pre_points));
        }

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

        pcs.open_with_preprocessing(rounds, &mut challenger, common.preprocessed.is_some())
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
        let random = if opt_r_data.is_some() {
            Some(opened_values[0][i][0].clone())
        } else {
            None
        };
        // Trace locals
        let tv = &trace_values_for_mats[i];
        let trace_local = tv[0].clone();
        let trace_next = if !airs[i].main_next_row_columns().is_empty() {
            Some(tv[1].clone())
        } else {
            None
        };

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
            random,
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

#[instrument(name = "compute quotient polynomial", skip_all)]
// TODO: Group some arguments to remove the `allow`?
#[allow(clippy::too_many_arguments)]
pub fn quotient_values<SC, A, Mat, LG>(
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
    A: Air<SymbolicAirBuilder<Val<SC>, SC::Challenge>>
        + for<'a> Air<ProverConstraintFolderWithLookups<'a, SC>>,
    Mat: Matrix<Val<SC>> + Sync,
    LG: LookupGadget + Sync,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>: Algebra<SC::Challenge>,
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

    let constraint_layout = get_constraint_layout(air, layout, lookups, lookup_gadget);
    let (base_alpha_powers, ext_alpha_powers) = constraint_layout.decompose_alpha(alpha);

    // Precompute per-instance data used by the hot inner loop to avoid repeated allocations.
    let packed_perm_challenges: Vec<PackedChallenge<SC>> = permutation_challenges
        .iter()
        .map(|&p_c| PackedChallenge::<SC>::from(p_c))
        .collect();
    let permutation_vals_packed: Vec<PackedChallenge<SC>> = permutation_vals
        .iter()
        .map(|&v| PackedChallenge::<SC>::from(v))
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

            let preprocessed_view = preprocessed
                .as_ref()
                .map_or_else(|| RowMajorMatrixView::new(&[], 0), |m| m.as_view());
            let inner_folder = ProverConstraintFolder {
                main: main.as_view(),
                preprocessed: preprocessed_view,
                preprocessed_window: RowWindow::from_view(&preprocessed_view),
                public_values,
                is_first_row,
                is_last_row,
                is_transition,
                base_alpha_powers: &base_alpha_powers,
                ext_alpha_powers: &ext_alpha_powers,
                base_constraints: Vec::with_capacity(constraint_layout.base_indices.len()),
                ext_constraints: Vec::with_capacity(constraint_layout.ext_indices.len()),
                constraint_index: 0,
                constraint_count: constraint_layout.total_constraints(),
            };

            let mut folder = ProverConstraintFolderWithLookups {
                inner: inner_folder,
                permutation: permutation.as_view(),
                permutation_challenges: &packed_perm_challenges,
                permutation_values: &permutation_vals_packed,
            };
            air.eval_with_lookups(&mut folder, lookups, lookup_gadget);

            // quotient(x) = constraints(x) / Z_H(x)
            let quotient = folder.inner.finalize_constraints() * inv_vanishing;

            // "Transpose" D packed base coefficients into WIDTH scalar extension coefficients.
            (0..core::cmp::min(quotient_size, PackedVal::<SC>::WIDTH))
                .map(move |idx_in_packing| quotient.extract(idx_in_packing))
        })
        .collect()
}
