use alloc::vec;
use alloc::vec::Vec;

use crate::config::{Challenge, Domain, SGC, Val, observe_base_as_ext, observe_instance_binding};
use crate::proof::{MultiCommitments, MultiOpenedValues, MultiProof, OpenedValuesWithLookups};
use crate::symbolic::{get_log_quotient_degree, get_symbolic_constraints};
use hashbrown::HashMap;
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::BasedVectorSpace;
use p3_field::PackedValue;
use p3_field::PrimeCharacteristicRing;
use p3_lookup::folders::ProverConstraintFolderWithLookups;
use p3_lookup::lookup_traits::{
    AirLookupHandler, EmptyLookupGadget, Kind, Lookup, LookupData, LookupGadget,
};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::IntoParallelIterator;
use p3_maybe_rayon::prelude::ParIterExt;
use p3_uni_stark::{
    OpenedValues, PackedChallenge, PackedVal, ProverConstraintFolder, SymbolicAirBuilder,
    SymbolicExpression,
};
use p3_util::log2_strict_usize;
use tracing::{debug_span, instrument};

#[derive(Debug, Clone)]
pub struct StarkInstance<'a, SC: SGC, A> {
    pub air: &'a A,
    pub trace: RowMajorMatrix<Val<SC>>,
    pub public_values: Vec<Val<SC>>,
    pub lookups: Vec<Lookup<Val<SC>>>,
}

#[instrument(skip_all)]
pub fn prove_multi<SC, A, LG>(
    config: &SC,
    instances: Vec<StarkInstance<SC, A>>,
    lookup_gadget: &LG,
) -> MultiProof<SC>
where
    SC: SGC,
    SymbolicExpression<SC::Challenge>: From<SymbolicExpression<Val<SC>>>,
    A: AirLookupHandler<SymbolicAirBuilder<Val<SC>, SC::Challenge>>
        + for<'a> AirLookupHandler<ProverConstraintFolderWithLookups<'a, SC>>,
    LG: LookupGadget,
{
    let pcs = config.pcs();
    let mut challenger = config.initialise_challenger();

    // TODO: No ZK support for multi-stark yet.
    if config.is_zk() != 0 {
        panic!("p3-multi-stark: ZK mode is not supported yet");
    }

    // Use instances in provided order.
    let degrees: Vec<usize> = instances.iter().map(|i| i.trace.height()).collect();
    let log_degrees: Vec<usize> = degrees.iter().copied().map(log2_strict_usize).collect();
    let log_ext_degrees: Vec<usize> = log_degrees.iter().map(|&d| d + config.is_zk()).collect();

    // Extract lookups and create lookup data in one pass.
    let (all_lookups, mut lookup_data): (Vec<_>, Vec<_>) = instances
        .iter()
        .map(|inst| {
            let (lookups, data): (Vec<_>, Vec<_>) = (
                inst.lookups.clone(),
                inst.lookups
                    .iter()
                    .map(|lookup| LookupData {
                        aux_idx: lookup.columns[0],
                        expected_cumulated: SC::Challenge::ZERO,
                    })
                    .collect(),
            );
            (lookups, data)
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
    let airs: Vec<_> = instances.iter().map(|inst| inst.air).collect();
    let pub_vals = instances
        .iter()
        .map(|inst| &inst.public_values)
        .collect::<Vec<_>>();

    // Precompute per-instance log_quotient_degrees and quotient_degrees in one pass.
    let (log_quotient_degrees, quotient_degrees): (Vec<usize>, Vec<usize>) = airs
        .iter()
        .zip(pub_vals.iter())
        .enumerate()
        .map(|(i, (air, pv))| {
            let lqd = get_log_quotient_degree::<Val<SC>, SC::Challenge, A, LG>(
                air,
                0,
                pv.len(),
                &all_lookups[i],
                &lookup_data[i],
                config.is_zk(),
                lookup_gadget,
            );
            let qd = 1 << (lqd + config.is_zk());
            (lqd, qd)
        })
        .unzip();

    // Observe the number of instances up front so the transcript can't be reinterpreted
    // with a different partitioning.
    let n_instances = airs.len();
    observe_base_as_ext::<SC>(&mut challenger, Val::<SC>::from_usize(n_instances));

    // Observe per-instance binding data: (log_ext_degree, log_degree), width, num quotient chunks.
    for i in 0..n_instances {
        observe_instance_binding::<SC>(
            &mut challenger,
            log_ext_degrees[i],
            log_degrees[i],
            A::width(airs[i]),
            quotient_degrees[i],
        );
    }

    // Commit to all traces in one multi-matrix commitment, preserving input order.
    let main_commit_inputs = instances
        .iter()
        .zip(ext_trace_domains.iter())
        .map(|(inst, dom)| (*dom, inst.trace.clone()))
        .collect::<Vec<_>>();
    let (main_commit, main_data) = pcs.commit(main_commit_inputs);

    // Observe main commitment and all public values (as base field elements).
    challenger.observe(main_commit.clone());
    for pv in &pub_vals {
        challenger.observe_slice(*pv);
    }

    // Sample the lookup challenges.
    let mut global_perm_challenges = HashMap::new();
    let mut challenges_per_instance = Vec::with_capacity(airs.len());
    for contexts in &all_lookups {
        let num_challenges = contexts.len() * lookup_gadget.num_challenges();
        let mut instance_challenges = Vec::with_capacity(num_challenges);
        for context in contexts {
            let cs = match &context.kind {
                Kind::Global(name) => {
                    let cs = global_perm_challenges.entry(name).or_insert_with(|| {
                        vec![
                            challenger.sample_algebra_element::<Challenge<SC>>(),
                            challenger.sample_algebra_element(),
                        ]
                    });
                    cs.clone()
                }
                Kind::Local => {
                    vec![
                        challenger.sample_algebra_element(),
                        challenger.sample_algebra_element(),
                    ]
                }
            };
            instance_challenges.extend(cs);
        }
        challenges_per_instance.push(instance_challenges);
    }

    // Get permutation matrices, if any, along with their associated trace domain
    let mut permutation_commit_inputs = Vec::with_capacity(n_instances);
    instances
        .iter()
        .enumerate()
        .zip(ext_trace_domains.iter().cloned())
        .for_each(|((i, inst), ext_domain)| {
            if !all_lookups[i].is_empty() {
                permutation_commit_inputs.push((
                    ext_domain,
                    lookup_gadget
                        .generate_permutation(
                            &inst.trace,
                            &all_lookups[i],
                            &mut lookup_data[i],
                            &challenges_per_instance[i],
                        )
                        .flatten_to_base(),
                ))
            }
        });

    // Commit to all traces in one multi-matrix commitment, preserving input order.
    let opt_permutation_commit_and_data = if !permutation_commit_inputs.is_empty() {
        Some(pcs.commit(permutation_commit_inputs))
    } else {
        None
    };

    // Compute quotient degrees and domains per instance inline in the loop below.

    // Get the random alpha to fold constraints.
    let alpha: Challenge<SC> = challenger.sample_algebra_element();

    // Build per-instance quotient domains and values, and split into chunks.
    let mut quotient_chunk_domains: Vec<Domain<SC>> = Vec::new();
    let mut quotient_chunk_mats: Vec<RowMajorMatrix<Val<SC>>> = Vec::new();
    // Track ranges so we can map openings back to instances.
    let mut quotient_chunk_ranges: Vec<(usize, usize)> = Vec::with_capacity(n_instances);

    // TODO: Parallelize this loop for better performance with multiple instances.
    for (i, trace_domain) in trace_domains.iter().enumerate() {
        let lqd = log_quotient_degrees[i];
        let quotient_degree = quotient_degrees[i];
        // Disjoint domain sized by extended degree + quotient degree; use ext domain for shift.
        let quotient_domain =
            ext_trace_domains[i].create_disjoint_domain(1 << (log_ext_degrees[i] + lqd));

        // Count constraints to size alpha powers packing.
        let (base_constraints, extension_constraints) = get_symbolic_constraints(
            airs[i],
            0,
            pub_vals[i].len(),
            &all_lookups[i],
            &lookup_data[i],
            lookup_gadget,
        );
        let constraint_cnt = base_constraints.len() + extension_constraints.len();

        // Get evaluations on quotient domain from the main commitment.
        let trace_on_quotient_domain =
            pcs.get_evaluations_on_domain(&main_data, i, quotient_domain);
        let permutation_on_quotient_domain =
            if let Some((_, perm_data)) = &opt_permutation_commit_and_data {
                Some(pcs.get_evaluations_on_domain(perm_data, i, quotient_domain))
            } else {
                None
            };

        // Compute quotient(x) = constraints(x)/Z_H(x) over quotient_domain, as extension values.
        let q_values = quotient_values::<SC, A, _>(
            airs[i],
            &pub_vals[i],
            *trace_domain,
            quotient_domain,
            trace_on_quotient_domain,
            permutation_on_quotient_domain,
            &all_lookups[i],
            &lookup_data[i],
            lookup_gadget,
            &challenges_per_instance[i],
            alpha,
            constraint_cnt,
        );

        // Flatten to base field and split into chunks.
        let q_flat = RowMajorMatrix::new_col(q_values).flatten_to_base();
        let chunk_mats = quotient_domain.split_evals(quotient_degree, q_flat);
        let chunk_domains = quotient_domain.split_domains(quotient_degree);

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

    let mut rounds = vec![];
    // Build opening rounds.
    // TODO: Add opening round for lookup aux columns.
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
    let round1 = (&main_data, round1_points.clone());
    rounds.push(round1);

    if let Some((_, perm_data)) = &opt_permutation_commit_and_data {
        let round2 = (perm_data, round1_points);
        rounds.push(round2);
    }

    let round3_points = quotient_chunk_ranges
        .iter()
        .cloned()
        .flat_map(|(s, e)| (s..e).map(|_| vec![zeta]))
        .collect::<Vec<_>>();
    let round3 = (&quotient_data, round3_points);
    rounds.push(round3);

    let (opened_values, opening_proof) = pcs.open(rounds, &mut challenger);
    assert_eq!(
        opened_values.len(),
        2,
        "expected [main, quotient] opening groups from PCS"
    );

    let is_lookup = opt_permutation_commit_and_data.is_some();
    // Rely on open order: [main, quotient] since ZK is disabled.
    let trace_idx = 0usize;
    let permutation_idx = 1usize;
    let quotient_idx = if is_lookup { 2usize } else { 1usize };

    // Parse trace opened values per instance.
    let trace_values_for_mats = &opened_values[trace_idx];
    assert_eq!(trace_values_for_mats.len(), n_instances);

    let permutation_values_for_mats = if is_lookup {
        &opened_values[permutation_idx]
    } else {
        &vec![]
    };
    let mut permutation_values_for_mats = permutation_values_for_mats.iter();

    // Parse quotient chunk opened values and map per instance.
    let mut quotient_openings_iter = opened_values[quotient_idx].iter();

    let mut per_instance: Vec<OpenedValuesWithLookups<Challenge<SC>>> =
        Vec::with_capacity(n_instances);
    for (i, (s, e)) in quotient_chunk_ranges.iter().copied().enumerate() {
        // Trace locals
        let tv = &trace_values_for_mats[i];

        // Not all AIRs have lookups, so for each instance, we first need to check whether it has lookups.
        let (permutation_local, permutation_next) = if !all_lookups[i].is_empty() {
            let perm_v = permutation_values_for_mats
                .next()
                .expect("permutation openings in bounds");
            (perm_v[0].clone(), perm_v[1].clone())
        } else {
            (vec![], vec![])
        };
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

        let base_opened = OpenedValues {
            trace_local,
            trace_next,
            quotient_chunks: qcs,
            random: None, // ZK not supported in multi-stark yet
        };

        per_instance.push(OpenedValuesWithLookups {
            base_opened_values: base_opened,
            permutation_local,
            permutation_next,
        });
    }

    let opt_permutation_commit = opt_permutation_commit_and_data
        .as_ref()
        .map(|(comm, _)| comm.clone());
    MultiProof {
        commitments: MultiCommitments {
            main: main_commit,
            quotient_chunks: quotient_commit,
            permutation: opt_permutation_commit,
        },
        opened_values: MultiOpenedValues {
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
pub fn quotient_values<SC, A, Mat>(
    air: &A,
    public_values: &Vec<Val<SC>>,
    trace_domain: Domain<SC>,
    quotient_domain: Domain<SC>,
    trace_on_quotient_domain: Mat,
    opt_permutation_on_quotient_domain: Option<Mat>,
    lookups: &[Lookup<Val<SC>>],
    lookup_data: &[LookupData<SC::Challenge>],
    lookup_gadget: &impl LookupGadget,
    permutation_challenges: &[SC::Challenge],
    alpha: SC::Challenge,
    constraint_count: usize,
) -> Vec<SC::Challenge>
where
    SC: SGC,
    A: for<'a> AirLookupHandler<ProverConstraintFolderWithLookups<'a, SC>>,
    Mat: Matrix<Val<SC>> + Sync,
{
    let quotient_size = quotient_domain.size();
    let width = trace_on_quotient_domain.width();
    let (perm_width, perm_height) = match &opt_permutation_on_quotient_domain {
        Some(mat) => (mat.width(), mat.height()),
        None => (0, 0),
    };

    let mut sels = debug_span!("Compute Selectors")
        .in_scope(|| trace_domain.selectors_on_coset(quotient_domain));

    let qdb = log2_strict_usize(quotient_domain.size()) - log2_strict_usize(trace_domain.size());
    let next_step = 1 << qdb;

    // We take PackedVal::<SC>::WIDTH worth of values at a time from a quotient_size slice, so we need to
    // pad with default values in the case where quotient_size is smaller than PackedVal::<SC>::WIDTH.
    for _ in quotient_size..PackedVal::<SC>::WIDTH {
        sels.is_first_row.push(Val::<SC>::default());
        sels.is_last_row.push(Val::<SC>::default());
        sels.is_transition.push(Val::<SC>::default());
        sels.inv_vanishing.push(Val::<SC>::default());
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
                width,
            );

            // Retrieve permutation trace as a matrix evaluated on the quotient domain.
            let permutation = if let Some(permutation_on_quotient_domain) =
                &opt_permutation_on_quotient_domain
            {
                RowMajorMatrix::new(
                    // Local permutation evaluation.
                    (0..perm_width)
                        .step_by(Challenge::<SC>::DIMENSION)
                        .map(|col_idx| {
                            PackedChallenge::<SC>::from_basis_coefficients_fn(|coeff_idx| {
                                PackedVal::<SC>::from_fn(|i| {
                                    permutation_on_quotient_domain
                                        .get((i_start + i) % perm_height, col_idx + coeff_idx)
                                        .unwrap()
                                })
                            })
                        })
                        .chain(
                            // Next permutation evaluation.
                            (0..perm_width)
                                .step_by(Challenge::<SC>::DIMENSION)
                                .map(|col_idx| {
                                    PackedChallenge::<SC>::from_basis_coefficients_fn(|coeff_idx| {
                                        PackedVal::<SC>::from_fn(|i| {
                                            permutation_on_quotient_domain
                                                .get(
                                                    (i_start + next_step + i) % perm_height,
                                                    col_idx + coeff_idx,
                                                )
                                                .unwrap()
                                        })
                                    })
                                }),
                        )
                        .collect::<Vec<_>>(),
                    perm_width,
                )
            } else {
                RowMajorMatrix::new(vec![], 0)
            };

            let accumulator = PackedChallenge::<SC>::ZERO;
            let base_folder = ProverConstraintFolder {
                main: main.as_view(),
                public_values,
                is_first_row,
                is_last_row,
                is_transition,
                alpha_powers: &alpha_powers,
                decomposed_alpha_powers: &decomposed_alpha_powers,
                accumulator,
                constraint_index: 0,
            };
            let mut folder = ProverConstraintFolderWithLookups {
                base: base_folder,
                permutation: permutation.as_view(),
                permutation_challenges,
            };
            <A as AirLookupHandler<ProverConstraintFolderWithLookups<'_, SC>>>::eval(
                &air,
                &mut folder,
                &lookups,
                lookup_data,
                lookup_gadget,
            );

            // quotient(x) = constraints(x) / Z_H(x)
            let quotient = folder.base.accumulator * inv_vanishing;

            // "Transpose" D packed base coefficients into WIDTH scalar extension coefficients.
            (0..core::cmp::min(quotient_size, PackedVal::<SC>::WIDTH)).map(move |idx_in_packing| {
                SC::Challenge::from_basis_coefficients_fn(|coeff_idx| {
                    quotient.as_basis_coefficients_slice()[coeff_idx].as_slice()[idx_in_packing]
                })
            })
        })
        .collect()
}

pub fn prove_multi_no_lookups<SC, A>(
    config: &SC,
    instances: Vec<StarkInstance<SC, A>>,
) -> MultiProof<SC>
where
    SC: SGC,
    SymbolicExpression<SC::Challenge>: From<SymbolicExpression<Val<SC>>>,
    A: AirLookupHandler<SymbolicAirBuilder<Val<SC>, SC::Challenge>>
        + for<'a> AirLookupHandler<ProverConstraintFolderWithLookups<'a, SC>>,
{
    let dummy_lookup_gadget = EmptyLookupGadget;
    prove_multi(config, instances, &dummy_lookup_gadget)
}
