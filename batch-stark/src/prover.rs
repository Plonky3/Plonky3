use alloc::vec;
use alloc::vec::Vec;

use p3_air::Air;
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{
    OpenedValues, ProverConstraintFolder, SymbolicAirBuilder, get_log_quotient_degree,
    get_symbolic_constraints, quotient_values,
};
use p3_util::log2_strict_usize;
use tracing::instrument;

use crate::common::CommonData;
use crate::config::{Challenge, Domain, StarkGenericConfig as SGC, Val, observe_instance_binding};
use crate::proof::{BatchCommitments, BatchOpenedValues, BatchProof};

#[derive(Debug)]
pub struct StarkInstance<'a, SC: SGC, A> {
    pub air: &'a A,
    pub trace: RowMajorMatrix<Val<SC>>,
    pub public_values: Vec<Val<SC>>,
}

#[instrument(skip_all)]
pub fn prove_batch<SC, A>(
    config: &SC,
    instances: Vec<StarkInstance<'_, SC, A>>,
    common: &CommonData<SC>,
) -> BatchProof<SC>
where
    SC: SGC,
    A: Air<SymbolicAirBuilder<Val<SC>>> + for<'a> Air<ProverConstraintFolder<'a, SC>>,
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

    // Precompute per-instance preprocessed widths, log_quotient_degrees and quotient_degrees.
    let mut preprocessed_widths = Vec::with_capacity(airs.len());
    let (log_quotient_degrees, quotient_degrees): (Vec<usize>, Vec<usize>) = airs
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
            let lqd = get_log_quotient_degree::<Val<SC>, A>(air, pre_w, pv.len(), config.is_zk());
            let qd = 1 << (lqd + config.is_zk());
            (lqd, qd)
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
            quotient_degrees[i],
        );
    }

    // Commit to all traces using a single batched commitment, preserving input order.
    let main_commit_inputs = instances
        .into_iter()
        .zip(ext_trace_domains.iter().cloned())
        .map(|(inst, dom)| (dom, inst.trace))
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

    // Compute quotient degrees and domains per instance inline in the loop below.

    // Get the random alpha to fold constraints.
    let alpha: Challenge<SC> = challenger.sample_algebra_element();

    // Build per-instance quotient domains and values, and split into chunks.
    let mut quotient_chunk_domains: Vec<Domain<SC>> = Vec::new();
    let mut quotient_chunk_mats: Vec<RowMajorMatrix<Val<SC>>> = Vec::new();
    // Track ranges so we can map openings back to instances.
    let mut quotient_chunk_ranges: Vec<(usize, usize)> = Vec::with_capacity(n_instances);

    // TODO: Parallelize this loop for better performance with many instances.
    for (i, trace_domain) in trace_domains.iter().enumerate() {
        let lqd = log_quotient_degrees[i];
        let quotient_degree = quotient_degrees[i];
        // Disjoint domain sized by extended degree + quotient degree; use ext domain for shift.
        let quotient_domain =
            ext_trace_domains[i].create_disjoint_domain(1 << (log_ext_degrees[i] + lqd));

        // Count constraints to size alpha powers packing.
        let constraint_cnt =
            get_symbolic_constraints(airs[i], preprocessed_widths[i], pub_vals[i].len()).len();

        // Get evaluations on quotient domain from the main commitment.
        let trace_on_quotient_domain =
            pcs.get_evaluations_on_domain(&main_data, i, quotient_domain);

        // Get preprocessed evaluations if this instance has preprocessed columns.
        let preprocessed_on_quotient_domain = common
            .preprocessed
            .as_ref()
            .and_then(|g| g.instances[i].as_ref().map(|meta| (g, meta)))
            .map(|(g, meta)| {
                pcs.get_evaluations_on_domain(&g.prover_data, meta.matrix_index, quotient_domain)
            });

        // Compute quotient(x) = constraints(x)/Z_H(x) over quotient_domain, as extension values.
        let q_values = quotient_values::<SC, A, _>(
            airs[i],
            &pub_vals[i],
            *trace_domain,
            quotient_domain,
            &trace_on_quotient_domain,
            preprocessed_on_quotient_domain.as_ref(),
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

        pcs.open(rounds, &mut challenger)
    };

    // Rely on open order: [main, quotient, preprocessed?]
    let trace_idx = 0usize;
    let quotient_idx = 1usize;

    // Parse trace opened values per instance.
    let trace_values_for_mats = &opened_values[trace_idx];
    assert_eq!(trace_values_for_mats.len(), n_instances);

    // Parse quotient chunk opened values and map per instance.
    let mut per_instance: Vec<OpenedValues<Challenge<SC>>> = Vec::with_capacity(n_instances);

    // Preprocessed openings, if a global preprocessed commitment exists.
    let preprocessed_openings = if common.preprocessed.is_some() {
        Some(&opened_values[2])
    } else {
        None
    };

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

        per_instance.push(OpenedValues {
            trace_local,
            trace_next,
            preprocessed_local,
            preprocessed_next,
            quotient_chunks: qcs,
            random: None, // ZK not supported in batch-stark yet
        });
    }

    BatchProof {
        commitments: BatchCommitments {
            main: main_commit,
            quotient_chunks: quotient_commit,
        },
        opened_values: BatchOpenedValues {
            instances: per_instance,
        },
        opening_proof,
        degree_bits: log_ext_degrees,
    }
}
