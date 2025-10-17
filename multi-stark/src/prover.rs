use alloc::vec;
use alloc::vec::Vec;

use p3_air::Air;
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::{BasedVectorSpace, PackedValue, PrimeCharacteristicRing};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use p3_uni_stark::{
    ProverConstraintFolder, SymbolicAirBuilder, get_log_quotient_degree, get_symbolic_constraints,
};
use p3_util::log2_strict_usize;
use tracing::instrument;

use crate::config::{Domain, MultiStarkGenericConfig as MSGC, PackedChallenge, PackedVal, Val};
use crate::proof::{InstanceOpenedValues, MultiCommitments, MultiOpenedValues, MultiProof};

#[derive(Debug)]
pub struct StarkInstance<'a, SC: MSGC, A> {
    pub air: &'a A,
    pub trace: RowMajorMatrix<Val<SC>>,
    pub public_values: Vec<Val<SC>>,
}

#[instrument(skip_all)]
#[allow(clippy::multiple_bound_locations)]
pub fn prove_multi<
    SC,
    #[cfg(debug_assertions)] A: for<'a> Air<p3_uni_stark::DebugConstraintBuilder<'a, Val<SC>>>,
    #[cfg(not(debug_assertions))] A,
>(
    config: &SC,
    instances: Vec<StarkInstance<SC, A>>,
) -> MultiProof<SC>
where
    SC: MSGC,
    A: Air<SymbolicAirBuilder<Val<SC>>> + for<'a> Air<ProverConstraintFolder<'a, SC>>,
{
    // Optional constraint checks in debug.
    // Optionally check constraints in debug builds (skipped here; uni-stark provides test coverage)

    let pcs = config.pcs();
    let mut challenger = config.initialise_challenger();

    // No ZK support in initial implementation.
    if config.is_zk() != 0 {
        panic!("p3-multi-stark: ZK mode is not supported yet");
    }

    // Use instances in provided order.
    let degrees: Vec<usize> = instances.iter().map(|i| i.trace.height()).collect();
    let log_degrees: Vec<usize> = degrees.iter().copied().map(log2_strict_usize).collect();
    let log_ext_degrees: Vec<usize> = log_degrees.iter().map(|&d| d + config.is_zk()).collect();

    // Trace domains for each instance.
    let trace_domains: Vec<Domain<SC>> = degrees
        .iter()
        .map(|&deg| pcs.natural_domain_for_degree(deg))
        .collect();
    // Extended domains (equal to base when ZK=0).
    let ext_trace_domains: Vec<Domain<SC>> = degrees
        .iter()
        .map(|&deg| pcs.natural_domain_for_degree(deg * (config.is_zk() + 1)))
        .collect();

    // Extract AIRs and public values; consume traces later without cloning.
    let airs: Vec<&A> = instances.iter().map(|i| i.air).collect();
    let pub_vals: Vec<Vec<Val<SC>>> = instances.iter().map(|i| i.public_values.clone()).collect();

    // Precompute per-instance log quotient degrees for binding and chunking.
    let log_quotient_degrees: Vec<usize> = airs
        .iter()
        .zip(pub_vals.iter())
        .map(|(air, pv)| get_log_quotient_degree::<Val<SC>, A>(air, 0, pv.len(), config.is_zk()))
        .collect();

    // Observe the number of instances up front so the transcript can't be reinterpreted
    // with a different partitioning.
    let n_instances = airs.len();
    challenger.observe(Val::<SC>::from_usize(n_instances));

    // Observe per-instance binding data: (log_ext_degree, log_degree), width, num public values, num quotient chunks.
    for i in 0..n_instances {
        let log_deg = log_degrees[i];
        let log_ext_deg = log_ext_degrees[i];
        challenger.observe(Val::<SC>::from_usize(log_ext_deg));
        challenger.observe(Val::<SC>::from_usize(log_deg));
        let width = A::width(airs[i]);
        challenger.observe(Val::<SC>::from_usize(width));
        let pv_len = pub_vals[i].len();
        challenger.observe(Val::<SC>::from_usize(pv_len));
        let num_chunks = 1 << (log_quotient_degrees[i] + config.is_zk());
        challenger.observe(Val::<SC>::from_usize(num_chunks));
    }

    // Commit to all traces in one multi-matrix commitment, preserving input order.
    let main_commit_inputs = instances
        .into_iter()
        .zip(ext_trace_domains.iter().cloned())
        .map(|(inst, dom)| (dom, inst.trace))
        .collect::<Vec<_>>();
    let (main_commit, main_data) = pcs.commit(main_commit_inputs);

    // Observe main commitment and all public values.
    challenger.observe(main_commit.clone());
    for pv in &pub_vals {
        challenger.observe_slice(pv);
    }

    // Compute quotient degrees and domains per instance inline in the loop below.

    // Get the random alpha to fold constraints.
    let alpha: <SC as p3_uni_stark::StarkGenericConfig>::Challenge =
        challenger.sample_algebra_element();

    // Build per-instance quotient domains and values, and split into chunks.
    let mut quotient_chunk_domains: Vec<Domain<SC>> = Vec::new();
    let mut quotient_chunk_mats: Vec<RowMajorMatrix<Val<SC>>> = Vec::new();
    // Track ranges so we can map openings back to instances.
    let mut quotient_chunk_ranges: Vec<(usize, usize)> = Vec::with_capacity(n_instances);

    for (i, (&_log_deg, trace_domain)) in log_degrees.iter().zip(trace_domains.iter()).enumerate() {
        let log_quot_deg = log_quotient_degrees[i];
        // Disjoint domain sized by extended degree + quotient degree; use ext domain for shift.
        let quotient_domain =
            ext_trace_domains[i].create_disjoint_domain(1 << (log_ext_degrees[i] + log_quot_deg));

        // Count constraints to size alpha powers packing.
        let constraint_cnt = get_symbolic_constraints(airs[i], 0, pub_vals[i].len()).len();

        // Get evaluations on quotient domain from the main commitment.
        let trace_on_quotient_domain =
            pcs.get_evaluations_on_domain(&main_data, i, quotient_domain);

        // Compute quotient(x) = constraints(x)/Z_H(x) over quotient_domain, as extension values.
        let q_values = quotient_values::<SC, A, _>(
            airs[i],
            &pub_vals[i],
            *trace_domain,
            quotient_domain,
            trace_on_quotient_domain,
            alpha,
            constraint_cnt,
        );

        // Flatten to base field and split into chunks.
        let q_flat = RowMajorMatrix::new_col(q_values).flatten_to_base();
        let num_chunks = 1 << (log_quot_deg + config.is_zk());
        let chunk_mats = quotient_domain.split_evals(num_chunks, q_flat);
        let chunk_domains = quotient_domain.split_domains(num_chunks);

        let start = quotient_chunk_domains.len();
        quotient_chunk_domains.extend(chunk_domains);
        quotient_chunk_mats.extend(chunk_mats);
        let end = quotient_chunk_domains.len();
        quotient_chunk_ranges.push((start, end));
    }

    // Commit to all quotient chunks together (single commitment like SP1).
    let quotient_commit_inputs = quotient_chunk_domains
        .iter()
        .cloned()
        .zip(quotient_chunk_mats.into_iter())
        .collect::<Vec<_>>();
    let (quotient_commit, quotient_data) = pcs.commit(quotient_commit_inputs);
    challenger.observe(quotient_commit.clone());

    // ZK disabled: no randomization round.

    // Sample OOD point.
    let zeta: SC::Challenge = challenger.sample_algebra_element();

    // Build opening rounds.
    let round1_points = trace_domains
        .iter()
        .map(|dom| {
            vec![
                zeta,
                dom.next_point(zeta)
                    .expect("zeta should be out-of-domain for the base trace domain"),
            ]
        })
        .collect::<Vec<_>>();
    let round1 = (&main_data, round1_points);
    let round2_points = quotient_chunk_ranges
        .iter()
        .cloned()
        .flat_map(|(s, e)| (s..e).map(|_| vec![zeta]))
        .collect::<Vec<_>>();
    let round2 = (&quotient_data, round2_points);
    let rounds = vec![round1, round2];

    let (opened_values, opening_proof) = pcs.open(rounds, &mut challenger);
    debug_assert_eq!(opened_values.len(), 2, "expected [main, quotient] groups");
    // Rely on open order: [main, quotient] since ZK is disabled.
    let trace_idx = 0usize;
    let quotient_idx = 1usize;

    // Parse trace opened values per instance.
    let trace_values_for_mats = &opened_values[trace_idx];
    assert_eq!(trace_values_for_mats.len(), n_instances);

    // Parse quotient chunk opened values and map per instance.
    let mut quotient_openings_iter = opened_values[quotient_idx].iter();

    let mut per_instance: Vec<InstanceOpenedValues<SC::Challenge>> =
        Vec::with_capacity(n_instances);
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

        per_instance.push(InstanceOpenedValues {
            trace_local,
            trace_next,
            quotient_chunks: qcs,
        });
    }

    MultiProof {
        commitments: MultiCommitments {
            main: main_commit,
            quotient_chunks: quotient_commit,
        },
        opened_values: MultiOpenedValues {
            instances: per_instance,
        },
        opening_proof,
        degree_bits: log_ext_degrees,
    }
}

#[instrument(name = "compute quotient polynomial (multi)", skip_all)]
#[allow(clippy::too_many_arguments)]
fn quotient_values<SC, A, Mat>(
    air: &A,
    public_values: &Vec<Val<SC>>,
    trace_domain: Domain<SC>,
    quotient_domain: Domain<SC>,
    trace_on_quotient_domain: Mat,
    alpha: <SC as p3_uni_stark::StarkGenericConfig>::Challenge,
    constraint_count: usize,
) -> Vec<<SC as p3_uni_stark::StarkGenericConfig>::Challenge>
where
    SC: MSGC,
    A: for<'a> Air<ProverConstraintFolder<'a, SC>>,
    Mat: Matrix<Val<SC>> + Sync,
{
    let quotient_size = quotient_domain.size();
    let width = trace_on_quotient_domain.width();
    let mut sels = trace_domain.selectors_on_coset(quotient_domain);

    let qdb = log2_strict_usize(quotient_domain.size()) - log2_strict_usize(trace_domain.size());
    let next_step = 1 << qdb;

    // Quotient domain size should be a power of two; partial final pack is only possible when smaller than WIDTH.
    debug_assert!(
        quotient_size % PackedVal::<SC>::WIDTH == 0 || quotient_size < PackedVal::<SC>::WIDTH,
        "Quotient domain size should be a power of two; partial final pack is only possible when smaller than WIDTH."
    );

    // Pad selector vectors to the next multiple of WIDTH to avoid out-of-bounds panics
    // when slicing in the parallel loop below.
    let packed_width = PackedVal::<SC>::WIDTH;
    let pad_to = quotient_size.div_ceil(packed_width) * packed_width;
    for _ in quotient_size..pad_to {
        sels.is_first_row.push(Val::<SC>::default());
        sels.is_last_row.push(Val::<SC>::default());
        sels.is_transition.push(Val::<SC>::default());
        sels.inv_vanishing.push(Val::<SC>::default());
    }

    let mut alpha_powers = alpha.powers().collect_n(constraint_count);
    alpha_powers.reverse();
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

            let main = RowMajorMatrix::new(
                trace_on_quotient_domain.vertically_packed_row_pair(i_start, next_step),
                width,
            );

            let accumulator = PackedChallenge::<SC>::ZERO;
            let mut folder = ProverConstraintFolder {
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
            air.eval(&mut folder);

            // quotient(x) = constraints(x) / Z_H(x)
            let quotient = folder.accumulator * inv_vanishing;

            (0..core::cmp::min(quotient_size, PackedVal::<SC>::WIDTH)).map(move |idx_in_packing| {
                SC::Challenge::from_basis_coefficients_fn(|coeff_idx| {
                    quotient.as_basis_coefficients_slice()[coeff_idx].as_slice()[idx_in_packing]
                })
            })
        })
        .collect()
}
