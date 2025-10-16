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

    // Use instances in provided order.
    let degrees: Vec<usize> = instances.iter().map(|i| i.trace.height()).collect();
    let log_degrees: Vec<usize> = degrees.iter().copied().map(log2_strict_usize).collect();

    // Trace domains for each instance.
    let trace_domains: Vec<Domain<SC>> = degrees
        .iter()
        .map(|&deg| pcs.natural_domain_for_degree(deg))
        .collect();
    // No extended domains; we commit at the base domain.

    // Observe per-instance binding data: degree bits (twice), width, and number of public values.
    for (i, inst) in instances.iter().enumerate() {
        let db = log_degrees[i];
        challenger.observe(Val::<SC>::from_usize(db));
        challenger.observe(Val::<SC>::from_usize(db));
        let width = A::width(inst.air);
        challenger.observe(Val::<SC>::from_usize(width));
        let pv_len = inst.public_values.len();
        challenger.observe(Val::<SC>::from_usize(pv_len));
    }

    // Commit to all traces in one multi-matrix commitment, preserving input order.
    let main_commit_inputs = instances
        .iter()
        .zip(trace_domains.iter())
        .map(|(inst, dom)| (*dom, inst.trace.clone()))
        .collect::<Vec<_>>();
    let (main_commit, main_data) = pcs.commit(main_commit_inputs);

    // Observe main commitment and all public values.
    challenger.observe(main_commit.clone());
    for inst in &instances {
        challenger.observe_slice(&inst.public_values);
    }

    // Compute quotient degrees and domains per instance inline in the loop below.

    // Get the random alpha to fold constraints.
    let alpha: <SC as p3_uni_stark::StarkGenericConfig>::Challenge =
        challenger.sample_algebra_element();

    // Build per-instance quotient domains and values, and split into chunks.
    let mut quotient_chunk_domains: Vec<Domain<SC>> = Vec::new();
    let mut quotient_chunk_mats: Vec<RowMajorMatrix<Val<SC>>> = Vec::new();
    // Track ranges so we can map openings back to instances.
    let mut quotient_chunk_ranges: Vec<(usize, usize)> = Vec::with_capacity(instances.len());

    for (i, ((inst, trace_domain), &log_deg)) in instances
        .iter()
        .zip(trace_domains.iter())
        .zip(log_degrees.iter())
        .enumerate()
    {
        let log_quot_deg =
            get_log_quotient_degree::<Val<SC>, A>(inst.air, 0, inst.public_values.len(), 0);
        let quotient_domain = trace_domain.create_disjoint_domain(1 << (log_deg + log_quot_deg));

        // Count constraints to size alpha powers packing.
        let constraint_cnt = get_symbolic_constraints(inst.air, 0, inst.public_values.len()).len();

        // Get evaluations on quotient domain from the main commitment.
        let trace_on_quotient_domain =
            pcs.get_evaluations_on_domain(&main_data, i, quotient_domain);

        // Compute quotient(x) = constraints(x)/Z_H(x) over quotient_domain, as extension values.
        let q_values = quotient_values::<SC, A, _>(
            inst.air,
            &inst.public_values,
            *trace_domain,
            quotient_domain,
            trace_on_quotient_domain,
            alpha,
            constraint_cnt,
        );

        // Flatten to base field and split into chunks.
        let q_flat = RowMajorMatrix::new_col(q_values).flatten_to_base();
        let num_chunks = 1 << log_quot_deg;
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

    // Sample OOD point.
    let zeta: SC::Challenge = challenger.sample_algebra_element();

    // Build opening rounds.
    let round1_points = trace_domains
        .iter()
        .map(|dom| vec![zeta, dom.next_point(zeta).unwrap()])
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
    let trace_idx = SC::Pcs::TRACE_IDX;
    let quotient_idx = SC::Pcs::QUOTIENT_IDX;

    // Parse trace opened values per instance.
    let trace_values_for_mats = &opened_values[trace_idx];
    assert_eq!(trace_values_for_mats.len(), instances.len());

    // Parse quotient chunk opened values and map per instance.
    let mut quotient_openings_iter = opened_values[quotient_idx].iter();

    let mut per_instance: Vec<InstanceOpenedValues<SC::Challenge>> =
        Vec::with_capacity(instances.len());
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
            random: None,
        },
        opened_values: MultiOpenedValues {
            instances: per_instance,
            random: None,
        },
        opening_proof,
        degree_bits: log_degrees,
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

    // pad packed slices if needed
    for _ in quotient_size..PackedVal::<SC>::WIDTH {
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
