use alloc::vec;
use alloc::vec::Vec;

use itertools::{Itertools, izip};
use p3_air::Air;
use p3_challenger::{CanObserve, CanSample, FieldChallenger};
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::{BasedVectorSpace, PackedValue, PrimeCharacteristicRing};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use p3_util::{log2_ceil_usize, log2_strict_usize};
use tracing::{debug_span, info_span, instrument};

use crate::{
    Commitments, Domain, OpenedValues, PackedChallenge, PackedVal, Proof, ProverConstraintFolder,
    StarkGenericConfig, SymbolicAirBuilder, SymbolicExpression, Val, get_symbolic_constraints,
};

/// Construct a STARK proof for a given AIR instance and trace.
///
/// This function performs the full STARK prover logic:
/// - commits to the trace
/// - samples a challenge alpha
/// - computes quotient polynomials (constraint / vanishing)
/// - commits to the quotient chunks
/// - samples a challenge zeta
/// - opens the committed polynomials at zeta (and zeta * g)
/// - returns a `Proof` object
///
/// Inputs:
/// - `config`: STARK configuration
/// - `air`: the AIR instance to prove
/// - `challenger`: the Fiat-Shamir challenger
/// - `trace`: trace matrix (main witness values)
/// - `public_values`: public inputs to the proof
///
/// Output: a complete `Proof` object with all required openings and commitments.
#[instrument(skip_all)]
#[allow(clippy::multiple_bound_locations)] // cfg not supported in where clauses?
pub fn prove<
    SC,
    #[cfg(debug_assertions)] A: for<'a> Air<crate::check_constraints::DebugConstraintBuilder<'a, Val<SC>>>,
    #[cfg(not(debug_assertions))] A,
>(
    config: &SC,
    air: &A,
    challenger: &mut SC::Challenger,
    trace: RowMajorMatrix<Val<SC>>,
    public_values: &Vec<Val<SC>>,
) -> Proof<SC>
where
    SC: StarkGenericConfig,
    A: Air<SymbolicAirBuilder<Val<SC>>> + for<'a> Air<ProverConstraintFolder<'a, SC>>,
{
    #[cfg(debug_assertions)]
    crate::check_constraints::check_constraints(air, &trace, public_values);

    // Get the trace length `n`, which determines the degree of the trace polynomials.
    let degree = trace.height();
    // Compute log₂(n), which is needed for constructing domains and commitments.
    let log_degree = log2_strict_usize(degree);

    // Symbolically extract all AIR constraints from the system.
    let symbolic_constraints = get_symbolic_constraints(air, 0, public_values.len());
    // Count how many constraints we have.
    let constraint_count = symbolic_constraints.len();
    // Find the maximum degree multiple among all constraints.
    let constraint_degree = symbolic_constraints
        .iter()
        .map(SymbolicExpression::degree_multiple)
        .max()
        .unwrap_or(0);
    // Compute log₂(d - 1), where `d` is the max degree of constraints, for quotient poly sizing.
    let log_quotient_degree = log2_ceil_usize(constraint_degree - 1);
    // Set the padded degree of the quotient polynomial to 2^log_quotient_degree.
    let quotient_degree = 1 << log_quotient_degree;

    // Load the polynomial commitment scheme interface.
    let pcs = config.pcs();
    // Construct the evaluation domain of size `n` for the trace.
    //
    // This depends on the chosen polynomial commitment scheme.
    let trace_domain = pcs.natural_domain_for_degree(degree);

    // Commit to the trace polynomials over the trace domain.
    let (trace_commit, trace_data) =
        info_span!("commit to trace data").in_scope(|| pcs.commit(vec![(trace_domain, trace)]));

    // Observe the log-degree of the trace as part of the Fiat-Shamir transcript.
    //
    // degree < 2^255 so we can safely cast log_degree to a u8.
    challenger.observe(Val::<SC>::from_u8(log_degree as u8));
    // TODO: Might be best practice to include other instance data here; see verifier comment.

    // Observe the Merkle root of the trace commitment.
    challenger.observe(trace_commit.clone());
    // Observe the public input values.
    challenger.observe_slice(public_values);
    // Sample the challenge `α` ∈ 𝔽^d (an extension field element).
    let alpha: SC::Challenge = challenger.sample_algebra_element();

    // Create a disjoint coset domain for quotient evaluation.
    let quotient_domain =
        trace_domain.create_disjoint_domain(1 << (log_degree + log_quotient_degree));

    // Evaluate the trace polynomials on this larger quotient domain.
    let trace_on_quotient_domain = pcs.get_evaluations_on_domain(&trace_data, 0, quotient_domain);

    // Evaluate all AIR constraints and divide by vanishing polynomial to get the quotient.
    let quotient_values = quotient_values(
        air,
        public_values,
        trace_domain,
        quotient_domain,
        trace_on_quotient_domain,
        alpha,
        constraint_count,
    );
    // Flatten the extension-valued quotient into a base field matrix.
    let quotient_flat = RowMajorMatrix::new_col(quotient_values).flatten_to_base();
    // Split the flat quotient into `quotient_degree` chunks.
    let quotient_chunks = quotient_domain.split_evals(quotient_degree, quotient_flat);
    // Corresponding subdomains for each quotient chunk.
    let qc_domains = quotient_domain.split_domains(quotient_degree);

    // Commit to each quotient chunk over its subdomain.
    let (quotient_commit, quotient_data) = info_span!("commit to quotient poly chunks")
        .in_scope(|| pcs.commit(izip!(qc_domains, quotient_chunks).collect_vec()));
    // Observe the commitment to the quotient chunks in the Fiat-Shamir transcript.
    challenger.observe(quotient_commit.clone());

    // Store the commitments into the proof.
    let commitments = Commitments {
        trace: trace_commit,
        quotient_chunks: quotient_commit,
    };

    // Sample the challenge `ζ` ∈ 𝔽^d: the point at which we open the committed polynomials.
    let zeta: SC::Challenge = challenger.sample();
    // Compute ζ⋅g: the "next" point in the domain (for transition constraints).
    let zeta_next = trace_domain.next_point(zeta).unwrap();

    // Open both the trace and the quotient polynomials at ζ (and ζ⋅g for transitions).
    let (opened_values, opening_proof) = info_span!("open").in_scope(|| {
        pcs.open(
            vec![
                // open trace at ζ and ζ⋅g
                (&trace_data, vec![vec![zeta, zeta_next]]),
                (
                    &quotient_data,
                    // open every chunk at ζ
                    (0..quotient_degree).map(|_| vec![zeta]).collect_vec(),
                ),
            ],
            challenger,
        )
    });
    // Extract trace evaluations at ζ and ζ⋅g
    let trace_local = opened_values[0][0][0].clone();
    let trace_next = opened_values[0][0][1].clone();

    // Extract quotient chunk evaluations at ζ
    let quotient_chunks = opened_values[1].iter().map(|v| v[0].clone()).collect_vec();

    // Wrap the opened values into a structured object.
    let opened_values = OpenedValues {
        trace_local,
        trace_next,
        quotient_chunks,
    };

    // Return the full proof.
    Proof {
        commitments,
        opened_values,
        opening_proof,
        degree_bits: log_degree,
    }
}

/// Compute the values of the quotient polynomial q(x) = c(x) / Z_H(x),
/// where c(x) is the sum of all constraints weighted by powers of alpha,
/// and Z_H(x) is the vanishing polynomial over the trace domain H.
///
/// This function:
/// - packs trace rows in chunks
/// - evaluates all AIR constraints
/// - divides by vanishing Z_H(x)
/// - returns q(x) ∈ 𝔽^d in a vector of extension field elements
#[instrument(name = "compute quotient polynomial", skip_all)]
fn quotient_values<SC, A, Mat>(
    air: &A,
    public_values: &Vec<Val<SC>>,
    trace_domain: Domain<SC>,
    quotient_domain: Domain<SC>,
    trace_on_quotient_domain: Mat,
    alpha: SC::Challenge,
    constraint_count: usize,
) -> Vec<SC::Challenge>
where
    SC: StarkGenericConfig,
    A: for<'a> Air<ProverConstraintFolder<'a, SC>>,
    Mat: Matrix<Val<SC>> + Sync,
{
    // Total number of evaluation points
    let quotient_size = quotient_domain.size();
    // Number of trace columns
    let width = trace_on_quotient_domain.width();

    // Compute row selectors over the quotient domain (first row, last row, etc.).
    let mut sels = debug_span!("Compute Selectors")
        .in_scope(|| trace_domain.selectors_on_coset(quotient_domain));

    // Compute `qdb = log2(quotient_size / trace_size)`
    let qdb = log2_strict_usize(quotient_domain.size()) - log2_strict_usize(trace_domain.size());
    // Next row offset = 2^qdb
    let next_step = 1 << qdb;

    // We take PackedVal::<SC>::WIDTH worth of values at a time from a quotient_size slice, so we need to
    // pad with default values in the case where quotient_size is smaller than PackedVal::<SC>::WIDTH.
    for _ in quotient_size..PackedVal::<SC>::WIDTH {
        sels.is_first_row.push(Val::<SC>::default());
        sels.is_last_row.push(Val::<SC>::default());
        sels.is_transition.push(Val::<SC>::default());
        sels.inv_vanishing.push(Val::<SC>::default());
    }

    // Compute α^i in reverse order: αⁿ⁻¹, ..., α¹, α⁰
    let mut alpha_powers = alpha.powers().take(constraint_count).collect_vec();
    alpha_powers.reverse();
    // alpha powers looks like Vec<EF> ~ Vec<[F; D]>
    // It's useful to also have access to the the transpose of this of form [Vec<F>; D].
    let decomposed_alpha_powers: Vec<_> = (0..SC::Challenge::DIMENSION)
        .map(|i| {
            alpha_powers
                .iter()
                .map(|x| x.as_basis_coefficients_slice()[i])
                .collect()
        })
        .collect();

    // Iterate in parallel over chunks of `WIDTH` rows in the quotient domain.
    (0..quotient_size)
        .into_par_iter()
        .step_by(PackedVal::<SC>::WIDTH)
        .flat_map_iter(|i_start| {
            let i_range = i_start..i_start + PackedVal::<SC>::WIDTH;

            // Load selectors for these rows.
            let is_first_row = *PackedVal::<SC>::from_slice(&sels.is_first_row[i_range.clone()]);
            let is_last_row = *PackedVal::<SC>::from_slice(&sels.is_last_row[i_range.clone()]);
            let is_transition = *PackedVal::<SC>::from_slice(&sels.is_transition[i_range.clone()]);
            let inv_vanishing = *PackedVal::<SC>::from_slice(&sels.inv_vanishing[i_range]);

            // Pack the trace values for these rows and their next rows.
            let main = RowMajorMatrix::new(
                trace_on_quotient_domain.vertically_packed_row_pair(i_start, next_step),
                width,
            );

            // Initialize the constraint accumulator to zero.
            let accumulator = PackedChallenge::<SC>::ZERO;

            // Construct the `ProverConstraintFolder, which applies constraints.
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
            // Evaluate the AIR constraints over these packed rows.
            air.eval(&mut folder);

            // Divide constraint polynomial by vanishing polynomial:
            // q(x) = c(x) / Z_H(x)
            let quotient = folder.accumulator * inv_vanishing;

            // "Transpose" D packed base coefficients into WIDTH scalar extension coefficients.
            (0..core::cmp::min(quotient_size, PackedVal::<SC>::WIDTH)).map(move |idx_in_packing| {
                SC::Challenge::from_basis_coefficients_fn(|coeff_idx| {
                    quotient.as_basis_coefficients_slice()[coeff_idx].as_slice()[idx_in_packing]
                })
            })
        })
        .collect()
}
