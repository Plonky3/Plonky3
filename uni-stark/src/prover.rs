//! Produce a proof that the given trace satisfies the given air.
//!
//! While this implementation, is designed to work with different proof schemes (Both the regular stark and the circle stark)
//! for simplicities sake, we focus on the regular stark proof scheme here. Information about the circle stark proof scheme
//! can be found in the paper https://eprint.iacr.org/2024/278.pdf.
//!
//! TODO: At some point we should write a similar overview for how the circle stark proof scheme works.
//!
//! Standard STARK:
//!
//! Definitions and Setup:
//! - Fix a field `F` with cryptographically large extension field `G` of degree `d + 1`. Additionally,
//!   Fix a basis `{1, b1, ..., b_d}` for `G` over `F`.
//! - Let `T` denote the trace of the computation. It is a matrix of height `N = 2^n` and width `l`.
//! - Let `H = <h>` denote a multiplicative subgroup of `F` of size `2^n` with generator `h`.
//! - Given the `i`th trace column `Ti`, we let `Ti(x)` denote the unique polynomial of degree `N`
//!   such that `Ti(h^j) = Ti[j]` for `j` in `0..N`.
//!   In other words, `Ti(x)` is the evaluation vector of `Ti(x)` over `H`.
//! - Let `C_{alpha}(X1, ..., Xl, Y1, ..., Yl, Z1, ..., Zj)` denote the constraint polynomial coming from the AIR.
//!   It can depend on the current row, the next row, a collection of selector polynomials and a challenge `alpha`.
//!   Assume for the purpose of the proof that the degree of `C_{alpha}` is `3``.
//! - Given a polynomial `f` and a set `D`, let `[[f, D]]` denote a merkle commitment to
//!   the evaluation vector of `f` over `D`. Similarly, `[[{f0, ..., fk}, D]]` denotes a combined merkle
//!   commitment to the evaluation vectors of the polynomials `f0, ..., fk` over `D`.
//!
//! The goal of the prover is to produce a proof that it knows a trace `T` such that:
//! `C_{alpha}(T1(x), ..., Tl(x), T1(gx), ..., Tl(gx), selectors(x)) = 0` for all choices of `alpha in G` and `x` in `H`.
//!
//! Proof Overview:
//!
//! We start by fixing a pair of elements `g0, g1` in `F\H` such that the cosets `g0 H` and `g1 H` are distinct.
//! Let `D` denote the union of those two cosets. Then, for every column `i`, the prover computes the evaluation vectors
//! of `Ti(x)` over `D`. The prover makes a combined merkle commitment `[[{T1, ..., Tl}, D]]`
//! to these vectors and sends it to `V`.
//!
//! Next the verifier responds with it's first challenge `alpha` which the prover uses to construct the constraint polynomial `C_{alpha}`.
//!
//! If the prover is telling the truth, it can find a polynomial `Q` of degree `< 2N - 1` such that
//! `Q(x) = C_{alpha}(T1(x), ..., Tl(x), T1(gx), ..., Tl(gx), selectors(x))/ZH(x)` where `ZH(x) = x^N - 1` is a vanishing polynomial
//! of the subgroup `H`.
//!
//! As `alpha` is in `G`, `Q(x)` will be a polynomial with coefficients in `G`. Hence we can split `Q(x)` into `d` polynomials
//! `Q0, ..., Q_d` such that `Q(x) = Q0(x) + b_1 Q_1(x) + ... + b_dQ_d(x)]` holds for all `x` in `F`. The polynomials
//! `Q0, ..., Q_d` are similarly of degree `<= 2N - 1` and their evaluation vectors can be easily derived from the evaluation
//! vector of `Q(x)`.
//!
//! Hence the prover now computes the evaluation vectors of `Q0, ..., Q_d` over `D` using the formula `C_{alpha}(T1(x), ..., Tl(x), T1(gx), ..., Tl(gx), selectors)/ZH(x)`
//!
//! Next, define `L_g0(x) = (x^N - (g1)^N)/(g0^N - (g1)^N)` and `L_{g1}(x) = (x^N - g1^N)/((g0)^N - g1^N)`.
//! Then `L_g0` is equal to `1` on `g0H` and `0` on `g1H` and `L_{gk}` is equal to `1` on `g1H` and `0` on `g0H`.
//!
//! Then the prover can decompose `Qi(x) = L_{g0}(x)qi0(x) + L_{g1}(x)qi1(x)` where `qi0(x)` and `qi1(x)` are
//! polynomials of degree `<= N - 1`. The evaluations of `qi0(x), qi1(x)` on `g0H` and `g1H` respectively are
//! exactly equal to the evaluations of `Qi(x)` on `g0H` and `g1H`. So the prover has access to these.
//!
//! The prover now computes the evaluation vectors of `qij(x)` over `D` and makes another
//! combined merkle commitment `[[{q00, q01, ..., qd0, qd1}, D]]` which it sends to `V`.
//!
//! Next the verifier responds with it's second challenge `zeta`. The prover responds with a list of evaluations
//! of `T1(zeta), ..., Tl(zeta)`, `T1(g0 zeta), ..., Tl(g0 zeta)` and `qij(zeta)`.
//!
//! The Verifier checks that `C_{alpha}(T1(zeta), ..., Tl(zeta), T1(h zeta), ..., Tl(h zeta), selectors(zeta))/ZH(zeta)`
//! is equal to  `L_{g0}(zeta)(q00(zeta) + b_1q_10(zeta) ... + b_dq_d0(zeta)) + L_{g1}(zeta)(q01(zeta) + b_1q_11(zeta) ... + b_dq_d1(zeta))`.
//!
//! Next the Verifier sends a third challenge `gamma` which the prover uses to combine all of their polynomials into
//! the single polynomial:
//! ```text
//!     f(x) = (T1(zeta) - T1(x))/(zeta - x) + gamma (T1(g * zeta) - T1(x))/(g * zeta - x)
//!             + ...
//!             + gamma^{2l - 2} (Tl(zeta) - Tl(x))/(zeta - x) + gamma^{2l - 1} (Tl(g * zeta) - Tl(x))/(g * zeta - x)
//!             + gamma^{2l} (q00(zeta) - q00(x))/(zeta - x)
//!             + ...
//!             + gamma^{2l + 2d + 1} (qd1(zeta) - qd1(x))/(zeta - x)
//! ```
//!
//! Note that the verifier is also able to compute `f(x)` whenever they get the values of `T1(x), ..., qd1(x)`.
//!
//! The prover now engages in the standard FRI protocol to prover that `f(x)` is a low degree polynomial with the exception
//! that, instead of opening values of `f(x)` it opens the values of `T1(x), ..., qd1(x)` using its previous commitments.
//!
//!
//! Why Does this work?
//!
//! Assume that the prover is lying and so no valid trace `T` exists. The prover has to commit to
//! the polynomials `T1, ..., Tl` before receiving `alpha` which means that, with high probability,
//! the polynomial `C_alpha(x) = C_{alpha}T1(x), ..., Tl(x), T1(hx), ..., Tl(hx), selectors(x))` will not be `0` at
//! all point in `H`. Hence `Q(x) = C_alpha(x)/Z_H(x)` will be a high degree polynomial.
//!
//! But the prover has to now commit to the polynomials `q00(x), ..., qd1(x)`. If some of the `q`'s, are high degree,
//! this will be caught by the FRI check. So the prover must commit low degree polynomials.
//!
//! Hence the polynomial
//! `Q'(x) = L_{g0}(x)(q00(x) + b_1q_10(x) ... + b_dq_d0(x)) + L_{g1}(x)(q01(x) + b_1q_11(x) ... + b_dq_d1(x))`
//! will be incorrect and so with high probability `Q'(zeta) =\= Q(zeta)`.
//!
//! Hence either the prover must lie about one of the evaluations `T0(zeta), T0(h * zeta), ..., qd1(zeta)`.
//!
//! But, given a polynomial `B` if `B(zeta) =\= a`, `(B(x) - a)/(x - zeta)` will be a high degree polynomial.
//!
//! Hence if the prover lies about one of the evaluations, this will also be caught by the FRI check.
//!
//! Thus if the proof passes, the verifier is convinced with high probability that the claimed statement is correct.

use alloc::vec;
use alloc::vec::Vec;
use core::iter;

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

/// Produce a proof that the given trace satisfies the given air.
///
/// Arguments:
/// Config: A collection of public data about the shape of the proof. It includes:
///     - A Polynomial Commitment Scheme.
///     - An Extension field from which random challenges are drawn.
///     - A Random Challenger used for the Fiat-Shamir implementation.
///     - TODO: Should this contain parts of the fri config? E.g. log_blowup?
///
/// air: TODO
/// trace: The execution trace to be proven:
///     - A matrix of height `N = 2^n` and width `l`.
///     - Each column `Ti` is interpreted as an evaluation vector of a polynomial `Ti(x)` over the initial domain `H`.         
/// public_values: A list of public values related to the proof.
///     - TODO: Should this be absorbed into SC?
#[instrument(skip_all)]
#[allow(clippy::multiple_bound_locations)] // cfg not supported in where clauses?
pub fn prove<
    SC,
    #[cfg(debug_assertions)] A: for<'a> Air<crate::check_constraints::DebugConstraintBuilder<'a, Val<SC>>>,
    #[cfg(not(debug_assertions))] A,
>(
    config: &SC,
    air: &A,
    trace: RowMajorMatrix<Val<SC>>,
    public_values: &Vec<Val<SC>>,
) -> Proof<SC>
where
    SC: StarkGenericConfig,
    A: Air<SymbolicAirBuilder<Val<SC>>> + for<'a> Air<ProverConstraintFolder<'a, SC>>,
{
    // In debug mode, check that every row of the trace satisfies the constraint polynomial.
    #[cfg(debug_assertions)]
    crate::check_constraints::check_constraints(air, &trace, public_values);

    // Compute the height `N = 2^n` and `log_2(height)`, `n`, of the trace.
    let degree = trace.height();
    let log_degree = log2_strict_usize(degree);

    // Find deg(C), the degree of the constraint polynomial.
    // For now let us assume that `deg(C) = 3`. TODO: Generalize this assumption.
    let symbolic_constraints = get_symbolic_constraints::<Val<SC>, A>(air, 0, public_values.len());
    let constraint_count = symbolic_constraints.len();
    let constraint_degree = symbolic_constraints
        .iter()
        .map(SymbolicExpression::degree_multiple)
        .max()
        .unwrap_or(0);

    // From the degree of the constraint polynomial, compute the number
    // of quotient polynomials we will split Q(x) into. This is chosen to
    // always be a power of 2.
    let log_quotient_degree = log2_ceil_usize(constraint_degree - 1);
    let num_quotient_chuncks = 1 << log_quotient_degree;

    // Initialize the PCS and the Challenger.
    let pcs = config.pcs();
    let mut challenger = config.initialise_challenger();

    // Get the subgroup `H` of size `N`. We treat each column `Ti` of
    // the trace as an evaluation vector of polynomials `Ti(x)` over `H`.
    // (In the Circle STARK case `H` is instead a standard position twin coset of size `N`)
    let initial_trace_domain = pcs.natural_domain_for_degree(degree);

    // Let `g` denote a generator of the multiplicative group of `F` and `H'` the unique
    // subgroup of `F` of size `N << pcs.config.log_blowup`.

    // For each trace column `Ti`, we compute the evaluation vector of `Ti(x)` over `gH'`. This
    // new extended trace `ET` is hashed into Merkle tree with it's rows bit-reversed.
    //      trace_commit contains the root of the tree
    //      trace_data contains the entire tree.
    //          - trace_data.leaves is the matrix containing `ET`.
    // TODO: Should this also return the domain `gH'`?
    let (trace_commit, trace_data) = info_span!("commit to trace data")
        .in_scope(|| pcs.commit(vec![(initial_trace_domain, trace)]));

    // Observe the instance.
    // degree < 2^255 so we can safely cast log_degree to a u8.
    challenger.observe(Val::<SC>::from_u8(log_degree as u8));
    // TODO: Might be best practice to include other instance data here; see verifier comment.

    challenger.observe(trace_commit.clone());
    challenger.observe_slice(public_values);

    // FIRST FIAT-SHAMIR CHALLENGE: Anything involved in the proof setup should be included by this point.

    // Get the first Fiat-Shamir challenge, `alpha`, which is used to combine the constraint polynomials.
    let alpha: SC::Challenge = challenger.sample_algebra_element();

    // A domain large enough to uniquely identify the quotient polynomial.
    // This domain must be contained in the domain over which `trace_data` is defined.
    // Explicitly it should be equal to `gK` for some subgroup `K` contained in `H'`.
    let quotient_domain =
        initial_trace_domain.create_disjoint_domain(1 << (log_degree + log_quotient_degree));

    // Return a the subset of the extended trace `ET` corresponding to the rows giving evaluations
    // over the quotient domain.
    //
    // This only works if the trace domain is `gH'` and the quotient domain is `gK` for some subgroup `K` contained in `H'`.
    // TODO: Make this explicit in `get_evaluations_on_domain` or otherwise fix this.
    let trace_on_quotient_domain = pcs.get_evaluations_on_domain(&trace_data, 0, quotient_domain);

    // Compute the quotient polynomial `Q(x)` by evaluating `C(T1(x), ..., Tl(x), T1(gx), ..., Tl(gx), selectors(x)) / Z_H(x)`
    // at every point in the quotient domain. The degree of `Q(x)` is `<= deg(C) * (N - 1) - N + 1 = 2(N - 1)`.
    // The `-N` comes from dividing by `Z_H(x)` and the `+1` is due to the `is_transition` selector.
    let quotient_values = quotient_values(
        air,
        public_values,
        initial_trace_domain,
        quotient_domain,
        trace_on_quotient_domain,
        alpha,
        constraint_count,
    );

    // Due to `alpha`, evaluations of Q all lie in the extension field `G`.
    // We flatten this into a matrix of `F` values by treating `G` as an `F`
    // vector space and so separating each element of `G` into `d = [G: F]` elements of `F`.
    //
    // This is valid to do because our domain lies in the base field `F`. Hence we can split
    // `Q(x)` into `d` polynomials `Q_0(x), ... , Q_{d-1}(x)` each contained in `F`.
    // such that `Q(x) = [Q_0(x), ... ,Q_{d-1}(x)]` holds for all `x` in `F`.
    let quotient_flat = RowMajorMatrix::new_col(quotient_values).flatten_to_base();

    // Currently each polynomial `Q_i(x)` is of degree `<= 2(N - 1)` and
    // we have it's evaluations over a the coset `gK of size `2N`. Let `k` be the chosen
    // generator of `K` which satisfies `k^2 = h`.
    //
    // We can split this coset into the sub-cosets `gH` and `gkH` each of size `N`.
    // Define `L_g(x) = (x^N - (gk)^N)/(g^N - (gk)^N)` and `L_{gk}(x) = (x^N - g^N)/((gk)^N - g^N)`.
    // Then `L_g` is equal to `1` on `gH` and `0` on `gkH` and `L_{gk}` is equal to `1` on `gkH` and `0` on `gH`.
    //
    // Thus we can decompose `Q_i(x) = L_{g}(x)q_{i0}(x) + L_{gk}(x)q_{i1}(x)`
    // where `q_{i0}(x)` and `q_{i1}(x)` are polynomials of degree `<= N - 1`.
    // Moreover the evaluations of `q_{i0}(x), q_{i1}(x)` on `gH` and `gkH` respectively are
    // exactly the evaluations of `Q_i(x)` on `gH` and `gkH`. Hence we can get these evaluation
    // vectors by simply splitting the evaluations of `Q_i(x)` into two halves.
    // quotient_chunks contains the evaluations of `q_{i0}(x)` and `q_{i1}(x)`.
    let quotient_chunks = quotient_domain.split_evals(num_quotient_chuncks, quotient_flat);
    let qc_domains = quotient_domain.split_domains(num_quotient_chuncks);

    // TODO: This treats the all `q_ij` as if they are evaluations over the domain `H`.
    // This doesn't matter for low degree-ness but we need to be careful when checking
    // equalities.

    // When computing quotient_data, we take `q_ij` (defined on `gj H`), re-interpret it to
    // be defined on `H`, e.g. replacing it with `q_ij'(x) = q_ij(gj x)`. Then we do a coset LDE to
    // get it's evaluations on `(g/gj) H'` which we commit to. Then when computing opening values
    // we again re-interpret the evaluation vector to be defined on `g H` e.g. replacing our
    // polynomial with `q_ij''(x) = q_ij'(gj^{-1} x) = q_ij(gj gj^{-1} x) = x`.
    // In other words our commitment does actually compute the evaluations of `q_ij(x)` over `gH`.
    // Despite seemingly doing something else...

    // For each polynomial `q_ij`, compute the evaluation vector of `q_ij(x)` over `gH'`. This
    // is then hashed into a Merkle tree with it's rows bit-reversed.
    //      quotient_commit contains the root of the tree
    //      quotient_data contains the entire tree.
    //          - quotient_data.leaves is a pair of matrices containing the `q_i0(x)` and `q_i1(x)`.
    let (quotient_commit, quotient_data) = info_span!("commit to quotient poly chunks")
        .in_scope(|| pcs.commit(izip!(qc_domains, quotient_chunks).collect_vec()));
    challenger.observe(quotient_commit.clone());

    // Combine our commitments to the trace and quotient polynomials into a single object which
    // will be passed to the verifier.
    let commitments = Commitments {
        trace: trace_commit,
        quotient_chunks: quotient_commit,
    };

    // Get the second Fiat-Shamir challenge, `zeta`, an opening point.
    // Along with `zeta_next = next(zeta)` where next is the unique successor linear function
    // on the initial domain. In the usual STARK case, this is `h * zeta`.
    //
    // TODO: What is this in the Circle STARK case?
    let zeta: SC::Challenge = challenger.sample();
    let zeta_next = initial_trace_domain.next_point(zeta).unwrap();

    // The prover opens the trace polynomials at `zeta` and `zeta_next` and the
    // and quotient polynomials at `zeta`.
    //
    // TODO: What are opened_values, opening_proof??
    // This also produces a FRI proof??
    // Why is zeta the right point to evaluate quotient_data at?
    let (opened_values, opening_proof) = info_span!("open").in_scope(|| {
        pcs.open(
            vec![
                (&trace_data, vec![vec![zeta, zeta_next]]),
                (
                    &quotient_data,
                    // open every chunk at zeta
                    iter::repeat_n(vec![zeta], num_quotient_chuncks).collect_vec(),
                ),
            ],
            &mut challenger,
        )
    });
    let trace_local = opened_values[0][0][0].clone();
    let trace_next = opened_values[0][0][1].clone();
    let quotient_chunks = opened_values[1].iter().map(|v| v[0].clone()).collect_vec();
    let opened_values = OpenedValues {
        trace_local,
        trace_next,
        quotient_chunks,
    };
    Proof {
        commitments,
        opened_values,
        opening_proof,
        degree_bits: log_degree,
    }
}

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
    let quotient_size = quotient_domain.size();
    let width = trace_on_quotient_domain.width();
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

            // "Transpose" D packed base coefficients into WIDTH scalar extension coefficients.
            (0..core::cmp::min(quotient_size, PackedVal::<SC>::WIDTH)).map(move |idx_in_packing| {
                SC::Challenge::from_basis_coefficients_fn(|coeff_idx| {
                    quotient.as_basis_coefficients_slice()[coeff_idx].as_slice()[idx_in_packing]
                })
            })
        })
        .collect()
}
