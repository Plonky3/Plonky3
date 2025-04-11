//! Produce a proof that the given trace satisfies the given air.
//!
//! While this implementation is designed to work with different proof schemes (both the regular Stark and the Circle Stark)
//! for simplicity we focus on the regular Stark proof scheme here. Information about the Circle Stark proof scheme
//! can be found in the paper https://eprint.iacr.org/2024/278.pdf.
//!
//! TODO: At some point we should write a similar overview for how the circle stark proof scheme works.
//!
//! Standard STARK:
//!
//! Definitions and Setup:
//! - Fix a field `F` with cryptographically large extension field `E` of degree `e + 1`. Additionally,
//!   Fix a basis `{1 = b_0, b_1, ..., b_e}` for `E` over `F`.
//! - Let `T` denote the trace of the computation. It is a matrix of height `N = 2^n` and width `w`.
//! - Let `H = <h>, K = <k>` denote multiplicative subgroups of `F` of size `N` and `2N`
//!   with generators `h` and `k` respectively which satisfy `k^2 = h`. Let `g` denote some nonzero element
//!   of `F` which is not in `H` or `K`.
//! - Given the `i`th trace column `T_i`, we let `T_i(x)` denote the unique polynomial of degree `N`
//!   such that `T_i(h^j) = T_i[j]` for `j` in `0..N`.
//!   In other words, `T_i` is the evaluation vector of `T_i(x)` over `H`.
//! - Let `C_{alpha}(X_1, ..., X_w, Y_1, ..., Y_w, Z_1, ..., Z_j)` denote the constraint polynomial coming from the AIR.
//!   The `X_i`'s control the dependence on the value in the current row and `i`'th column.
//!   The `Y_i`'s control the dependence on the value in the next row and `i`'th column.
//!   The `Z`'s control the dependence on the (`j`) row-selector polynomials.
//!   The `alpha` is a challenge from the verifier used to combine all constraints into a single polynomial.
//!   Assume for the purpose of this overview that the degree of `C_{alpha}` is `3`.
//! - Given a polynomial `f` and a set `D`, let `f<D>` denote the evaluation vector of `f` over `D`.
//!   Additionally, let `[[f<D>]]` denote a merkle commitment to that evaluation vector.
//!   the evaluation vector of `f` over `D`. Similarly, if `{f_0, ..., f_k}` is a collection
//!   of polynomials, `{f_0, ..., f_k}<D>` denotes the collection of evaluation vectors `f_0<D>, ..., f_k<D>`
//!   and `[[{f_0, ..., f_k}<D>]]` is a batched commitment to those vectors.
//!
//! The goal of the prover is to produce a proof that it knows a trace `T` such that:
//! `C_{alpha}(T_1(x), ..., T_w(x), T_1(hx), ..., T_w(hx), selectors(x)) = 0` for all `x` in `H`.
//!
//! Proof Overview:
//!
//! To start with, for every column `i`, the prover computes the evaluation vectors of `T_i<gK>`.
//! The prover makes a combined merkle commitment `[[{T_1, ..., T_w}<gK>]]` to these vectors and sends it to the verifier.
//!
//! Next the verifier responds with their first challenge `alpha`.
//! The prover uses this to construct the constraint polynomial `C_{alpha}`.
//!
//! If the prover is telling the truth, they can find a polynomial `Q` of degree `< 2N - 1` such that
//! `Q(x) = C_{alpha}(T_1(x), ..., T_w(x), T_1(gx), ..., T_w(gx), selectors(x))/Z_H(x)`
//! where `Z_H(x) = x^N - 1` is a vanishing polynomial of the subgroup `H`.
//!
//! As `alpha` is in `E`, `Q(x) \in E[X]`. The prover can then use the basis `{1 = b_0, b_1, ..., b_e}`
//! for `E` over `F`to split `Q(x)` into `e + 1` polynomials `Q_0, ..., Q_e \in F[X]` such that
//! `Q(x) = Q_0(x) + b_1 Q_1(x) + ... + b_e Q_e(x)`. These polynomials `Q_i` have the same degree (`2N - 1`)
//! as `Q` and their evaluation vectors can be read off the evaluation vector of `Q(x)`.
//!
//! The prover now computes the evaluation vectors of `Q_0, ..., Q_e` over `gK` using the evaluations
//! of the `T`'s selectors and `Z_H` over `gK`.
//!
//! Next, define (using the fact that `k^N = -1`):
//! ```text
//!    L_g(x)    = (x^N - (gk)^N)/(g^N - (gk)^N) = (x^N + g^N)/2g^N
//!    L_{gk}(x) = (x^N - g^N)/(g^N - (gk)^N)    = -(x^N - g^N)/2g^N.
//! ```
//! Then `L_g` is equal to `1` on `gH` and `0` on `gkH` and `L_{gk}` is equal to `1` on `gkH` and `0` on `gH`.
//!
//! Using this, the prover can decompose `Q_i(x) = L_{g}(x)q_{i0}(x) + L_{gk}(x)q_{i1}(x)`
//! where `q_{i0}(x)` and `q_{i1}(x)` are polynomials of degree `<= N - 1`. The evaluations
//! of `q_{i0}(x), q_{i1}(x)` on `gH` and `gkH` respectively are exactly equal to the evaluations
//! of `Q_i(x)` on `gH` and `gkH`. So the prover can access these for free.
//!
//! The prover now computes the evaluation vectors of `q_{ij}(x)` over `gK` and makes another
//! combined merkle commitment `[[{q_{00}, q_{01}, ..., q_{d0}, q_{d1}}<gK>]]` which it sends to the verifier.
//!
//! The verifier responds with its second challenge `zeta`.
//!
//! The prover takes this challenge and computes the evaluations `T_i(zeta), T_i(h zeta)` and `q_{ij}(zeta)` for all
//! the `T`'s and `q`'s. They send these evaluations to the verifier.
//!
//! The verifier checks that:
//! ```text
//!     C_{alpha}(T_1(zeta), ..., T_w(zeta), T_1(h zeta), ..., T_w(h zeta), selectors(zeta))/Z_H(zeta)
//!         = L_{g0}(zeta)(q_{00}(zeta) + b_1 q_{10}(zeta) ... + b_d q_{d0}(zeta)) + L_{g1}(zeta)(q_{01}(zeta) + b_1 q_{11}(zeta) ... + b_d q_{d1}(zeta)).
//! ```
//!
//! Provided this check passes, the verifier sends their third challenge `gamma`.
//!
//! The prover now uses `gamma` to combine all of their polynomials into
//! the single polynomial:
//! ```text
//!     f(x) = (T_1(zeta) - T_1(x))/(zeta - x) + gamma (T_1(h * zeta) - T_1(x))/(h * zeta - x)
//!             + ...
//!             + gamma^{2l - 2} (T_w(zeta) - T_w(x))/(zeta - x) + gamma^{2l - 1} (T_w(h * zeta) - T_w(x))/(h * zeta - x)
//!             + gamma^{2l} (q_{00}(zeta) - q_{00}(x))/(zeta - x)
//!             + ...
//!             + gamma^{2l + 2d + 1} (q_{d1}(zeta) - q_{d1}(x))/(zeta - x)
//! ```
//!
//! Note that the verifier is also able to compute `f(x)` whenever they get the values of `T_1(x), T_1(h x), ..., q_{d1}(x)`
//!
//! The prover and verifier now engage in the standard FRI protocol to prove that `f(x)` is a low degree polynomial.
//! The one small modification is that, instead of opening values of `f(x)` the prover
//! opens the values of `T_1(x), T_1(h x), ..., q_{d1}(x)` using its previous commitments.
//!
//!
//! Why Does this work?
//!
//! Assume that the prover is lying and so no valid trace `T` exists. The prover has to commit to
//! the polynomials `T_1, ..., T_w` before receiving `alpha` which means that, with high probability,
//! the polynomial `C_{alpha}(x) = C_{alpha}(T_1(x), ..., T_w(x), T_1(hx), ..., T_w(hx), selectors(x))` will not be `0` at
//! all points in `H`. Hence `Q(x) = C_alpha(x)/Z_H(x)` will not be a polynomial.
//!
//! But the prover has to now commit to the polynomials `q_{00}(x), ..., q_{d1}(x)`. If some of the `q`'s, are high degree,
//! this will be caught by the FRI check. So the prover must commit low degree polynomials.
//!
//! Hence the polynomial
//! `Q'(x) = L_{g0}(x)(q_{00}(x) + b_1 q_{10}(x) ... + b_d q_{d0}(x)) + L_{g1}(x)(q_{01}(x) + b_1 q_{11}(x) ... + b_d q_{d1}(x))`
//! will be incorrect and so with high probability `Q'(zeta) =\= Q(zeta)`.
//!
//! Hence either the prover must lie about one of the evaluations `T_0(zeta), T_0(h * zeta), ..., q_{d1}(zeta)`.
//!
//! But, given a polynomial `B` if `B(zeta) =\= a`, `(B(x) - a)/(x - zeta)` will not be a polynomial and so
//! the evaluations of `B` over the coset `gH` will
//!
//! Hence if the prover lies about one of the evaluations, this will be caught by the FRI check.
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
///     - A matrix of height `N = 2^n` and width `w`.
///     - Each column `T_i` is interpreted as an evaluation vector of a polynomial `T_i(x)` over the initial domain `H`.         
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

    // Compute the constraint polynomials as vectors of symbolic expressions.
    let symbolic_constraints = get_symbolic_constraints::<Val<SC>, A>(air, 0, public_values.len());

    // Count the number of constraints that we have.
    let constraint_count = symbolic_constraints.len();

    // Find the total degree of the multivariate constraint polynomial `C`.
    //
    // For now in comments we assume that `deg(C) = 3`.
    let constraint_degree = symbolic_constraints
        .iter()
        .map(SymbolicExpression::degree_multiple)
        .max()
        .unwrap_or(0);

    // From the degree of the constraint polynomial, compute the number
    // of quotient polynomials we will split Q(x) into. This is chosen to
    // always be a power of 2.
    let log_quotient_degree = log2_ceil_usize(constraint_degree - 1);
    let num_quotient_chunks = 1 << log_quotient_degree;

    // Initialize the PCS and the Challenger.
    let pcs = config.pcs();
    let mut challenger = config.initialise_challenger();

    // Get the subgroup `H` of size `N`. We treat each column `T_i` of
    // the trace as an evaluation vector of polynomials `T_i(x)` over `H`.
    // (In the Circle STARK case `H` is instead a standard position twin coset of size `N`)
    let initial_trace_domain = pcs.natural_domain_for_degree(degree);

    // Let `g` denote a generator of the multiplicative group of `F` and `H'` the unique
    // subgroup of `F` of size `N << pcs.config.log_blowup`.

    // For each trace column `T_i`, we compute the evaluation vector of `T_i(x)` over `H'`. This
    // new extended trace `ET` is hashed into a Merkle tree with its rows bit-reversed.
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

    // Observe the Merkle root of the trace commitment.
    challenger.observe(trace_commit.clone());

    // Observe the public input values.
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

    // Compute the quotient polynomial `Q(x)` by evaluating `C(T_1(x), ..., T_w(x), T_1(hx), ..., T_w(hx), selectors(x)) / Z_H(x)`
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

    // Due to `alpha`, evaluations of `Q` all lie in the extension field `E`.
    // We flatten this into a matrix of `F` values by treating `E` as an `F`
    // vector space and so separating each element of `E` into `e + 1 = [E: F]` elements of `F`.
    //
    // This is valid to do because our domain lies in the base field `F`. Hence we can split
    // `Q(x)` into `e + 1` polynomials `Q_0(x), ... , Q_e(x)` each contained in `F`.
    // such that `Q(x) = [Q_0(x), ... ,Q_e(x)]` holds for all `x` in `F`.
    let quotient_flat = RowMajorMatrix::new_col(quotient_values).flatten_to_base();

    // Currently each polynomial `Q_i(x)` is of degree `<= 2(N - 1)` and
    // we have it's evaluations over a the coset `gK of size `2N`. Let `k` be the chosen
    // generator of `K` which satisfies `k^2 = h`.
    //
    // We can split this coset into the sub-cosets `gH` and `gkH` each of size `N`.
    // Define:  L_g(x)    = (x^N - (gk)^N)/(g^N - (gk)^N) = (x^N + g^N)/2g^N
    //          L_{gk}(x) = (x^N - g^N)/(g^N - (gk)^N)    = -(x^N - g^N)/2g^N.
    // Then `L_g` is equal to `1` on `gH` and `0` on `gkH` and `L_{gk}` is equal to `1` on `gkH` and `0` on `gH`.
    //
    // Thus we can decompose `Q_i(x) = L_{g}(x)q_{i0}(x) + L_{gk}(x)q_{i1}(x)`
    // where `q_{i0}(x)` and `q_{i1}(x)` are polynomials of degree `<= N - 1`.
    // Moreover the evaluations of `q_{i0}(x), q_{i1}(x)` on `gH` and `gkH` respectively are
    // exactly the evaluations of `Q_i(x)` on `gH` and `gkH`. Hence we can get these evaluation
    // vectors by simply splitting the evaluations of `Q_i(x)` into two halves.
    // quotient_chunks contains the evaluations of `q_{i0}(x)` and `q_{i1}(x)`.
    let quotient_chunks = quotient_domain.split_evals(num_quotient_chunks, quotient_flat);
    let qc_domains = quotient_domain.split_domains(num_quotient_chunks);

    // TODO: This treats the all `q_{ij}` as if they are evaluations over the domain `H`.
    // This doesn't matter for low degree-ness but we need to be careful when checking
    // equalities.

    // When computing quotient_data, we take `q_{ij}` (defined on `g (k^j) H`), re-interpret it to
    // be defined on `H`, e.g. replacing it with `q_{ij}'(x) = q_{ij}(g (k^j) x)`. Then we do a coset LDE to
    // get it's evaluations on `(g/g (k^j)) H' = (k^{-j}) H'` which we commit to. Then when computing opening values
    // we again re-interpret the evaluation vector to be defined on `g H'` e.g. replacing our
    // polynomial with `q_{ij}''(x) = q_{ij}'(g^{-1} k^{-j} x) = q_{ij}(g^{-1} k^{-j} g (k^j) x) = q_{ij}(x)`.
    // In other words our commitment does actually compute the evaluations of `q_{ij}(x)` over `gH`.
    // Despite seemingly doing something else... Should try to make this clearer.

    // For each polynomial `q_{ij}`, compute the evaluation vector of `q_{ij}(x)` over `gH'`. This
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
                    iter::repeat_n(vec![zeta], num_quotient_chunks).collect_vec(),
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

/// Compute the values of the quotient polynomial q(x) = c(x) / Z_H(x),
/// where c(x) is the sum of all constraints weighted by powers of alpha,
/// and Z_H(x) is the vanishing polynomial over the trace domain H.
///
/// This function returns a vector extension field elements
/// corresponding to evaluations of q(x) over the quotient domain.
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

    // Compute row selectors over the quotient domain.
    // The three selectors we compute are is_first_row, is_last_row and is_transition (aka is not last row).
    // Additionally we compute the inverse of the vanishing polynomial Z_H(x).
    let mut sels = debug_span!("Compute Selectors")
        .in_scope(|| trace_domain.selectors_on_coset(quotient_domain));

    // Plonky3 support constraint which depend on both the current row of the trace as well
    // as the next row. While these rows start off adjacent, after the log blow up they will be further apart.
    //
    // next_step records how far away the "next" row is from the current row.
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

            // Load selectors for these rows
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
