//! Quotient polynomial recomposition for recursive STARK verification.

use alloc::vec::Vec;

use itertools::Itertools;
use p3_circuit::CircuitBuilder;
use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing};
use p3_uni_stark::StarkGenericConfig;

use crate::Target;
use crate::traits::{Recursive, RecursivePcs};

/// Reconstructs the quotient polynomial evaluation `Q(ζ)` from opened chunks.
///
/// # Mathematical Background
///
/// When the quotient polynomial `Q(X)` is split into chunks `Q_i(X)` over disjoint domains
/// `D_i`, we reconstruct `Q(ζ)` at an out-of-domain point using:
///
/// ```text
/// Q(ζ) = ∑_i Q_i(·) · L_i(ζ)
/// ```
///
/// where `L_i(ζ)` is the Lagrange interpolation coefficient:
///
/// ```text
/// L_i(ζ) = ∏_{j≠i} [Z_j(ζ) / Z_j(g_i)]
/// ```
///
/// Here:
/// - `Z_j(X) = (X/g_j)^|D_j| - 1` is the vanishing polynomial for domain `D_j`
/// - `g_i` is the first point (generator) of domain `D_i`
/// - `Q_i(·)` is evaluated in the extension field basis
///
/// # Optimization: Constant Pre-computation
///
/// The critical insight is that `Z_j(g_i)` values are **field constants** that can be computed
/// outside the circuit! This allows us to restructure the computation as:
///
/// ```text
/// L_i(ζ) = [∏_j Z_j(ζ) / Z_i(ζ)] · [1 / ∏_{j≠i} Z_j(g_i)]
///          └─────────────────────┘   └────────────────────┘
///            computed in-circuit      pre-computed constant
/// ```
///
/// This transforms the denominator from O(N^2) in-circuit operations to O(N^2) native field
/// operations.
///
/// # Arguments
///
/// * `circuit` - The circuit builder for adding verification constraints.
/// * `quotient_chunks_domains` - Domains `[D_0, ..., D_{n-1}]` for each chunk.
/// * `quotient_chunks` - Coefficient vectors for each chunk in extension field basis.
/// * `zeta` - The out-of-domain evaluation point `ζ`.
/// * `pcs` - The polynomial commitment scheme providing domain operations.
///
/// # Returns
///
/// A `Target` representing the circuit variable for `Q(ζ)`.
pub fn recompose_quotient_from_chunks_circuit<
    SC: StarkGenericConfig,
    InputProof: Recursive<SC::Challenge>,
    OpeningProof: Recursive<SC::Challenge>,
    Comm: Recursive<SC::Challenge>,
    Domain: Copy,
>(
    circuit: &mut CircuitBuilder<SC::Challenge>,
    quotient_chunks_domains: &[Domain],
    quotient_chunks: &[Vec<Target>],
    zeta: Target,
    pcs: &SC::Pcs,
) -> Target
where
    SC::Pcs: RecursivePcs<SC, InputProof, OpeningProof, Comm, Domain>,
    SC::Challenge: PrimeCharacteristicRing,
{
    // Compute the Lagrange interpolation coefficients.
    let zps = compute_quotient_chunk_products::<SC, _, _, _, _>(
        circuit,
        quotient_chunks_domains,
        zeta,
        pcs,
    );

    // Combine chunk evaluations with their interpolation coefficients.
    compute_quotient_evaluation::<SC>(circuit, quotient_chunks, &zps)
}

/// Computes a vanishing polynomial evaluation as a **native field operation** (outside the circuit).
///
/// This helper function evaluates `Z_H(point) = (point/g)^|H| - 1` for a domain `H` as a
/// regular field computation, producing a constant that can be embedded in the circuit.
///
/// # Explainer
///
/// When computing Lagrange interpolation coefficients `L_i(ζ) = ∏_{j≠i} [Z_j(ζ) / Z_j(g_i)]`,
/// the denominator terms `Z_j(g_i)` are **field constants** (not dependent on witness values).
/// By computing them natively, we:
///
/// 1. **Eliminate circuit operations**: No constraints needed for these evaluations
/// 2. **Reduce proof size**: Fewer gates and witness elements
/// 3. **Improve verification time**: Fewer operations to verify
///
/// This single optimization transforms O(N^2·log|D|) circuit operations into O(N^2·log|D|)
/// native operations that are "free" from the circuit's perspective.
///
/// # Mathematical Formula
///
/// For a multiplicative coset `H = {g, gω, ..., gω^(n-1)}`:
///
/// ```text
/// Z_H(point) = (point/g)^n - 1
/// ```
///
/// where `n = |H|` and `g` is the coset generator.
///
/// # Arguments
///
/// * `pcs` - PCS providing domain metadata (generator and size).
/// * `domain` - The domain `H` for which to compute the vanishing polynomial.
/// * `point` - The evaluation point.
///
/// # Returns
///
/// The field element `Z_H(point)`.
fn vanishing_poly_at_point_native<
    SC: StarkGenericConfig,
    InputProof: Recursive<SC::Challenge>,
    OpeningProof: Recursive<SC::Challenge>,
    Comm: Recursive<SC::Challenge>,
    Domain,
>(
    pcs: &SC::Pcs,
    domain: &Domain,
    point: SC::Challenge,
) -> SC::Challenge
where
    SC::Pcs: RecursivePcs<SC, InputProof, OpeningProof, Comm, Domain>,
{
    // Normalize: compute point/g where g is the coset generator
    let normalized = point * pcs.first_point(domain).inverse();

    // Exponentiate: compute (point/g)^n where n = 2^(log_size)
    let power = normalized.exp_power_of_2(pcs.log_size(domain));

    // Subtract: Z_H(point) = (point/g)^n - 1
    power - SC::Challenge::ONE
}

/// Computes Lagrange interpolation coefficients with O(N) in-circuit complexity.
///
/// This function computes the product terms needed for Lagrange interpolation:
///
/// ```text
/// L_i(ζ) = [∏_j Z_j(ζ) / Z_i(ζ)] · [1 / ∏_{j≠i} Z_j(g_i)]
/// ```
///
/// We split this into two parts:
/// 1. **In-circuit numerator**: `∏_j Z_j(ζ) / Z_i(ζ)` - computed using total product
/// 2. **Pre-computed denominator**: `1 / ∏_{j≠i} Z_j(g_i)` - computed as native field constant
///
/// # Arguments
///
/// * `circuit` - The circuit builder.
/// * `quotient_chunks_domains` - All quotient chunk domains.
/// * `zeta` - The out-of-domain evaluation point.
/// * `pcs` - The polynomial commitment scheme.
///
/// # Returns
///
/// A vector `[L_0(ζ), L_1(ζ), ..., L_{n-1}(ζ)]` of Lagrange interpolation coefficients.
fn compute_quotient_chunk_products<
    SC: StarkGenericConfig,
    InputProof: Recursive<SC::Challenge>,
    OpeningProof: Recursive<SC::Challenge>,
    Comm: Recursive<SC::Challenge>,
    Domain: Copy,
>(
    circuit: &mut CircuitBuilder<SC::Challenge>,
    quotient_chunks_domains: &[Domain],
    zeta: Target,
    pcs: &SC::Pcs,
) -> Vec<Target>
where
    SC::Pcs: RecursivePcs<SC, InputProof, OpeningProof, Comm, Domain>,
    SC::Challenge: PrimeCharacteristicRing,
{
    if quotient_chunks_domains.is_empty() {
        return Vec::new();
    }

    // Phase 1: In-circuit computation of numerators

    // Compute Z_j(ζ) for all domains j in-circuit.
    //
    // This is the only place we evaluate vanishing polynomials at ζ in the circuit.
    // Cost: O(N·log|D|) constraints
    let vp_zeta_values = quotient_chunks_domains
        .iter()
        .map(|&domain| {
            vanishing_poly_at_point_circuit::<SC, _, _, _, _>(pcs, &domain, zeta, circuit)
        })
        .collect_vec();

    // Compute the total product ∏_j Z_j(ζ).
    //
    // Cost: O(N) multiplications
    let total_vp_zeta_product = circuit.mul_many(&vp_zeta_values);

    // Phase 2: Native pre-computation of denominator constants

    // Pre-compute the denominator constants ∏_{j≠i} Z_j(g_i) OUTSIDE the circuit.
    //
    // Why this works: g_i (first points) are public constants, so Z_j(g_i) are constants.
    // We compute them as native field operations and embed them as circuit constants.
    //
    // Cost: O(N^2·log|D|) native field operations (no circuit cost!)
    let den_constants = quotient_chunks_domains
        .iter()
        .enumerate()
        .map(|(i, &domain_i)| {
            let fp_i = pcs.first_point(&domain_i);
            quotient_chunks_domains
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| j != i)
                    .fold(SC::Challenge::ONE, |acc, (_, &domain_j)| {
                        // Compute Z_j(g_i) natively (no circuit operations!)
                        acc * vanishing_poly_at_point_native::<
                            SC,
                            InputProof,
                            OpeningProof,
                            Comm,
                            Domain,
                        >(pcs, &domain_j, fp_i)
                    })
        })
        .collect_vec();

    // Phase 3: In-circuit combination

    // Pre-lift all denominator constants once before the loop
    let den_targets: Vec<_> = den_constants
        .iter()
        .map(|&c| circuit.define_const(c))
        .collect();

    // For each chunk i, compute L_i(ζ) = [∏_j Z_j(ζ) / Z_i(ζ)] / den_i
    //
    // Cost: O(N) divisions (2 divisions per chunk)
    vp_zeta_values
        .iter()
        .enumerate()
        .map(|(i, &vp_zeta_i)| {
            // Numerator: (∏_j Z_j(ζ)) / Z_i(ζ)
            //
            // Reuse the total product and divide by the i-th vanishing polynomial
            let num = circuit.div(total_vp_zeta_product, vp_zeta_i);

            // Denominator: Pre-computed constant ∏_{j≠i} Z_j(g_i)
            //
            // Simply embed the constant into the circuit (no computation!)
            let den = den_targets[i];

            // Final Lagrange coefficient: L_i(ζ) = numerator / denominator
            circuit.div(num, den)
        })
        .collect()
}

/// Computes the quotient polynomial evaluation.
///
/// Given chunk coefficients and Lagrange interpolation coefficients, this computes:
///
/// ```text
/// Q(ζ) = ∑_i Q_i(·) · L_i(ζ)
/// ```
///
/// where each `Q_i(·)` is evaluated from its extension field basis representation:
///
/// ```text
/// Q_i(·) = ∑_j e_j · chunk_i[j]
/// ```
///
/// # Arguments
///
/// * `circuit` - The circuit builder.
/// * `opened_quotient_chunks` - Coefficient vectors in extension field basis.
/// * `zps` - Lagrange interpolation coefficients `[L_0(ζ), ..., L_{n-1}(ζ)]`.
///
/// # Returns
///
/// A `Target` representing `Q(ζ)`.
fn compute_quotient_evaluation<SC: StarkGenericConfig>(
    circuit: &mut CircuitBuilder<SC::Challenge>,
    opened_quotient_chunks: &[Vec<Target>],
    zps: &[Target],
) -> Target
where
    SC::Challenge: PrimeCharacteristicRing,
{
    // Get the extension field degree.
    let d = SC::Challenge::DIMENSION;

    // Handle edge cases: empty chunks or trivial extension
    if d == 0 || opened_quotient_chunks.is_empty() {
        return circuit.define_const(SC::Challenge::ZERO);
    }

    // Phase 1: Pre-compute extension field basis elements

    // Compute the basis elements [e_0, e_1, ..., e_{d-1}] once and reuse for all chunks.
    //
    // Cost: O(d) - done once
    let basis_targets: Vec<_> = (0..d)
        .map(|i| {
            let basis_elem =
                SC::Challenge::ith_basis_element(i).expect("Basis index should be in range [0, d)");
            circuit.define_const(basis_elem)
        })
        .collect();

    // Phase 2: Evaluate each chunk polynomial using inner product

    // For each chunk, compute Q_i = ∑_j e_j · chunk_i[j] using inner product.
    //
    // Cost: O(n·d) multiply-adds
    let chunk_evals = opened_quotient_chunks
        .iter()
        .map(|chunk| {
            // Validate chunk length in debug builds
            debug_assert_eq!(
                chunk.len(),
                d,
                "Chunk length must match field extension degree"
            );

            // Compute Q_i = <chunk, basis> = ∑_j chunk[j] · e_j
            circuit.inner_product(chunk, &basis_targets)
        })
        .collect_vec();

    // Phase 3: Combine chunk evaluations with Lagrange coefficients

    // Compute the final sum Q(ζ) = ∑_i Q_i · L_i(ζ) using inner product.
    //
    // Cost: O(n) multiply-adds
    circuit.inner_product(&chunk_evals, zps)
}

/// Computes a vanishing polynomial evaluation **in-circuit**.
///
/// This function adds circuit constraints to compute `Z_H(point) = (point/g)^n - 1`
/// for a domain `H`, where the result is a circuit variable (not a constant).
///
/// # Mathematical Formula
///
/// For a multiplicative coset `H = {g, gω, ..., gω^(n-1)}`:
///
/// ```text
/// Z_H(point) = (point/g)^n - 1
/// ```
///
/// where `n = |H|` and `g` is the coset generator.
///
/// # Arguments
///
/// * `pcs` - PCS providing domain metadata.
/// * `domain` - The domain `H`.
/// * `point` - The evaluation point (a circuit variable).
/// * `circuit` - The circuit builder.
///
/// # Returns
///
/// A `Target` representing `Z_H(point)` in the circuit.
fn vanishing_poly_at_point_circuit<
    SC: StarkGenericConfig,
    InputProof: Recursive<SC::Challenge>,
    OpeningProof: Recursive<SC::Challenge>,
    Comm: Recursive<SC::Challenge>,
    Domain,
>(
    pcs: &SC::Pcs,
    domain: &Domain,
    point: Target,
    circuit: &mut CircuitBuilder<SC::Challenge>,
) -> Target
where
    SC::Pcs: RecursivePcs<SC, InputProof, OpeningProof, Comm, Domain>,
{
    // Normalize: compute point/g where g is the coset generator
    //
    // Cost: 1 multiplication constraint
    let inv = circuit.define_const(pcs.first_point(domain).inverse());
    let mul = circuit.mul(point, inv);

    // Exponentiate: compute (point/g)^n where n = 2^(log_size)
    //
    // Cost: log_2(n) squaring constraints
    let exp = circuit.exp_power_of_2(mul, pcs.log_size(domain));

    // Subtract: Z_H(point) = (point/g)^n - 1
    //
    // Cost: 1 subtraction constraint
    let one = circuit.define_const(SC::Challenge::ONE);
    circuit.sub(exp, one)
}
