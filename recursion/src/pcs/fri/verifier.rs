#![allow(clippy::too_many_arguments)]
#![allow(clippy::option_if_let_else)]

use alloc::collections::btree_map::BTreeMap;
use alloc::string::ToString;
use alloc::vec::Vec;
use alloc::{format, vec};
use core::iter;

use hashbrown::HashMap;
use p3_circuit::ops::Poseidon2Config;
use p3_circuit::{CircuitBuilder, NonPrimitiveOpId};
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::{BasedVectorSpace, ExtensionField, Field, PrimeField64, TwoAdicField};
use p3_matrix::Dimensions;
use p3_util::zip_eq::zip_eq;

use super::{FriProofTargets, InputProofTargets};
use crate::Target;
use crate::pcs::{verify_batch_circuit, verify_batch_circuit_from_extension_opened};
use crate::traits::{ComsWithOpeningsTargets, Recursive, RecursiveExtensionMmcs, RecursiveMmcs};
use crate::verifier::{ObservableCommitment, VerificationError};

/// Pack lifted base field targets into packed extension field targets for MMCS verification.
///
/// Converts N targets (each representing a lifted base field element `EF([v, 0, 0, 0])`)
/// into ceil(N / EF::DIMENSION) targets (each representing a packed extension element).
///
/// For `BinomialExtensionField<F, D>`, the packing is:
/// `packed[i] = lifted[i*D] + lifted[i*D+1]*X + lifted[i*D+2]*X^2 + ...`
/// where `X` is the extension basis element.
fn pack_lifted_to_ext<F, EF>(builder: &mut CircuitBuilder<EF>, lifted: &[Target]) -> Vec<Target>
where
    F: Field,
    EF: ExtensionField<F> + BasedVectorSpace<F>,
{
    if lifted.is_empty() {
        return Vec::new();
    }

    let d = EF::DIMENSION;

    // Get the extension basis elements: {1, X, X^2, ..., X^(D-1)}
    let basis: Vec<EF> = (0..d)
        .map(|i| {
            let mut coeffs = vec![F::ZERO; d];
            coeffs[i] = F::ONE;
            EF::from_basis_coefficients_slice(&coeffs).expect("valid basis element")
        })
        .collect();

    // Lift basis elements to circuit constants once and reuse across all chunks.
    let basis_consts: Vec<Target> = basis.iter().map(|b| builder.define_const(*b)).collect();

    // Add a zero constant for padding partial chunks
    let zero = builder.define_const(EF::ZERO);

    lifted
        .chunks(d)
        .map(|chunk| {
            // packed = chunk[0]*basis[0] + chunk[1]*basis[1] + ... + chunk[D-1]*basis[D-1]
            // Since basis[0] = 1, start with chunk[0]
            let mut packed = chunk[0];
            for j in 1..d {
                let val = if j < chunk.len() { chunk[j] } else { zero };
                packed = builder.mul_add(val, basis_consts[j], packed);
            }
            packed
        })
        .collect()
}

/// Build MMCS commitment-cap rows from Fiat–Shamir observation targets (lifted base scalars).
///
/// For D=4-style configs, adjacent lifted coordinates are packed into one extension element via
/// [`pack_lifted_to_ext`]. For D=1 width-16 configs over a high-degree challenge extension, the
/// inner hash absorbs one base element per rate slot (lifted scalars); cap rows must **not** be
/// packed, so each row has `rate_ext` targets matching the D=1 per-base MMCS hash path.
fn commitment_cap_rows_from_lifted<F, EF>(
    builder: &mut CircuitBuilder<EF>,
    perm_config: Poseidon2Config,
    lifted: &[Target],
) -> Vec<Vec<Target>>
where
    F: Field,
    EF: ExtensionField<F> + BasedVectorSpace<F>,
{
    let rate_ext = perm_config.rate_ext();
    if perm_config.d() == 1 && EF::DIMENSION > 1 {
        debug_assert_eq!(
            lifted.len() % rate_ext,
            0,
            "lifted cap length should be a multiple of rate_ext"
        );
        lifted.chunks(rate_ext).map(|c| c.to_vec()).collect()
    } else {
        let packed = pack_lifted_to_ext::<F, EF>(builder, lifted);
        packed.chunks(rate_ext).map(|c| c.to_vec()).collect()
    }
}

/// Per-phase configuration for the FRI fold chain.
#[derive(Clone, Debug)]
pub struct FoldPhaseConfig {
    pub beta: Target,
    /// Packed extension field sibling evaluations (arity - 1 values).
    pub siblings: Vec<Target>,
    pub roll_in: Option<Target>,
}

/// Optimized one-hot computation for 2 bits.
fn one_hot_from_two_bits<EF: Field>(
    builder: &mut CircuitBuilder<EF>,
    b0: Target,
    b1: Target,
) -> [Target; 4] {
    let one = builder.define_const(EF::ONE);
    let nb0 = builder.sub(one, b0);
    let nb1 = builder.sub(one, b1);

    let h0 = builder.mul(nb0, nb1); // 00
    let h1 = builder.mul(b0, nb1); // 01
    let h2 = builder.mul(nb0, b1); // 10
    let h3 = builder.mul(b0, b1); // 11

    [h0, h1, h2, h3]
}

/// Optimized one-hot computation for 3 bits.
fn one_hot_from_three_bits<EF: Field>(
    builder: &mut CircuitBuilder<EF>,
    b0: Target,
    b1: Target,
    b2: Target,
) -> [Target; 8] {
    let one = builder.define_const(EF::ONE);
    let nb0 = builder.sub(one, b0);
    let nb1 = builder.sub(one, b1);
    let nb2 = builder.sub(one, b2);

    // Shared products for (b1, b2).
    let t00 = builder.mul(nb1, nb2); // b1=0, b2=0
    let t01 = builder.mul(nb1, b2); // b1=0, b2=1
    let t10 = builder.mul(b1, nb2); // b1=1, b2=0
    let t11 = builder.mul(b1, b2); // b1=1, b2=1

    // Index j = b0 + 2*b1 + 4*b2 (little-endian).
    let h0 = builder.mul(nb0, t00); // 0,0,0 -> j=0
    let h1 = builder.mul(b0, t00); // 1,0,0 -> j=1
    let h2 = builder.mul(nb0, t10); // 0,1,0 -> j=2
    let h3 = builder.mul(b0, t10); // 1,1,0 -> j=3
    let h4 = builder.mul(nb0, t01); // 0,0,1 -> j=4
    let h5 = builder.mul(b0, t01); // 1,0,1 -> j=5
    let h6 = builder.mul(nb0, t11); // 0,1,1 -> j=6
    let h7 = builder.mul(b0, t11); // 1,1,1 -> j=7

    [h0, h1, h2, h3, h4, h5, h6, h7]
}

/// Optimized one-hot computation for 4 bits, using two 2-bit one-hots.
#[unroll::unroll_for_loops]
fn one_hot_from_four_bits<EF: Field>(
    builder: &mut CircuitBuilder<EF>,
    bits: &[Target],
) -> Vec<Target> {
    debug_assert_eq!(bits.len(), 4);
    let low = one_hot_from_two_bits(builder, bits[0], bits[1]);
    let high = one_hot_from_two_bits(builder, bits[2], bits[3]);

    let mut result = Vec::with_capacity(16);
    for j in 0..16 {
        let low_idx = j & 3;
        let high_idx = j >> 2;
        let val = builder.mul(low[low_idx], high[high_idx]);
        result.push(val);
    }
    result
}

/// Compute a one-hot encoding from `log_arity` index bits.
/// Returns a vector of `2^log_arity` targets where `result[j] = 1` iff j matches the
/// integer value of the input bits (little-endian).
fn one_hot_from_bits<EF: Field>(builder: &mut CircuitBuilder<EF>, bits: &[Target]) -> Vec<Target> {
    let log_arity = bits.len();
    let arity = 1usize << log_arity;

    match log_arity {
        0 => {
            // Degenerate case: arity 1, always index 0.
            vec![builder.define_const(EF::ONE)]
        }
        1 => {
            // One bit: [!b0, b0]
            let one = builder.define_const(EF::ONE);
            let b0 = bits[0];
            let nb0 = builder.sub(one, b0);
            vec![nb0, b0]
        }
        2 => {
            let [h0, h1, h2, h3] = one_hot_from_two_bits(builder, bits[0], bits[1]);
            vec![h0, h1, h2, h3]
        }
        3 => {
            let [h0, h1, h2, h3, h4, h5, h6, h7] =
                one_hot_from_three_bits(builder, bits[0], bits[1], bits[2]);
            vec![h0, h1, h2, h3, h4, h5, h6, h7]
        }
        4 => one_hot_from_four_bits(builder, bits),
        _ => {
            let one = builder.define_const(EF::ONE);
            // Precompute negations of bits once to avoid rebuilding `1 - bit` inside the inner loop for every index j.
            let not_bits: Vec<Target> = bits.iter().map(|&bit| builder.sub(one, bit)).collect();

            let mut one_hot = Vec::with_capacity(arity);
            for j in 0..arity {
                let mut product = one;
                for (k, &bit) in bits.iter().enumerate() {
                    if (j >> k) & 1 == 1 {
                        product = builder.mul(product, bit);
                    } else {
                        product = builder.mul(product, not_bits[k]);
                    }
                }
                one_hot.push(product);
            }
            one_hot
        }
    }
}

/// Reconstruct the full evaluation row from `folded` + `siblings` using the index bits.
///
/// `index_in_group_bits` are the `log_arity` lowest bits of the current start_index.
/// The native verifier does:
///   evals[index_in_group] = folded; evals[j] = siblings[...] for j != index_in_group
///
/// This circuit version uses a one-hot encoding to place values correctly.
fn reconstruct_evals<EF: Field>(
    builder: &mut CircuitBuilder<EF>,
    folded: Target,
    siblings: &[Target],
    index_in_group_bits: &[Target],
) -> Vec<Target> {
    builder.push_scope("fri_reconstruct_evals");
    let log_arity = index_in_group_bits.len();
    let arity = 1usize << log_arity;
    debug_assert_eq!(siblings.len(), arity - 1);

    let evals = match log_arity {
        // Arity 1
        0 => vec![folded],

        // Arity 2: one bit b0, siblings[0]
        //
        // idx = 0: [folded, siblings[0]]
        // idx = 1: [siblings[0], folded]
        1 => {
            let b0 = index_in_group_bits[0];
            let s0 = siblings[0];

            let e0 = builder.select(b0, s0, folded); // if b0==0 -> folded, else s0
            let e1 = builder.select(b0, folded, s0); // if b0==0 -> s0,     else folded

            vec![e0, e1]
        }

        // Arity 4: two bits [b0, b1] (little-endian), idx = b0 + 2*b1
        //
        // siblings = [s0, s1, s2]
        //
        // Generic semantics (from the original one_hot + cum implementation):
        //   idx=0: [folded, s0, s1, s2]
        //   idx=1: [s0,     folded, s1, s2]
        //   idx=2: [s0,     s1,     folded, s2]
        //   idx=3: [s0,     s1,     s2,     folded]
        2 => {
            let b0 = index_in_group_bits[0];
            let b1 = index_in_group_bits[1];

            let s0 = siblings[0];
            let s1 = siblings[1];
            let s2 = siblings[2];

            // One-hot over {0,1,2,3}
            let [h0, h1, h2, h3] = one_hot_from_two_bits(builder, b0, b1);

            // e0: folded if idx=0, else s0
            //   => e0 = s0 + h0 * (folded - s0)
            let diff_f_s0 = builder.sub(folded, s0);
            let e0 = builder.mul_add(h0, diff_f_s0, s0);

            // e1: s0 if idx=0, folded if idx=1, s1 if idx∈{2,3}
            //   => e1 = f*h1 + s0*h0 + s1*(h2 + h3)
            let h2_plus_h3 = builder.add(h2, h3);
            let s1_term_for_e1 = builder.mul(s1, h2_plus_h3);
            let f_h1 = builder.mul(folded, h1);
            let tmp_e1 = builder.mul_add(s0, h0, f_h1);
            let e1 = builder.add(tmp_e1, s1_term_for_e1);

            // e2: s1 if idx∈{0,1}, folded if idx=2, s2 if idx=3
            //   => e2 = f*h2 + s1*(h0 + h1) + s2*h3
            let h0_plus_h1 = builder.add(h0, h1);
            let s1_term_for_e2 = builder.mul(s1, h0_plus_h1);
            let tmp_e2 = builder.mul_add(folded, h2, s1_term_for_e2);
            let e2 = builder.mul_add(s2, h3, tmp_e2);

            // e3: s2 if idx∈{0,1,2}, folded if idx=3
            //   => e3 = s2 + h3 * (folded - s2)
            let diff_f_s2 = builder.sub(folded, s2);
            let e3 = builder.mul_add(h3, diff_f_s2, s2);

            vec![e0, e1, e2, e3]
        }

        // Arity 8: three bits [b0, b1, b2] (little-endian), idx = b0 + 2*b1 + 4*b2
        //
        // siblings = [s0, s1, s2, s3, s4, s5, s6]
        //
        // Same placement as one_hot + cum + select, in closed form:
        //   e_j = h_j * folded + siblings[j] * sum_{k>j} h_k + siblings[j-1] * sum_{k<j} h_k
        // with the missing sibling terms treated as zero (j = 0 or j = 7).
        3 => {
            let b0 = index_in_group_bits[0];
            let b1 = index_in_group_bits[1];
            let b2 = index_in_group_bits[2];

            let s0 = siblings[0];
            let s1 = siblings[1];
            let s2 = siblings[2];
            let s3 = siblings[3];
            let s4 = siblings[4];
            let s5 = siblings[5];
            let s6 = siblings[6];

            let [h0, h1, h2, h3, h4, h5, h6, h7] = one_hot_from_three_bits(builder, b0, b1, b2);

            // P[j] = sum_{k < j} h_k for j = 1..=7
            let p1 = h0;
            let p2 = builder.add(p1, h1);
            let p3 = builder.add(p2, h2);
            let p4 = builder.add(p3, h3);
            let p5 = builder.add(p4, h4);
            let p6 = builder.add(p5, h5);
            let p7 = builder.add(p6, h6);

            // S[j] = sum_{k > j} h_k for j = 0..=6
            let su6 = h7;
            let su5 = builder.add(h6, su6);
            let su4 = builder.add(h5, su5);
            let su3 = builder.add(h4, su4);
            let su2 = builder.add(h3, su3);
            let su1 = builder.add(h2, su2);
            let su0 = builder.add(h1, su1);

            let e0_sib = builder.mul(s0, su0);
            let e0 = builder.mul_add(h0, folded, e0_sib);

            let e1_lo = builder.mul(s0, p1);
            let e1_sib = builder.mul_add(s1, su1, e1_lo);
            let e1 = builder.mul_add(h1, folded, e1_sib);

            let e2_lo = builder.mul(s1, p2);
            let e2_sib = builder.mul_add(s2, su2, e2_lo);
            let e2 = builder.mul_add(h2, folded, e2_sib);

            let e3_lo = builder.mul(s2, p3);
            let e3_sib = builder.mul_add(s3, su3, e3_lo);
            let e3 = builder.mul_add(h3, folded, e3_sib);

            let e4_lo = builder.mul(s3, p4);
            let e4_sib = builder.mul_add(s4, su4, e4_lo);
            let e4 = builder.mul_add(h4, folded, e4_sib);

            let e5_lo = builder.mul(s4, p5);
            let e5_sib = builder.mul_add(s5, su5, e5_lo);
            let e5 = builder.mul_add(h5, folded, e5_sib);

            let e6_lo = builder.mul(s5, p6);
            let e6_sib = builder.mul_add(s6, su6, e6_lo);
            let e6 = builder.mul_add(h6, folded, e6_sib);

            let e7_sib = builder.mul(s6, p7);
            let e7 = builder.mul_add(h7, folded, e7_sib);

            vec![e0, e1, e2, e3, e4, e5, e6, e7]
        }

        // Generic path for larger arities
        _ => {
            let one_hot = one_hot_from_bits(builder, index_in_group_bits);

            // Compute cumulative sum: cum[j] = sum(one_hot[0..=j]) ∈ {0, 1}
            let mut cum = Vec::with_capacity(arity);
            cum.push(one_hot[0]);
            for j in 1..arity {
                cum.push(builder.add(cum[j - 1], one_hot[j]));
            }

            let mut evals = Vec::with_capacity(arity);
            for j in 0..arity {
                let left_idx = if j > 0 { j - 1 } else { 0 };
                let right_idx = if j < arity - 1 { j } else { arity - 2 };
                let actual_sibling =
                    builder.select(cum[j], siblings[left_idx], siblings[right_idx]);
                let eval_j = builder.select(one_hot[j], folded, actual_sibling);
                evals.push(eval_j);
            }
            evals
        }
    };

    builder.pop_scope();
    evals
}

/// Compute the subgroup evaluation points for a single FRI fold phase.
///
/// Returns `(xs, subgroup_start)` where:
/// - `xs[i] = subgroup_start * omega^{br(i)}` are the `arity` evaluation points
///   in bit-reversed order,
/// - `omega = two_adic_generator(log_arity)`,
/// - `subgroup_start = two_adic_generator(log_folded_height + log_arity)^{rev(parent_index)}`.
///
/// When `precomputed_subgroup_start` is `Some`, the select-mul chain for
/// `subgroup_start` is skipped and the provided value is used directly.
fn compute_subgroup_points<F, EF>(
    builder: &mut CircuitBuilder<EF>,
    log_arity: usize,
    subgroup_start: Target,
) -> (Vec<Target>, Target)
where
    F: Field + TwoAdicField,
    EF: ExtensionField<F>,
{
    builder.push_scope("fri_compute_subgroup_points");

    let arity = 1usize << log_arity;

    // Compute xs[i] = subgroup_start * omega^{br(i)}
    let omega = F::two_adic_generator(log_arity);
    let omega_br_consts: Vec<Target> = (0..arity)
        .map(|i| {
            let br_i = p3_util::reverse_bits_len(i, log_arity);
            let omega_br = omega.exp_u64(br_i as u64);
            builder.define_const(EF::from(omega_br))
        })
        .collect();

    let mut xs = Vec::with_capacity(arity);
    for &omega_br_const in omega_br_consts.iter() {
        let xi = builder.mul(subgroup_start, omega_br_const);
        xs.push(xi);
    }

    builder.pop_scope();
    (xs, subgroup_start)
}

/// Precompute `subgroup_start` for every FRI phase within a single query.
///
/// All phases compute `g_i^{rev(parent_index_i)}` where
/// `g_i = two_adic_generator(log_current_height_i)`. Because the reversed parent
/// bits for phase `i` are a prefix of phase 0's bits, and
/// `g_i = g_0^{2^{cumulative_bits_i}}`, we derive later phases from the
/// intermediate of phase 0's chain:
/// `subgroup_start_i = (g_0^{N_i})^{2^{cumulative_bits_i}}`.
fn precompute_subgroup_starts<F, EF>(
    builder: &mut CircuitBuilder<EF>,
    index_bits: &[Target],
    log_max_height: usize,
    log_arities: &[usize],
    cumulative_bits: &[usize],
) -> Vec<Target>
where
    F: Field + TwoAdicField,
    EF: ExtensionField<F>,
{
    let num_phases = log_arities.len();
    let one = builder.define_const(EF::ONE);

    // log_folded_height[i] = log_max_height - cumulative_bits[i+1]
    let log_folded_heights: Vec<usize> = (0..num_phases)
        .map(|i| log_max_height - cumulative_bits[i + 1])
        .collect();

    let max_chain_len = log_folded_heights[0];

    if max_chain_len == 0 {
        builder.pop_scope();
        return vec![one; num_phases];
    }

    let g_0 = F::two_adic_generator(log_max_height);
    let powers_of_g: Vec<_> = iter::successors(Some(g_0), |&prev| Some(prev.square()))
        .take(max_chain_len)
        .map(|p| builder.define_const(EF::from(p)))
        .collect();

    let parent_offset_0 = cumulative_bits[1]; // = log_arities[0]

    let mut capture_at: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (i, &lf) in log_folded_heights
        .iter()
        .enumerate()
        .take(num_phases)
        .skip(1)
    {
        if lf > 0 {
            capture_at.entry(lf).or_default().push(i);
        }
    }

    let mut g_pow = one;
    let mut result = vec![one; num_phases];

    for j in 0..max_chain_len {
        let bit = index_bits[parent_offset_0 + max_chain_len - 1 - j];
        let multiplier = builder.select(bit, powers_of_g[j], one);
        g_pow = builder.mul(g_pow, multiplier);

        let bits_done = j + 1;
        if let Some(phase_indices) = capture_at.get(&bits_done) {
            for &phase_i in phase_indices {
                result[phase_i] = builder.exp_power_of_2(g_pow, cumulative_bits[phase_i]);
            }
        }
    }

    // Phase 0: full chain, cumulative_bits[0] = 0, no squaring.
    result[0] = g_pow;

    result
}

/// Precompute and cache powers `beta^{2^k}` for all fold phases.
fn precompute_beta_powers_per_phase<EF: Field>(
    builder: &mut CircuitBuilder<EF>,
    betas: &[Target],
    log_arities: &[usize],
) -> Vec<Target> {
    builder.push_scope("fri_precompute_betas");

    debug_assert_eq!(betas.len(), log_arities.len());
    let result = betas
        .iter()
        .zip(log_arities.iter())
        .map(|(&beta, &log_arity)| builder.exp_power_of_2(beta, log_arity))
        .collect();

    builder.pop_scope();
    result
}

/// Single arity-2 fold at a point: given (e0, e1) and evaluation point beta,
/// returns the folded value using (e0 + e1)/2 + (e1 - e0)*beta/(2*x0).
fn arity2_fold_at_point<EF: Field>(
    builder: &mut CircuitBuilder<EF>,
    e0: Target,
    e1: Target,
    beta: Target,
    x0: Target,
) -> Target {
    let neg_half = builder.define_const(EF::NEG_ONE * EF::ONE.halve());
    let inv = builder.div(neg_half, x0);
    let e1_minus_e0 = builder.sub(e1, e0);
    let beta_minus_x0 = builder.sub(beta, x0);
    let t = builder.mul(beta_minus_x0, e1_minus_e0);
    builder.mul_add(t, inv, e0)
}

/// Perform a single FRI fold phase with arbitrary arity.
///
/// For log_arity > 1 we use k sequential arity-2 folds (beta, beta^2, ...)
/// instead of one Lagrange interpolation, reducing batch inversions to one per step.
/// Arity 4 and 8 use the same fold schedule with the loop unrolled (no subgroup `Vec`s).
///
/// When `precomputed_evals` is `Some`, those evals are reused instead of
/// rebuilding them via `reconstruct_evals`.
fn fold_one_phase<F, EF>(
    builder: &mut CircuitBuilder<EF>,
    folded: Target,
    siblings: &[Target],
    beta: Target,
    index_bits: &[Target],
    bits_consumed: usize,
    log_arity: usize,
    roll_in: Option<Target>,
    precomputed_beta_pow: Option<Target>,
    precomputed_evals: Option<&[Target]>,
    precomputed_subgroup_start: Target,
) -> Target
where
    F: Field + TwoAdicField,
    EF: ExtensionField<F>,
{
    builder.push_scope("fri_fold_one_phase");

    let index_in_group_bits = &index_bits[bits_consumed..bits_consumed + log_arity];

    // For arity 2, use the optimized formula
    if log_arity == 1 {
        let sibling = siblings[0];
        let one = builder.define_const(EF::ONE);
        let two = builder.define_const(EF::TWO);
        let neg_one = builder.define_const(EF::NEG_ONE);
        let neg_half = builder.define_const(EF::NEG_ONE * EF::ONE.halve());
        let sibling_is_right = builder.sub(one, index_bits[bits_consumed]);

        let e0 = builder.select(sibling_is_right, folded, sibling);
        let x0 = precomputed_subgroup_start;
        let inv = builder.div(neg_half, x0);

        let d = builder.sub(sibling, folded);
        let two_b_m1 = builder.mul_add(two, sibling_is_right, neg_one);
        let e1_minus_e0 = builder.mul(two_b_m1, d);

        let beta_minus_x0 = builder.sub(beta, x0);
        let t = builder.mul(beta_minus_x0, e1_minus_e0);
        let mut new_folded = builder.mul_add(t, inv, e0);

        if let Some(ro) = roll_in {
            let beta_sq = precomputed_beta_pow.unwrap_or_else(|| builder.mul(beta, beta));
            new_folded = builder.mul_add(beta_sq, ro, new_folded);
        }
        builder.pop_scope();
        return new_folded;
    }

    let owned_evals;
    let evals: &[Target] = match precomputed_evals {
        Some(e) => e,
        None => {
            owned_evals = reconstruct_evals(builder, folded, siblings, index_in_group_bits);
            &owned_evals
        }
    };

    // Unrolled fold tree for arity 4 / 8: same x0 and beta schedule as the generic loop,
    // but no `compute_subgroup_points` / `data` / `omega_s_br` vectors.
    let mut new_folded = if log_arity == 2 {
        let omega = F::two_adic_generator(2);
        let ss = precomputed_subgroup_start;

        let x_at_step0 = |builder: &mut CircuitBuilder<EF>, j: usize| {
            let br = p3_util::reverse_bits_len(2 * j, 2);
            let w = builder.define_const(EF::from(omega.exp_u64(br as u64)));
            builder.mul(ss, w)
        };

        let x00 = x_at_step0(builder, 0);
        let x01 = x_at_step0(builder, 1);
        let f0 = arity2_fold_at_point::<EF>(builder, evals[0], evals[1], beta, x00);
        let f1 = arity2_fold_at_point::<EF>(builder, evals[2], evals[3], beta, x01);

        let beta2 = builder.mul(beta, beta);
        let ss2 = builder.mul(ss, ss);
        let omega_s = omega.exp_u64(1 << 1);
        let br = p3_util::reverse_bits_len(0, 1);
        let w = builder.define_const(EF::from(omega_s.exp_u64(br as u64)));
        let x_step1 = builder.mul(ss2, w);
        arity2_fold_at_point::<EF>(builder, f0, f1, beta2, x_step1)
    } else if log_arity == 3 {
        let omega = F::two_adic_generator(3);
        let ss = precomputed_subgroup_start;

        let x_at_step0 = |builder: &mut CircuitBuilder<EF>, j: usize| {
            let br = p3_util::reverse_bits_len(2 * j, 3);
            let w = builder.define_const(EF::from(omega.exp_u64(br as u64)));
            builder.mul(ss, w)
        };

        let x00 = x_at_step0(builder, 0);
        let x01 = x_at_step0(builder, 1);
        let x02 = x_at_step0(builder, 2);
        let x03 = x_at_step0(builder, 3);
        let f0 = arity2_fold_at_point::<EF>(builder, evals[0], evals[1], beta, x00);
        let f1 = arity2_fold_at_point::<EF>(builder, evals[2], evals[3], beta, x01);
        let f2 = arity2_fold_at_point::<EF>(builder, evals[4], evals[5], beta, x02);
        let f3 = arity2_fold_at_point::<EF>(builder, evals[6], evals[7], beta, x03);

        let beta2 = builder.mul(beta, beta);
        let ss2 = builder.mul(ss, ss);
        let omega_s1 = omega.exp_u64(1 << 1);

        let x_at_step1 = |builder: &mut CircuitBuilder<EF>, j: usize| {
            let br = p3_util::reverse_bits_len(2 * j, 2);
            let w = builder.define_const(EF::from(omega_s1.exp_u64(br as u64)));
            builder.mul(ss2, w)
        };
        let x10 = x_at_step1(builder, 0);
        let x11 = x_at_step1(builder, 1);
        let g0 = arity2_fold_at_point::<EF>(builder, f0, f1, beta2, x10);
        let g1 = arity2_fold_at_point::<EF>(builder, f2, f3, beta2, x11);

        let beta4 = builder.mul(beta2, beta2);
        let ss4 = builder.mul(ss2, ss2);
        let omega_s2 = omega.exp_u64(1 << 2);
        let br = p3_util::reverse_bits_len(0, 1);
        let w = builder.define_const(EF::from(omega_s2.exp_u64(br as u64)));
        let x_step2 = builder.mul(ss4, w);
        arity2_fold_at_point::<EF>(builder, g0, g1, beta4, x_step2)
    } else {
        // General path: k sequential arity-2 folds (beta, beta^2, ...) instead of
        // one Lagrange interpolation, matching the native optimization to reduce inversions.
        let (xs, subgroup_start) =
            compute_subgroup_points::<F, EF>(builder, log_arity, precomputed_subgroup_start);

        let mut subgroup_start_powers: Vec<Target> = vec![subgroup_start];
        for _ in 1..log_arity {
            let prev = subgroup_start_powers.last().copied().unwrap();
            subgroup_start_powers.push(builder.mul(prev, prev));
        }

        let omega = F::two_adic_generator(log_arity);
        let mut data: Vec<Target> = evals.to_vec();
        let mut current_beta = beta;

        for (step, ss) in subgroup_start_powers
            .into_iter()
            .enumerate()
            .take(log_arity)
        {
            let num_pairs = data.len() / 2;
            if step == 0 {
                for j in 0..num_pairs {
                    data[j] = arity2_fold_at_point::<EF>(
                        builder,
                        data[2 * j],
                        data[2 * j + 1],
                        current_beta,
                        xs[2 * j],
                    );
                }
            } else {
                let log_domain = log_arity - step;
                let omega_s = omega.exp_u64(1 << step);
                let omega_s_br: Vec<Target> = (0..num_pairs)
                    .map(|j| {
                        let br_2j = p3_util::reverse_bits_len(2 * j, log_domain);
                        let c = omega_s.exp_u64(br_2j as u64);
                        builder.define_const(EF::from(c))
                    })
                    .collect();
                for j in 0..num_pairs {
                    let x0 = builder.mul(ss, omega_s_br[j]);
                    data[j] = arity2_fold_at_point::<EF>(
                        builder,
                        data[2 * j],
                        data[2 * j + 1],
                        current_beta,
                        x0,
                    );
                }
            }
            data.truncate(num_pairs);
            if step < log_arity - 1 {
                current_beta = builder.mul(current_beta, current_beta);
            }
        }

        data[0]
    };

    // Roll-in: folded += beta^{2^log_arity} * roll_in
    if let Some(ro) = roll_in {
        let beta_pow =
            precomputed_beta_pow.unwrap_or_else(|| builder.exp_power_of_2(beta, log_arity));
        new_folded = builder.mul_add(beta_pow, ro, new_folded);
    }

    builder.pop_scope();
    new_folded
}

/// Perform the full FRI fold chain with variable arity per phase.
fn fold_chain_circuit<F, EF>(
    builder: &mut CircuitBuilder<EF>,
    initial_folded_eval: Target,
    index_bits: &[Target],
    phases: &[FoldPhaseConfig],
    log_arities: &[usize],
    cumulative_bits: &[usize],
    beta_pows_per_phase: &[Target],
) -> Target
where
    F: Field + TwoAdicField,
    EF: ExtensionField<F>,
{
    builder.push_scope("fold_chain_circuit");

    let log_max_height = index_bits.len();

    let subgroup_starts = precompute_subgroup_starts::<F, EF>(
        builder,
        index_bits,
        log_max_height,
        log_arities,
        cumulative_bits,
    );

    let mut folded = initial_folded_eval;
    let mut bits_consumed = 0usize;

    for (i, phase) in phases.iter().enumerate() {
        let log_arity = log_arities[i];
        folded = fold_one_phase::<F, EF>(
            builder,
            folded,
            &phase.siblings,
            phase.beta,
            index_bits,
            bits_consumed,
            log_arity,
            phase.roll_in,
            Some(beta_pows_per_phase[i]),
            None,
            subgroup_starts[i],
        );
        bits_consumed += log_arity;
    }

    builder.pop_scope();
    folded
}

/// Evaluate a polynomial at a point `x` using Horner's method.
/// Given coefficients [c0, c1, c2, ...], compute `p(x) = c0 + x*(c1 + x*(c2 + ...))`.
fn evaluate_polynomial<EF: Field>(
    builder: &mut CircuitBuilder<EF>,
    coefficients: &[Target],
    point: Target,
) -> Target {
    builder.push_scope("evaluate_polynomial");

    assert!(
        !coefficients.is_empty(),
        "we should have at least a constant polynomial"
    );
    if coefficients.len() == 1 {
        return coefficients[0];
    }

    let zero = builder.define_const(EF::ZERO);
    let mut result = zero;
    for &coeff in coefficients.iter().rev() {
        result = builder.horner_acc_step(result, point, coeff, zero);
    }

    builder.pop_scope(); // close `evaluate_polynomial` scope
    result
}

/// Precompute powers `g^{2^j}` (as circuit constants) for a two-adic generator of the
/// given height. The result can be shared across queries.
fn precompute_two_adic_powers<F, EF>(
    builder: &mut CircuitBuilder<EF>,
    log_height: usize,
) -> Vec<Target>
where
    F: Field + TwoAdicField,
    EF: ExtensionField<F>,
{
    builder.push_scope("fri_precompute_two_adic_powers");

    let g = F::two_adic_generator(log_height);
    let result = iter::successors(Some(g), |&prev| Some(prev.square()))
        .take(log_height)
        .map(|p| builder.define_const(EF::from(p)))
        .collect();

    builder.pop_scope();
    result
}

/// Compute the final query point after all FRI folding rounds.
///
/// After consuming `total_bits_consumed` bits through all fold phases, the remaining
/// bits form the domain index for the final polynomial evaluation.
fn compute_final_query_point<F, EF>(
    builder: &mut CircuitBuilder<EF>,
    index_bits: &[Target],
    log_max_height: usize,
    total_bits_consumed: usize,
    powers_of_g: &[Target],
) -> Target
where
    F: Field + TwoAdicField,
    EF: ExtensionField<F>,
{
    builder.push_scope("compute_final_query_point");

    let domain_index_bits: Vec<Target> = index_bits[total_bits_consumed..log_max_height].to_vec();

    // Pad bits and reverse
    let mut reversed_bits = vec![builder.define_const(EF::ZERO); total_bits_consumed];
    reversed_bits.extend(domain_index_bits.iter().rev().copied());

    let one = builder.define_const(EF::ONE);
    let mut result = one;
    for (&bit, &power) in reversed_bits.iter().zip(powers_of_g.iter()) {
        let multiplier = builder.select(bit, power, one);
        result = builder.mul(result, multiplier);
    }

    builder.pop_scope();
    result
}

/// Precompute evaluation points for all unique heights.
///
/// Runs a single select-mul chain for the tallest height's reversed index bits and derives
/// smaller heights via `exp_power_of_2` on captured intermediates.
fn precompute_evaluation_points<F, EF>(
    builder: &mut CircuitBuilder<EF>,
    unique_heights_desc: &[usize],
    index_bits: &[Target],
    log_global_max_height: usize,
) -> BTreeMap<usize, Target>
where
    F: Field + TwoAdicField,
    EF: ExtensionField<F>,
{
    builder.push_scope("precompute_evaluation_points");

    debug_assert!(
        !unique_heights_desc.is_empty(),
        "must have at least one height"
    );
    debug_assert!(
        unique_heights_desc.windows(2).all(|w| w[0] > w[1]),
        "heights must be sorted in strictly descending order"
    );

    let h_max = unique_heights_desc[0];
    let bits_reduced = log_global_max_height - h_max;
    let rev_bits: Vec<Target> = index_bits[bits_reduced..bits_reduced + h_max]
        .iter()
        .rev()
        .copied()
        .collect();

    let g = F::two_adic_generator(h_max);
    let powers_of_g: Vec<_> = iter::successors(Some(g), |&prev| Some(prev.square()))
        .take(h_max)
        .map(|p| builder.define_const(EF::from(p)))
        .collect();

    let capture_set: BTreeMap<usize, ()> =
        unique_heights_desc[1..].iter().map(|&h| (h, ())).collect();

    let one = builder.define_const(EF::ONE);
    let generator = builder.alloc_const(EF::from(F::GENERATOR), "coset_generator");
    let mut g_pow = one;
    let mut result = BTreeMap::new();

    for i in 0..h_max {
        let multiplier = builder.select(rev_bits[i], powers_of_g[i], one);
        g_pow = builder.mul(g_pow, multiplier);

        let bits_done = i + 1;
        if capture_set.contains_key(&bits_done) {
            let derived = builder.exp_power_of_2(g_pow, h_max - bits_done);
            let x = builder.alloc_mul(generator, derived, "eval_point");
            result.insert(bits_done, x);
        }
    }

    let x_max = builder.alloc_mul(generator, g_pow, "eval_point");
    result.insert(h_max, x_max);

    builder.pop_scope(); // close `precompute_evaluation_points` scope
    result
}

/// Compute reduced opening for a single matrix in circuit form (EF-field).
///
/// Uses Horner's method to evaluate the polynomial in alpha without an explicit
/// alpha-power chain, saving one multiplication per column.
fn compute_single_reduced_opening<EF: Field>(
    builder: &mut CircuitBuilder<EF>,
    opened_values: &[Target], // Values at evaluation point x
    point_values: &[Target],  // Values at challenge point z
    alpha_pow: Target,        // Current alpha power (for this height)
    alpha: Target,            // Alpha challenge
    alpha_powers_set: &mut HashMap<usize, Target>,
    inv_z_minus_x: Target, // 1 / (z - x), shared across matrices at same (height, z)
) -> (Target, Target) // (new_alpha_pow, reduced_opening_contrib)
{
    builder.push_scope("compute_single_reduced_opening");

    let n = opened_values.len();

    if n == 0 {
        let zero = builder.define_const(EF::ZERO);
        builder.pop_scope();
        return (alpha_pow, zero);
    }

    // Horner's method with inline subtraction via HornerAcc:
    //   inner = 0
    //   inner = inner * alpha + p_at_z[n-1] - p_at_x[n-1]
    //   inner = inner * alpha + p_at_z[n-2] - p_at_x[n-2]
    //   ...
    //   inner = inner * alpha + p_at_z[0] - p_at_x[0]
    //
    // Each step emits a single HornerAcc ALU op (no intermediate witnesses).
    let zero = builder.define_const(EF::ZERO);
    let mut inner = zero;
    for i in (0..n).rev() {
        inner = builder.horner_acc_step(inner, alpha, point_values[i], opened_values[i]);
    }

    // reduced_opening = alpha_pow * inner * (1 / (z - x))
    let numerator = builder.mul(alpha_pow, inner);
    let reduced_opening = builder.mul(numerator, inv_z_minus_x);

    // Advance alpha_pow by alpha^n using square-and-multiply
    let alpha_n = if let Some(alpha_n) = alpha_powers_set.get(&n) {
        *alpha_n
    } else {
        let alpha_n = circuit_exp_by_constant(builder, alpha, n);
        alpha_powers_set.insert(n, alpha_n);
        alpha_n
    };
    let new_alpha_pow = builder.mul(alpha_pow, alpha_n);

    builder.pop_scope();
    (new_alpha_pow, reduced_opening)
}

/// Computes `base^n` in-circuit using square-and-multiply.
///
/// Cost: `floor(log2(n)) + popcount(n) - 1` multiplications.
fn circuit_exp_by_constant<EF: Field>(
    builder: &mut CircuitBuilder<EF>,
    base: Target,
    n: usize,
) -> Target {
    debug_assert!(n > 0);
    if n == 1 {
        return base;
    }
    let num_bits = usize::BITS - n.leading_zeros();
    // Start from the MSB (already implicit as `base`), process remaining bits top-down.
    let mut result = base;
    for i in (0..num_bits - 1).rev() {
        result = builder.mul(result, result);
        if (n >> i) & 1 == 1 {
            result = builder.mul(result, base);
        }
    }
    result
}

/// Compute reduced openings grouped **by height** with **per-height alpha powers**,
/// Returns a vector of (log_height, ro) sorted by descending height, plus the MMCS op IDs.
///
/// Reference (Plonky3): `p3_fri::verifier::open_input`
#[allow(clippy::type_complexity)]
fn open_input<F, EF, Comm>(
    builder: &mut CircuitBuilder<EF>,
    log_global_max_height: usize,
    index_bits: &[Target],
    alpha: Target,
    log_blowup: usize,
    commitments_with_opening_points: &ComsWithOpeningsTargets<Comm, TwoAdicMultiplicativeCoset<F>>,
    batch_opened_values: &[Vec<Vec<Target>>], // Per batch -> per matrix -> per column
    permutation_config: Option<Poseidon2Config>,
    pre_packed_input_caps: Option<&[Vec<Vec<Target>>]>,
) -> Result<(Vec<(usize, Target)>, Vec<NonPrimitiveOpId>), VerificationError>
where
    F: Field + TwoAdicField + PrimeField64,
    EF: ExtensionField<F>,
    Comm: ObservableCommitment,
{
    builder.push_scope("open_input");

    for &b in index_bits {
        builder.assert_bool(b);
    }
    debug_assert_eq!(
        index_bits.len(),
        log_global_max_height,
        "index_bits.len() must equal log_global_max_height"
    );

    // Collect unique heights across all matrices and precompute evaluation points.
    let unique_heights_desc: Vec<usize> = {
        let mut heights: Vec<usize> = commitments_with_opening_points
            .iter()
            .flat_map(|(_, mats)| {
                mats.iter()
                    .map(|(domain, _)| domain.log_size() + log_blowup)
            })
            .collect();
        heights.sort_unstable();
        heights.dedup();
        heights.reverse();
        heights
    };

    let eval_points = if unique_heights_desc.is_empty() {
        BTreeMap::new()
    } else {
        precompute_evaluation_points::<F, EF>(
            builder,
            &unique_heights_desc,
            index_bits,
            log_global_max_height,
        )
    };

    // height -> (alpha_pow_for_this_height, ro_sum_for_this_height)
    let mut reduced_openings = BTreeMap::<usize, (Target, Target)>::new();
    let mut mmcs_op_ids = Vec::new();

    // Process each batch
    for (batch_idx, ((batch_commit, mats), batch_openings)) in zip_eq(
        commitments_with_opening_points.iter(),
        batch_opened_values.iter(),
        VerificationError::InvalidProofShape(
            "Opened values and commitments count must match".to_string(),
        ),
    )?
    .enumerate()
    {
        // Recursive MMCS verification for this batch
        if let Some(perm_config) = permutation_config {
            // Use pre-packed cap if available, otherwise pack on the fly
            let commitment_cap: Vec<Vec<Target>> = if let Some(pre_packed) = pre_packed_input_caps {
                pre_packed[batch_idx].clone()
            } else {
                let lifted_commitment = batch_commit.to_observation_targets();
                commitment_cap_rows_from_lifted::<F, EF>(builder, perm_config, &lifted_commitment)
            };

            // Match native `p3_fri::verifier::open_input`: width is unused by MerkleTreeMmcs
            // verification (only height drives grouping); see Plonky3 TODO on Dimensions.width.
            let dimensions: Vec<Dimensions> = mats
                .iter()
                .map(|(domain, _)| Dimensions {
                    height: 1 << (domain.log_size() + log_blowup),
                    width: 0,
                })
                .collect();

            let op_ids = verify_batch_circuit::<F, EF>(
                builder,
                perm_config,
                &commitment_cap,
                &dimensions,
                index_bits,
                batch_openings,
            )
            .map_err(|e| {
                VerificationError::InvalidProofShape(format!(
                    "MMCS verification failed for batch {batch_idx}: {e:?}"
                ))
            })?;
            mmcs_op_ids.extend(op_ids);
        }

        let mut alpha_powers_set = HashMap::new();
        let mut inv_z_minus_x_cache: HashMap<(usize, Target), Target> = HashMap::new();

        // Group matrices in this batch by log_height. Within a height, when every matrix
        // exposes a single shared opening point z, we can run ONE big Horner chain across
        // all matrices instead of per-matrix chains. This eliminates K-1 alpha_pow advances
        // and K-1 mul_add boundary ops per group of K matrices, plus reduces wasted slots
        // in the K-step packed Horner schedule.
        type MatRef<'a> = (&'a [Target], &'a [(Target, Vec<Target>)]);
        let mut height_groups: BTreeMap<usize, Vec<MatRef<'_>>> = BTreeMap::new();

        for (mat_idx, ((mat_domain, mat_points_and_values), mat_opening)) in zip_eq(
            mats.iter(),
            batch_openings.iter(),
            VerificationError::InvalidProofShape(format!(
                "batch {batch_idx}: opened_values and point_values count must match"
            )),
        )?
        .enumerate()
        {
            for (_, ps_at_z) in mat_points_and_values {
                if mat_opening.len() != ps_at_z.len() {
                    return Err(VerificationError::InvalidProofShape(format!(
                        "batch {batch_idx} mat {mat_idx}: opened_values columns must match point_values columns"
                    )));
                }
            }
            let log_height = mat_domain.log_size() + log_blowup;
            height_groups
                .entry(log_height)
                .or_default()
                .push((mat_opening.as_slice(), mat_points_and_values.as_slice()));
        }

        for (log_height, matrices) in &height_groups {
            let x = eval_points[log_height];

            // Fast-path detection: all matrices in this height group expose exactly one
            // (z, ps_at_z) pair AND they all share the same z. This is the common case
            // (zeta-only opening per matrix, with a single shared challenge) and the only
            // case where one big Horner chain reproduces the original semantics.
            let unified_z: Option<Target> = if matrices.iter().all(|(_, pv)| pv.len() == 1) {
                let z0 = matrices[0].1[0].0;
                if matrices.iter().all(|(_, pv)| pv[0].0 == z0) {
                    Some(z0)
                } else {
                    None
                }
            } else {
                None
            };

            if let Some(z) = unified_z {
                // Resolve inv_z_minus_x (cached across batches at same (height, z)).
                let inv_z_minus_x =
                    *inv_z_minus_x_cache
                        .entry((*log_height, z))
                        .or_insert_with(|| {
                            let z_minus_x = builder.sub(z, x);
                            let one = builder.define_const(EF::ONE);
                            builder.div(one, z_minus_x)
                        });

                // Run one big reverse-Horner chain across all matrices in this group.
                //
                // Originally, matrix m contributes `alpha_pow_at_m * inv_z * inner_m` with
                // `alpha_pow_at_m = alpha_pow_initial * alpha^(prefix_<m)`. The total is
                //     sum_m alpha^(prefix_<m) * inner_m
                // which equals reverse-Horner over the concatenation
                //     [diff_M, diff_{M-1}, ..., diff_1] (each diff_m in column-reverse order)
                // because reverse-Horner of `[c_0, c_1, ..., c_{N-1}]` (last element processed
                // first) yields `c_0 + alpha*c_1 + ... + alpha^{N-1}*c_{N-1}`.
                let zero = builder.define_const(EF::ZERO);
                let mut inner = zero;
                let mut total_n = 0usize;
                for (mat_opening, points_and_values) in matrices.iter().rev() {
                    let ps_at_z = &points_and_values[0].1;
                    for i in (0..mat_opening.len()).rev() {
                        inner = builder.horner_acc_step(inner, alpha, ps_at_z[i], mat_opening[i]);
                    }
                    total_n += mat_opening.len();
                }

                // alpha^total_n via cached square-and-multiply.
                let alpha_total_n = match alpha_powers_set.get(&total_n) {
                    Some(&v) => v,
                    None => {
                        let v = circuit_exp_by_constant(builder, alpha, total_n);
                        alpha_powers_set.insert(total_n, v);
                        v
                    }
                };

                let (alpha_pow_h, ro_h) =
                    reduced_openings.entry(*log_height).or_insert_with(|| {
                        (
                            builder.define_const(EF::ONE),
                            builder.define_const(EF::ZERO),
                        )
                    });
                let alpha_pow_old = *alpha_pow_h;
                // ro_h += (alpha_pow * inv_z_minus_x) * inner
                let c = builder.mul(alpha_pow_old, inv_z_minus_x);
                *ro_h = builder.mul_add(c, inner, *ro_h);
                // alpha_pow *= alpha^total_n
                *alpha_pow_h = builder.mul(alpha_pow_old, alpha_total_n);
            } else {
                // Fallback: per-matrix per-z, identical to the original implementation.
                for (mat_opening, points_and_values) in matrices {
                    for (z, ps_at_z) in points_and_values.iter() {
                        let inv_z_minus_x = *inv_z_minus_x_cache
                            .entry((*log_height, *z))
                            .or_insert_with(|| {
                                let z_minus_x = builder.sub(*z, x);
                                let one = builder.define_const(EF::ONE);
                                builder.div(one, z_minus_x)
                            });

                        let alpha_pow_value = {
                            let (alpha_pow_h, _ro_h) =
                                reduced_openings.entry(*log_height).or_insert_with(|| {
                                    (
                                        builder.define_const(EF::ONE),
                                        builder.define_const(EF::ZERO),
                                    )
                                });
                            *alpha_pow_h
                        };

                        let (new_alpha_pow_h, ro_contrib) = compute_single_reduced_opening(
                            builder,
                            mat_opening,
                            ps_at_z,
                            alpha_pow_value,
                            alpha,
                            &mut alpha_powers_set,
                            inv_z_minus_x,
                        );

                        let entry = reduced_openings.get_mut(log_height).expect("entry");
                        entry.1 = builder.add(entry.1, ro_contrib);
                        entry.0 = new_alpha_pow_h;
                    }
                }
            }
        }

        // `reduced_openings` would have a log_height = log_blowup entry only if there was a
        // trace matrix of height 1. In this case `f` is constant, so `(f(zeta) - f(x))/(zeta - x)`
        // must equal `0`.
        if let Some((_ap, ro0)) = reduced_openings.get(&log_blowup) {
            let zero = builder.define_const(EF::ZERO);
            builder.connect(*ro0, zero);
        }
    }

    builder.pop_scope(); // close `open_input` scope

    // Into descending (height, ro) list
    let reduced_list: Vec<_> = reduced_openings
        .into_iter()
        .rev()
        .map(|(h, (_ap, ro))| (h, ro))
        .collect();
    Ok((reduced_list, mmcs_op_ids))
}

/// Verify FRI arithmetic in-circuit with optional MMCS verification.
///
/// Supports variable-arity FRI folding: each phase may fold by a different arity
/// determined by `log_arities` extracted from the proof.
///
/// When `permutation_config` is `Some`, this function performs full recursive MMCS
/// verification for both input batch openings and commit-phase openings.
/// When `None`, only arithmetic verification is performed (for testing).
///
/// Returns the list of non-primitive operation IDs that require private data
/// (Merkle sibling values) to be set by the runner.
///
/// Reference (Plonky3): `p3_fri::verifier::verify_fri`
pub fn verify_fri_circuit<F, EF, RecMmcs, Inner, Witness, Comm>(
    builder: &mut CircuitBuilder<EF>,
    fri_proof_targets: &FriProofTargets<F, EF, RecMmcs, InputProofTargets<F, EF, Inner>, Witness>,
    alpha: Target,
    betas: &[Target],
    index_bits_per_query: &[Vec<Target>],
    commitments_with_opening_points: &ComsWithOpeningsTargets<Comm, TwoAdicMultiplicativeCoset<F>>,
    log_blowup: usize,
    permutation_config: Option<Poseidon2Config>,
) -> Result<Vec<NonPrimitiveOpId>, VerificationError>
where
    F: Field + TwoAdicField + PrimeField64,
    EF: ExtensionField<F>,
    RecMmcs: RecursiveExtensionMmcs<F, EF>,
    RecMmcs::Commitment: ObservableCommitment,
    Inner: RecursiveMmcs<F, EF>,
    Witness: Recursive<EF>,
    Comm: ObservableCommitment,
{
    builder.push_scope("verify_fri");

    let num_phases = betas.len();
    let num_queries = fri_proof_targets.query_proofs.len();
    let log_arities = &fri_proof_targets.log_arities;

    let total_log_reduction: usize = log_arities.iter().sum();

    tracing::debug!(
        "verify_fri_circuit: num_phases={}, num_queries={}, log_blowup={}, log_arities={:?}",
        num_phases,
        num_queries,
        log_blowup,
        log_arities,
    );

    // Validate shape.
    if num_phases != fri_proof_targets.commit_phase_commits.len() {
        return Err(VerificationError::InvalidProofShape(format!(
            "betas length must equal number of commit-phase commitments: expected {}, got {}",
            num_phases,
            fri_proof_targets.commit_phase_commits.len()
        )));
    }

    if num_phases != fri_proof_targets.commit_pow_witnesses.len() {
        return Err(VerificationError::InvalidProofShape(format!(
            "Number of commit-phase commitments must equal number of commit-phase pow witnesses: expected {}, got {}",
            num_phases,
            fri_proof_targets.commit_pow_witnesses.len()
        )));
    }

    if log_arities.len() != num_phases {
        return Err(VerificationError::InvalidProofShape(format!(
            "log_arities length must equal number of phases: expected {}, got {}",
            num_phases,
            log_arities.len()
        )));
    }

    if num_queries != index_bits_per_query.len() {
        return Err(VerificationError::InvalidProofShape(format!(
            "index_bits_per_query length must equal number of query proofs: expected {}, got {}",
            num_queries,
            index_bits_per_query.len()
        )));
    }

    let log_max_height = index_bits_per_query[0].len();
    if index_bits_per_query
        .iter()
        .any(|v| v.len() != log_max_height)
    {
        return Err(VerificationError::InvalidProofShape(
            "all index_bits_per_query entries must have same length".to_string(),
        ));
    }

    if betas.is_empty() {
        return Err(VerificationError::InvalidProofShape(
            "FRI must have at least one fold phase".to_string(),
        ));
    }

    // With variable arity: log_max_height = total_log_reduction + log_final_poly_len + log_blowup
    let log_final_poly_len = log_max_height
        .checked_sub(total_log_reduction)
        .and_then(|x| x.checked_sub(log_blowup))
        .ok_or_else(|| {
            VerificationError::InvalidProofShape(
                "Invalid FRI parameters: log_max_height too small for given log_arities"
                    .to_string(),
            )
        })?;

    let expected_final_poly_len = 1 << log_final_poly_len;
    let actual_final_poly_len = fri_proof_targets.final_poly.len();

    if actual_final_poly_len != expected_final_poly_len {
        return Err(VerificationError::InvalidProofShape(format!(
            "Final polynomial length mismatch: expected 2^{log_final_poly_len} = {expected_final_poly_len}, got {actual_final_poly_len}"
        )));
    }

    // Precompute cumulative bits consumed after each phase.
    // cumulative_bits[i] = sum(log_arities[0..i])
    let mut cumulative_bits = Vec::with_capacity(num_phases + 1);
    cumulative_bits.push(0usize);
    for &la in log_arities {
        cumulative_bits.push(cumulative_bits.last().unwrap() + la);
    }

    // Precompute the folded height after each phase for roll-in mapping.
    // folded_height_after[i] = log_max_height - cumulative_bits[i+1]
    let folded_height_after: Vec<usize> = (0..num_phases)
        .map(|i| log_max_height - cumulative_bits[i + 1])
        .collect();

    // Precompute shared beta powers and generator powers used across queries.
    let beta_pows_per_phase = precompute_beta_powers_per_phase(builder, betas, log_arities);
    let powers_of_g_final = precompute_two_adic_powers::<F, EF>(builder, log_max_height);

    // Pre-pack commitment caps once so they can be reused across all queries.
    // Each input batch commitment and each commit-phase commitment is packed from
    // lifted representation to extension representation a single time.
    let pre_packed_input_caps: Option<Vec<Vec<Vec<Target>>>> =
        permutation_config.map(|perm_config| {
            commitments_with_opening_points
                .iter()
                .map(|(commit, _)| {
                    let lifted = commit.to_observation_targets();
                    commitment_cap_rows_from_lifted::<F, EF>(builder, perm_config, &lifted)
                })
                .collect()
        });

    let pre_packed_commit_caps: Option<Vec<Vec<Vec<Target>>>> =
        permutation_config.map(|perm_config| {
            fri_proof_targets
                .commit_phase_commits
                .iter()
                .map(|commit| {
                    let lifted = commit.to_observation_targets();
                    commitment_cap_rows_from_lifted::<F, EF>(builder, perm_config, &lifted)
                })
                .collect()
        });

    // Collect all MMCS operation IDs for private data setting
    let mut all_mmcs_op_ids = Vec::new();

    // For each query, extract opened values from proof and compute reduced openings and fold.
    for (q, query_proof) in fri_proof_targets.query_proofs.iter().enumerate() {
        builder.push_scope("verify_fri_query");
        let batch_opened_values: Vec<Vec<Vec<Target>>> = query_proof
            .input_proof
            .iter()
            .map(|batch| batch.opened_values.clone())
            .collect();

        // Arithmetic `open_input` to get (height, ro) descending, plus MMCS op IDs
        let (reduced_by_height, input_mmcs_ops) = open_input::<F, EF, Comm>(
            builder,
            log_max_height,
            &index_bits_per_query[q],
            alpha,
            log_blowup,
            commitments_with_opening_points,
            &batch_opened_values,
            permutation_config,
            pre_packed_input_caps.as_deref(),
        )?;
        all_mmcs_op_ids.extend(input_mmcs_ops);

        if reduced_by_height.is_empty() {
            return Err(VerificationError::InvalidProofShape(
                "No reduced openings; did you commit to zero polynomials?".to_string(),
            ));
        }
        if reduced_by_height[0].0 != log_max_height {
            return Err(VerificationError::InvalidProofShape(format!(
                "First reduced opening must be at max height {}, got {}",
                log_max_height, reduced_by_height[0].0
            )));
        }
        let initial_folded_eval = reduced_by_height[0].1;

        // Pack sibling values for each phase (variable count per phase).
        let sibling_values_per_phase: Vec<Vec<Target>> = query_proof
            .commit_phase_openings
            .iter()
            .map(|opening| opening.sibling_values_packed(builder))
            .collect();

        if sibling_values_per_phase.len() != num_phases {
            return Err(VerificationError::InvalidProofShape(format!(
                "commit_phase_openings count must match phases: expected {}, got {}",
                num_phases,
                sibling_values_per_phase.len()
            )));
        }

        // Build height-aligned roll-ins using variable-arity cumulative bits.
        let mut roll_ins: Vec<Option<Target>> = vec![None; num_phases];
        for &(h, ro) in reduced_by_height.iter().skip(1) {
            // Find the phase whose folded height matches h
            let phase_idx = folded_height_after.iter().position(|&fh| fh == h);
            if let Some(i) = phase_idx {
                if roll_ins[i].is_some() {
                    return Err(VerificationError::InvalidProofShape(format!(
                        "duplicate roll-in for phase {i} (height {h})",
                    )));
                }
                roll_ins[i] = Some(ro);
            } else {
                let zero = builder.define_const(EF::ZERO);
                builder.connect(ro, zero);
            }
        }

        // Compute the final query point using total bits consumed
        builder.push_scope("compute_final_query_point");
        let final_query_point = compute_final_query_point::<F, EF>(
            builder,
            &index_bits_per_query[q],
            log_max_height,
            total_log_reduction,
            &powers_of_g_final,
        );
        builder.pop_scope();

        let final_poly_eval =
            evaluate_polynomial(builder, &fri_proof_targets.final_poly, final_query_point);

        // Commit-phase MMCS verification with variable arity.
        // When MMCS verification is active, the fold chain is computed as part of
        // the MMCS loop (each phase calls fold_one_phase), so the final
        // current_folded is connected directly to final_poly_eval — no separate
        // fold_chain_circuit call is needed.
        // When MMCS verification is not active (no Poseidon2 table), we fall back
        // to fold_chain_circuit for the arithmetic fold constraint.
        if let Some(perm_config) = permutation_config {
            let subgroup_starts = precompute_subgroup_starts::<F, EF>(
                builder,
                &index_bits_per_query[q],
                log_max_height,
                log_arities,
                &cumulative_bits,
            );

            let mut current_folded = initial_folded_eval;
            let mut bits_consumed = 0usize;
            let mut log_current_height = log_max_height;

            for (phase_idx, (commit, _opening)) in fri_proof_targets
                .commit_phase_commits
                .iter()
                .zip(query_proof.commit_phase_openings.iter())
                .enumerate()
            {
                let log_arity = log_arities[phase_idx];
                let arity = 1usize << log_arity;
                let log_folded_height = log_current_height - log_arity;
                let siblings = &sibling_values_per_phase[phase_idx];

                // Skip MMCS verification for height 0 (no Merkle tree)
                if log_folded_height == 0 {
                    current_folded = fold_one_phase::<F, EF>(
                        builder,
                        current_folded,
                        siblings,
                        betas[phase_idx],
                        &index_bits_per_query[q],
                        bits_consumed,
                        log_arity,
                        roll_ins[phase_idx],
                        Some(beta_pows_per_phase[phase_idx]),
                        None,
                        subgroup_starts[phase_idx],
                    );
                    bits_consumed += log_arity;
                    log_current_height = log_folded_height;
                    continue;
                }

                builder.push_scope("fri_commit_phase_mmcs");

                let index_in_group_bits =
                    &index_bits_per_query[q][bits_consumed..bits_consumed + log_arity];
                let evals =
                    reconstruct_evals(builder, current_folded, siblings, index_in_group_bits);

                // Use pre-packed commit-phase cap
                let commitment_cap: Vec<Vec<Target>> =
                    if let Some(ref pre_packed) = pre_packed_commit_caps {
                        pre_packed[phase_idx].clone()
                    } else {
                        let lifted_commitment = commit.to_observation_targets();
                        commitment_cap_rows_from_lifted::<F, EF>(
                            builder,
                            perm_config,
                            &lifted_commitment,
                        )
                    };

                // Dimensions: width = arity, height = 2^log_folded_height
                let folded_height = 1usize << log_folded_height;
                let dimensions = vec![Dimensions {
                    height: folded_height,
                    width: arity,
                }];

                // Parent index bits start after index_in_group bits
                let parent_bit_start = bits_consumed + log_arity;
                let parent_bit_end = (parent_bit_start + log_folded_height).min(log_max_height);
                let zero = builder.define_const(EF::ZERO);

                let mut parent_index_bits: Vec<Target> =
                    index_bits_per_query[q][parent_bit_start..parent_bit_end].to_vec();
                while parent_index_bits.len() < log_folded_height {
                    parent_index_bits.push(zero);
                }

                let commit_phase_ops = verify_batch_circuit_from_extension_opened::<F, EF>(
                    builder,
                    perm_config,
                    &commitment_cap,
                    &dimensions,
                    &parent_index_bits,
                    core::slice::from_ref(&evals),
                )
                .map_err(|e| {
                    VerificationError::InvalidProofShape(format!(
                        "Commit-phase MMCS verification failed for query {q}, phase {phase_idx}: {e:?}"
                    ))
                })?;
                all_mmcs_op_ids.extend(commit_phase_ops);

                // Fold reusing the pre-built evals and subgroup_start
                current_folded = fold_one_phase::<F, EF>(
                    builder,
                    current_folded,
                    siblings,
                    betas[phase_idx],
                    &index_bits_per_query[q],
                    bits_consumed,
                    log_arity,
                    roll_ins[phase_idx],
                    Some(beta_pows_per_phase[phase_idx]),
                    Some(&evals),
                    subgroup_starts[phase_idx],
                );

                bits_consumed += log_arity;
                log_current_height = log_folded_height;

                builder.pop_scope(); // close fri_commit_phase_mmcs
            }

            // The MMCS loop already computed the full fold chain; connect directly.
            builder.connect(current_folded, final_poly_eval);
        } else {
            // No MMCS verification — use fold_chain_circuit for the arithmetic constraint.
            let mut fold_phases = Vec::with_capacity(num_phases);
            for i in 0..num_phases {
                fold_phases.push(FoldPhaseConfig {
                    beta: betas[i],
                    siblings: sibling_values_per_phase[i].clone(),
                    roll_in: roll_ins[i],
                });
            }

            builder.push_scope("fri_fold_chain_no_mmcs query");
            let folded_eval = fold_chain_circuit::<F, EF>(
                builder,
                initial_folded_eval,
                &index_bits_per_query[q],
                &fold_phases,
                log_arities,
                &cumulative_bits,
                &beta_pows_per_phase,
            );
            builder.pop_scope();
            builder.connect(folded_eval, final_poly_eval);
        }

        builder.pop_scope(); // close verify_fri_query
    }

    builder.pop_scope();

    Ok(all_mmcs_op_ids)
}
