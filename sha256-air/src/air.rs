//! Constraints for the SHA-256 compression AIR.
//!
//! # Overview
//!
//! One row encodes one compression. Five constraint blocks are emitted per row:
//!
//! - Bit range checks on every unpacked column.
//! - Bridges between the packed input state and the chain bit decompositions.
//! - Message schedule for `W[16..64]`.
//! - 64 compression rounds.
//! - Finalization: add input state to working variables.
//!
//! # Degree budget
//!
//! Every emitted constraint has degree at most 3.
//!
//! - Boolean checks: degree 2.
//! - Packing identities: degree 1.
//! - `Ch` per bit: degree 2.
//! - `Maj` per bit: degree 3.
//! - Sigma XOR3 per bit: degree 3.
//! - Addition helpers: degree 3.

use alloc::vec::Vec;
use core::array;
use core::borrow::Borrow;

use p3_air::utils::{add2, add3, pack_bits_le};
use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::{Dup, PrimeCharacteristicRing, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use crate::columns::{CHAIN_LEN, NUM_SHA256_COLS, Sha256Cols};
use crate::constants::{
    BITS_PER_LIMB, BLOCK_WORDS, SCHEDULE_EXTENSIONS, SHA256_K, STATE_WORDS, U32_LIMBS, WORD_BITS,
};
use crate::generation::generate_trace_rows;

/// AIR that enforces one SHA-256 compression per row.
///
/// Unit struct: all numerical values live in the constants module, and the
/// constraint logic is driven entirely by const generics.
#[derive(Debug, Default, Clone, Copy)]
pub struct Sha256Air;

impl Sha256Air {
    /// Build a random trace containing `num_hashes` compressions.
    ///
    /// # Arguments
    ///
    /// - `num_hashes`: number of rows to produce. Must be a power of two.
    /// - `extra_capacity_bits`: log2 blowup factor reserved for the low-degree extension.
    ///
    /// # Returns
    ///
    /// A row-major trace matrix consumable by the Plonky3 uni-stark prover.
    ///
    /// # Panics
    ///
    /// Panics if `num_hashes` is not a power of two.
    pub fn generate_trace_rows<F: PrimeField64>(
        &self,
        num_hashes: usize,
        extra_capacity_bits: usize,
    ) -> RowMajorMatrix<F> {
        // Deterministic RNG so two invocations with the same arguments yield identical traces
        //
        // Useful for reproducible benchmarks and fuzzing.
        let mut rng = SmallRng::seed_from_u64(1);
        // Every input packs a 16-word block followed by an 8-word chaining state - 24 u32's per row.
        let inputs: Vec<[u32; BLOCK_WORDS + STATE_WORDS]> =
            (0..num_hashes).map(|_| rng.random()).collect();
        // Delegate to the shared generator so test vectors and random batches hit the same code path.
        generate_trace_rows(inputs, extra_capacity_bits)
    }
}

impl<F> BaseAir<F> for Sha256Air {
    fn width(&self) -> usize {
        // Column count is pinned by the `repr(C)` row struct.
        NUM_SHA256_COLS
    }

    fn main_next_row_columns(&self) -> Vec<usize> {
        // Each row is self-contained, so no next-row columns are needed.
        Vec::new()
    }
}

impl<AB: AirBuilder> Air<AB> for Sha256Air {
    #[inline]
    fn eval(&self, builder: &mut AB) {
        // Read the current row as a typed column struct.
        let main = builder.main();
        let local: &Sha256Cols<AB::Var> = main.current_slice().borrow();

        // Phase 1: ensure every unpacked column actually holds a boolean.
        eval_bit_range_checks::<AB>(builder, local);
        // Phase 2: tie the packed input state to the first four chain entries.
        eval_initial_state::<AB>(builder, local);
        // Phase 3: enforce the 48 message-schedule recurrence equations.
        eval_message_schedule::<AB>(builder, local);
        // Phase 4: enforce the 64 compression rounds.
        eval_compression::<AB>(builder, local);
        // Phase 5: enforce the output equals input + final working variables.
        eval_finalization::<AB>(builder, local);
    }
}

/// Emit `b * (b - 1) = 0` for every bit-valued column in the row.
///
/// Required because downstream XOR and AND identities only produce correct
/// results when their inputs are strictly `{0, 1}`.
fn eval_bit_range_checks<AB: AirBuilder>(builder: &mut AB, local: &Sha256Cols<AB::Var>) {
    // Range-check every message-schedule word, including the 16 block words
    // supplied as pure witness and the 48 expanded words.
    for word in &local.w {
        builder.assert_bools(*word);
    }

    // Range-check the `a` chain.
    // - Indices 0..4 cover the input state bits;
    // - Indices 4..68 cover every `new_a` produced by the compression loop.
    for word in &local.a_chain {
        builder.assert_bools(*word);
    }
    // Symmetric range check for the `e` chain.
    for word in &local.e_chain {
        builder.assert_bools(*word);
    }
}

/// Assert that each packed `H[i]` agrees with the matching chain entry.
///
/// Layout mirrored from the columns module:
///
/// ```text
///     H_0 <-> a_chain[3]   H_4 <-> e_chain[3]
///     H_1 <-> a_chain[2]   H_5 <-> e_chain[2]
///     H_2 <-> a_chain[1]   H_6 <-> e_chain[1]
///     H_3 <-> a_chain[0]   H_7 <-> e_chain[0]
/// ```
fn eval_initial_state<AB: AirBuilder>(builder: &mut AB, local: &Sha256Cols<AB::Var>) {
    // Words H_0..H_3 live in the reversed prefix of the `a` chain.
    for i in 0..4 {
        // The chain stores (d, c, b, a) at indices (0, 1, 2, 3), so invert.
        let chain_idx = 3 - i;
        let bits = &local.a_chain[chain_idx];
        assert_packed_equals_bits::<AB>(builder, &local.h_in[i], bits);
    }
    // Words H_4..H_7 live in the reversed prefix of the `e` chain.
    for i in 0..4 {
        let chain_idx = 3 - i;
        let bits = &local.e_chain[chain_idx];
        assert_packed_equals_bits::<AB>(builder, &local.h_in[4 + i], bits);
    }
}

/// Enforce the 48 new message-schedule words.
///
/// For every `t` in `[16, 64)`:
///
/// ```text
///     small_sigma0 = ROTR_7 (W[t - 15]) XOR ROTR_18(W[t - 15]) XOR SHR_3 (W[t - 15])
///     small_sigma1 = ROTR_17(W[t - 2 ]) XOR ROTR_19(W[t - 2 ]) XOR SHR_10(W[t - 2 ])
///     tmp          = small_sigma1 + W[t - 7]                   (mod 2^32)
///     W[t]         = tmp + small_sigma0 + W[t - 16]            (mod 2^32)
/// ```
///
/// The four-term sum is split into two three-term sums because the shared
/// add helpers saturate at three addends.
fn eval_message_schedule<AB: AirBuilder>(builder: &mut AB, local: &Sha256Cols<AB::Var>) {
    // Loop over the 48 expanded schedule positions. `i` indexes the packed
    // auxiliary columns; `t` indexes into the unpacked `w` array.
    for i in 0..SCHEDULE_EXTENSIONS {
        let t = i + BLOCK_WORDS;

        // Bind small_sigma0 to the packed column via per-bit XOR-3 expansion.
        assert_sigma_matches::<AB>(
            builder,
            &local.w[t - 15],
            SigmaSpec::SmallSigma0,
            &local.sched_sigma0[i],
        );

        // Bind small_sigma1 to the packed column via per-bit XOR-3 expansion.
        assert_sigma_matches::<AB>(
            builder,
            &local.w[t - 2],
            SigmaSpec::SmallSigma1,
            &local.sched_sigma1[i],
        );

        // Three-term add #1: tmp = small_sigma1 + W[t - 7].
        //
        // W[t - 7] is consumed as an expression packed on the fly from its
        // bit decomposition - no extra column is needed for it.
        let w_tm7_packed_expr = pack_word::<AB>(&local.w[t - 7]);
        add2(
            builder,
            &local.sched_tmp[i],
            &local.sched_sigma1[i],
            &w_tm7_packed_expr,
        );

        // Three-term add #2: w_packed = tmp + small_sigma0 + W[t - 16].
        let sched_sigma0_expr: [AB::Expr; U32_LIMBS] = local.sched_sigma0[i].map(Into::into);
        let w_tm16_packed_expr = pack_word::<AB>(&local.w[t - 16]);
        add3(
            builder,
            &local.w_packed[i],
            &local.sched_tmp[i],
            &sched_sigma0_expr,
            &w_tm16_packed_expr,
        );

        // Bridge: w_packed must agree with the bit decomposition the next
        // round will consume.
        assert_packed_equals_bits::<AB>(builder, &local.w_packed[i], &local.w[t]);
    }
}

/// Enforce one compression round per iteration.
///
/// Each round reads its eight working variables from the two chains and
/// evaluates:
///
/// ```text
///     big_sigma1 = ROTR_6 (e) XOR ROTR_11(e) XOR ROTR_25(e)
///     ch         = (e AND f) XOR (NOT e AND g)
///     tmp1       = h + big_sigma1 + ch               (mod 2^32)
///     T1         = tmp1 + K[t] + W[t]                (mod 2^32)
///     big_sigma0 = ROTR_2 (a) XOR ROTR_13(a) XOR ROTR_22(a)
///     maj        = (a AND b) XOR (a AND c) XOR (b AND c)
///     T2         = big_sigma0 + maj                  (mod 2^32)
///     new_a      = T1 + T2                           (mod 2^32)
///     new_e      = d  + T1                           (mod 2^32)
/// ```
fn eval_compression<AB: AirBuilder>(builder: &mut AB, local: &Sha256Cols<AB::Var>) {
    for (t, round) in local.rounds.iter().enumerate() {
        // Read the eight working variables for round `t` from the chains.
        //
        //     chain[t + 3]  -->  slot a (or e)
        //     chain[t + 2]  -->  slot b (or f)
        //     chain[t + 1]  -->  slot c (or g)
        //     chain[t + 0]  -->  slot d (or h)
        let a_bits = &local.a_chain[t + 3];
        let b_bits = &local.a_chain[t + 2];
        let c_bits = &local.a_chain[t + 1];
        let d_bits = &local.a_chain[t];
        let e_bits = &local.e_chain[t + 3];
        let f_bits = &local.e_chain[t + 2];
        let g_bits = &local.e_chain[t + 1];
        let h_bits = &local.e_chain[t];

        // big_sigma1 check: reduces to a per-bit XOR3 of rotated `e` bits.
        assert_sigma_matches::<AB>(builder, e_bits, SigmaSpec::BigSigma1, &round.sigma1_e);

        // Ch check: disjoint AND-terms collapse the XOR to an addition.
        //
        //     Ch_i = e_i * f_i + (1 - e_i) * g_i
        assert_ch_matches::<AB>(builder, e_bits, f_bits, g_bits, &round.ch);

        // Three-term add: tmp1 = big_sigma1 + ch + h.
        let ch_expr: [AB::Expr; U32_LIMBS] = round.ch.map(Into::into);
        let h_packed_expr = pack_word::<AB>(h_bits);
        add3(
            builder,
            &round.tmp1,
            &round.sigma1_e,
            &ch_expr,
            &h_packed_expr,
        );

        // Inject K[t] as a constant expression, one 16-bit limb at a time.
        let k_expr: [AB::Expr; U32_LIMBS] = [
            // Low 16 bits of K[t].
            AB::Expr::from_u32(SHA256_K[t] & 0xFFFF),
            // High 16 bits of K[t].
            AB::Expr::from_u32(SHA256_K[t] >> BITS_PER_LIMB),
        ];
        // Three-term add: T1 = tmp1 + K[t] + W[t].
        let w_packed_expr = pack_word::<AB>(&local.w[t]);
        add3(builder, &round.t1, &round.tmp1, &k_expr, &w_packed_expr);

        // big_sigma0 check: per-bit XOR3 of rotated `a` bits.
        assert_sigma_matches::<AB>(builder, a_bits, SigmaSpec::BigSigma0, &round.sigma0_a);

        // Maj check: degree-3 identity `a * b + c * (a XOR b)`.
        assert_maj_matches::<AB>(builder, a_bits, b_bits, c_bits, &round.maj);

        // Two-term add: T2 = big_sigma0 + maj.
        let maj_expr: [AB::Expr; U32_LIMBS] = round.maj.map(Into::into);
        add2(builder, &round.t2, &round.sigma0_a, &maj_expr);

        // Two-term add: new_a = T1 + T2.
        let t2_expr: [AB::Expr; U32_LIMBS] = round.t2.map(Into::into);
        add2(builder, &round.new_a_packed, &round.t1, &t2_expr);

        // Two-term add: new_e = d + T1.
        let d_packed_expr = pack_word::<AB>(d_bits);
        add2(builder, &round.new_e_packed, &round.t1, &d_packed_expr);

        // Bridge: packed new_a must match the next chain slot bit-by-bit, so
        // round t+1 reads the exact value we just produced.
        assert_packed_equals_bits::<AB>(builder, &round.new_a_packed, &local.a_chain[t + 4]);
        // Same bridge for new_e.
        assert_packed_equals_bits::<AB>(builder, &round.new_e_packed, &local.e_chain[t + 4]);
    }
}

/// Enforce `H_out[i] = H_in[i] + final_state[i] (mod 2^32)`.
///
/// The final working variables sit at the tail of the two chains:
///
/// ```text
///     final a = a_chain[67]    final e = e_chain[67]
///     final b = a_chain[66]    final f = e_chain[66]
///     final c = a_chain[65]    final g = e_chain[65]
///     final d = a_chain[64]    final h = e_chain[64]
/// ```
fn eval_finalization<AB: AirBuilder>(builder: &mut AB, local: &Sha256Cols<AB::Var>) {
    // H'_0..H'_3 are obtained by adding H[i] and the tail of the `a` chain.
    for i in 0..4 {
        let final_bits = &local.a_chain[CHAIN_LEN - 1 - i];
        let packed_expr = pack_word::<AB>(final_bits);
        add2(builder, &local.h_out[i], &local.h_in[i], &packed_expr);
    }
    // H'_4..H'_7 are obtained by adding H[i] and the tail of the `e` chain.
    for i in 0..4 {
        let final_bits = &local.e_chain[CHAIN_LEN - 1 - i];
        let packed_expr = pack_word::<AB>(final_bits);
        add2(
            builder,
            &local.h_out[4 + i],
            &local.h_in[4 + i],
            &packed_expr,
        );
    }
}

/// Pack a 32-bit word of boolean-valued columns into a 2-limb expression.
///
/// # Arguments
///
/// - `bits`: 32 boolean columns in little-endian order.
///
/// # Returns
///
/// A two-element array `[lo_expr, hi_expr]` where:
/// - `lo_expr` equals `sum_{i=0}^{15} 2^i * bits[i]`.
/// - `hi_expr` equals `sum_{i=0}^{15} 2^i * bits[16 + i]`.
#[inline]
fn pack_word<AB: AirBuilder>(bits: &[AB::Var; WORD_BITS]) -> [AB::Expr; U32_LIMBS] {
    [
        // Low limb: bits 0..16.
        pack_bits_le::<AB::Expr, _, _>(bits[..BITS_PER_LIMB].iter().copied()),
        // High limb: bits 16..32.
        pack_bits_le::<AB::Expr, _, _>(bits[BITS_PER_LIMB..].iter().copied()),
    ]
}

/// Assert `packed` equals the packing of `bits`.
///
/// Bridges the packed and unpacked views of a single 32-bit word.
#[inline]
fn assert_packed_equals_bits<AB: AirBuilder>(
    builder: &mut AB,
    packed: &[AB::Var; U32_LIMBS],
    bits: &[AB::Var; WORD_BITS],
) {
    // Re-derive the packed expression from the bits.
    let [lo, hi] = pack_word::<AB>(bits);
    // Emit one equality per limb.
    builder.assert_zeros([packed[0].into() - lo, packed[1].into() - hi]);
}

/// Selector for the four SHA-256 "sigma" combinators.
///
/// Each variant fixes three integer amounts and whether the third operand is
/// a rotation or a logical shift.
///
/// ```text
///     BigSigma0:   ROTR_2   XOR ROTR_13  XOR ROTR_22
///     BigSigma1:   ROTR_6   XOR ROTR_11  XOR ROTR_25
///     SmallSigma0: ROTR_7   XOR ROTR_18  XOR SHR_3
///     SmallSigma1: ROTR_17  XOR ROTR_19  XOR SHR_10
/// ```
#[derive(Copy, Clone)]
enum SigmaSpec {
    BigSigma0,
    BigSigma1,
    SmallSigma0,
    SmallSigma1,
}

/// Return the rotation amounts and shift kind for a sigma variant.
///
/// # Returns
///
/// A tuple `(r1, r2, r3_or_s3, kind)`:
/// - `r1`, `r2`: rotation amounts for the first two XOR operands.
/// - `r3_or_s3`: rotation or shift amount for the third operand.
/// - `kind`: whether the third operand is a rotation or a logical shift.
#[inline]
const fn sigma_params(spec: SigmaSpec) -> (u32, u32, u32, ShiftKind) {
    match spec {
        SigmaSpec::BigSigma0 => (2, 13, 22, ShiftKind::Rotate),
        SigmaSpec::BigSigma1 => (6, 11, 25, ShiftKind::Rotate),
        SigmaSpec::SmallSigma0 => (7, 18, 3, ShiftKind::Logical),
        SigmaSpec::SmallSigma1 => (17, 19, 10, ShiftKind::Logical),
    }
}

/// Distinguishes cyclic-rotate-right from logical-shift-right.
///
/// - Rotate wraps high bits around to the low end.
/// - Logical injects zeros at the high end.
#[derive(Copy, Clone)]
enum ShiftKind {
    Rotate,
    Logical,
}

/// Assert `packed` equals `sigma_spec(bits)` in packed form.
///
/// # Algorithm
///
/// Per output bit `i` in `0..32`:
///
/// ```text
///     out_i = src_r1[i] XOR src_r2[i] XOR src_r3_or_s3[i]
/// ```
///
/// where `src_rk[i]` reads `bits[(i + rk) mod 32]` for a rotation, or
/// `bits[i + rk]` if that index is `< 32` and `0` otherwise for a logical
/// shift.
///
/// The 16 per-bit expressions in each limb are then combined via Horner into
/// a single polynomial and compared with the committed packed column.
fn assert_sigma_matches<AB: AirBuilder>(
    builder: &mut AB,
    bits: &[AB::Var; WORD_BITS],
    spec: SigmaSpec,
    packed: &[AB::Var; U32_LIMBS],
) {
    let (r1, r2, r3, kind) = sigma_params(spec);

    // Fetch the "third" operand bit with the correct shift semantics.
    let get_shifted_bit = |i: usize| -> AB::Expr {
        match kind {
            // Rotate: wrap the index around modulo 32.
            ShiftKind::Rotate => bits[(i + r3 as usize) % WORD_BITS].into(),
            // Logical shift right: inject zero once the source index runs
            // past the high end of the word.
            ShiftKind::Logical => {
                let src = i + r3 as usize;
                if src < WORD_BITS {
                    bits[src].into()
                } else {
                    AB::Expr::ZERO
                }
            }
        }
    };

    // Per-limb accumulators holding the packed value rebuilt from bits.
    let mut accumulators: [AB::Expr; U32_LIMBS] = [AB::Expr::ZERO; U32_LIMBS];
    for (limb, acc_slot) in accumulators.iter_mut().enumerate() {
        // Bit range covered by this limb: [lo, hi).
        let lo = limb * BITS_PER_LIMB;
        let hi = lo + BITS_PER_LIMB;
        let mut acc = AB::Expr::ZERO;
        // Horner: fold from the high bit down so each step doubles the
        // accumulator then adds the current bit.
        for i in (lo..hi).rev() {
            // Fetch the three operand bits with their respective rotation or
            // shift amounts.
            let b1: AB::Expr = bits[(i + r1 as usize) % WORD_BITS].into();
            let b2: AB::Expr = bits[(i + r2 as usize) % WORD_BITS].into();
            let b3 = get_shifted_bit(i);
            // Arithmetic XOR3 for boolean inputs:
            //     b1 + b2 + b3 - 2*(b1*b2 + b1*b3 + b2*b3) + 4*b1*b2*b3
            let bit_value = b1.xor3(&b2, &b3);
            // Horner step: shift the accumulator left by 1 then add the bit.
            acc = acc.double() + bit_value;
        }
        *acc_slot = acc;
    }

    // Emit one equality per limb between the stored packed value and the
    // reconstructed expression.
    builder.assert_zeros([
        packed[0].into() - core::mem::replace(&mut accumulators[0], AB::Expr::ZERO),
        packed[1].into() - core::mem::replace(&mut accumulators[1], AB::Expr::ZERO),
    ]);
}

/// Assert `packed` equals `Ch(e, f, g)` in packed form.
///
/// # Algorithm
///
/// The two AND-terms are disjoint on boolean inputs, so the XOR collapses to
/// an addition:
///
/// ```text
///     Ch_i = e_i * f_i + (1 - e_i) * g_i
/// ```
///
/// That bit expression has degree 2, so the packed constraint has degree 2.
fn assert_ch_matches<AB: AirBuilder>(
    builder: &mut AB,
    e: &[AB::Var; WORD_BITS],
    f: &[AB::Var; WORD_BITS],
    g: &[AB::Var; WORD_BITS],
    packed: &[AB::Var; U32_LIMBS],
) {
    let mut accumulators: [AB::Expr; U32_LIMBS] = [AB::Expr::ZERO; U32_LIMBS];
    for (limb, acc_slot) in accumulators.iter_mut().enumerate() {
        let lo = limb * BITS_PER_LIMB;
        let hi = lo + BITS_PER_LIMB;
        let mut acc = AB::Expr::ZERO;
        // Horner, high to low bit.
        for i in (lo..hi).rev() {
            // Pull the three operand bits into expressions.
            let ei: AB::Expr = e[i].into();
            let fi: AB::Expr = f[i].into();
            let gi: AB::Expr = g[i].into();
            // Degree-2 bit identity equivalent to (e AND f) XOR (NOT e AND g).
            let ch_i = ei.dup() * fi + (AB::Expr::ONE - ei) * gi;
            // Horner step.
            acc = acc.double() + ch_i;
        }
        *acc_slot = acc;
    }

    // Per-limb equality check.
    builder.assert_zeros([
        packed[0].into() - core::mem::replace(&mut accumulators[0], AB::Expr::ZERO),
        packed[1].into() - core::mem::replace(&mut accumulators[1], AB::Expr::ZERO),
    ]);
}

/// Assert `packed` equals `Maj(a, b, c)` in packed form.
///
/// # Algorithm
///
/// The boolean majority of three inputs satisfies:
///
/// ```text
///     Maj_i = a_i * b_i + c_i * (a_i XOR b_i)
///           = a_i * b_i + c_i * (a_i + b_i - 2 * a_i * b_i)
/// ```
///
/// That expression has degree 3 per bit.
fn assert_maj_matches<AB: AirBuilder>(
    builder: &mut AB,
    a: &[AB::Var; WORD_BITS],
    b: &[AB::Var; WORD_BITS],
    c: &[AB::Var; WORD_BITS],
    packed: &[AB::Var; U32_LIMBS],
) {
    let mut accumulators: [AB::Expr; U32_LIMBS] = [AB::Expr::ZERO; U32_LIMBS];
    for (limb, acc_slot) in accumulators.iter_mut().enumerate() {
        let lo = limb * BITS_PER_LIMB;
        let hi = lo + BITS_PER_LIMB;
        let mut acc = AB::Expr::ZERO;
        // Horner, high to low bit.
        for i in (lo..hi).rev() {
            // Pull the three operand bits into expressions.
            let ai: AB::Expr = a[i].into();
            let bi: AB::Expr = b[i].into();
            let ci: AB::Expr = c[i].into();
            // Degree-3 identity: `a*b + c * xor(a, b)` matches the bitwise
            // majority on boolean inputs.
            let maj_i = ai.dup() * bi.dup() + ci * ai.xor(&bi);
            // Horner step.
            acc = acc.double() + maj_i;
        }
        *acc_slot = acc;
    }

    // Per-limb equality check.
    builder.assert_zeros([
        packed[0].into() - core::mem::replace(&mut accumulators[0], AB::Expr::ZERO),
        packed[1].into() - core::mem::replace(&mut accumulators[1], AB::Expr::ZERO),
    ]);
}
