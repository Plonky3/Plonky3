//! AIR constraint definitions for the Monolith permutation.
//!
//! # Overview
//!
//! Each trace row encodes one full permutation:
//!
//! ```text
//!   inputs --> initial Concrete --> R rounds of (Bars -> Bricks -> Concrete -> RC)
//!                              --> final round    (Bars -> Bricks -> Concrete)
//! ```
//!
//! Per-round checks:
//!
//! - Bits: boolean and reconstruct the Bar input.
//! - S-box: committed Bar output equals chi(bits).
//! - Bricks: each position adds the square of its predecessor.
//! - Concrete + RC: post-state = MDS(Bricks state) + RC (no RC on the final round).
//!
//! # Constraint Degree
//!
//! All constraints are degree ≤ 3.
//!
//! - A `log_blowup = 1` prover accepts at most degree 3.
//! - Chi natively reaches degree 4 (`bit XOR triple_AND`).
//! - The AND product gets its own committed column: a degree-3 binding plus
//!   a degree-2 XOR replace the degree-4 step.
//! - Boolean, reconstruction, Bricks, Concrete constraints stay ≤ 2.

use alloc::vec;
use alloc::vec::Vec;
use core::borrow::Borrow;

use p3_air::utils::pack_bits_le;
use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;
use p3_mds::util::mds_multiply;
use p3_monolith::MonolithBars;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use crate::columns::{MonolithCols, MonolithRoundCols, num_cols};
use crate::generation::generate_trace_rows;

/// Limb widths for the Mersenne31 prime, summing to 31 bits.
///
/// - Three leading 8-bit limbs use the 3-input AND chi S-box.
/// - One trailing 7-bit limb uses the 2-input AND chi S-box.
pub const MERSENNE31_LIMB_BITS: &[usize] = &[8, 8, 8, 7];

/// Limb widths for the Goldilocks prime, summing to 64 bits.
///
/// All eight limbs are 8 bits wide and use the 3-input AND chi S-box.
pub const GOLDILOCKS_8_LIMB_BITS: &[usize] = &[8, 8, 8, 8, 8, 8, 8, 8];

/// Algebraic constraints that pin one full Monolith permutation per trace row.
///
/// # Per-constraint degree
///
/// - Bit boolean check: 2.
/// - Bit reconstruction: 1.
/// - Chi AND product witness: 3 (8-bit limbs) / 2 (7-bit limb).
/// - Chi output equality: 2.
/// - Canonical-pattern walk step: 2 (or 1 for "modulus-zero" positions).
/// - Bricks (Feistel squaring): 2.
/// - Concrete (dense MDS): 2.
/// - Round constant addition: 1.
///
/// Overall max: 3, set by the chi AND product.
#[derive(Debug, Clone)]
pub struct MonolithAir<
    F: PrimeCharacteristicRing,
    const WIDTH: usize,
    const NUM_FULL_ROUNDS: usize,
    const NUM_BARS: usize,
    const FIELD_BITS: usize,
> {
    /// Round constants for each of the `NUM_FULL_ROUNDS` rounds.
    ///
    /// The final round (round `NUM_FULL_ROUNDS`) has no round constants.
    pub(crate) round_constants: [[F; WIDTH]; NUM_FULL_ROUNDS],

    /// Dense MDS matrix for the Concrete layer.
    ///
    /// `mds_matrix[row][col]` is the entry at row `row`, column `col`.
    /// The Concrete layer computes `state = mds_matrix * state`.
    pub(crate) mds_matrix: [[F; WIDTH]; WIDTH],

    /// Bit widths of each limb in the Bar decomposition.
    ///
    /// For Mersenne31: `[8, 8, 8, 7]` (three 8-bit + one 7-bit limb).
    /// For Goldilocks: `[8, 8, 8, 8, 8, 8, 8, 8]` (eight 8-bit limbs).
    ///
    /// The sum of all limb widths must equal `FIELD_BITS`.
    pub(crate) limb_bits: &'static [usize],

    /// Bits of the field modulus, least-significant first.
    ///
    /// Drives the bit-by-bit `< p` walk that enforces canonical bit encodings.
    pub(crate) modulus_lsb_to_msb: [bool; FIELD_BITS],
}

impl<
    F: PrimeCharacteristicRing,
    const WIDTH: usize,
    const NUM_FULL_ROUNDS: usize,
    const NUM_BARS: usize,
    const FIELD_BITS: usize,
> MonolithAir<F, WIDTH, NUM_FULL_ROUNDS, NUM_BARS, FIELD_BITS>
{
    /// Build the AIR from a permutation's parameters.
    ///
    /// # Arguments
    ///
    /// - Round constants, one vector per round that adds them.
    /// - Dense MDS matrix indexed row-first.
    /// - Limb widths driving the chi decomposition.
    ///
    /// # Field invariant
    ///
    /// - The field order must fit in a `u64` (top bit unset).
    ///
    /// # Limb width invariants
    ///
    /// - Sum to the field-bit parameter.
    /// - Every non-trailing entry equals 8.
    /// - Trailing entry lies in `3..=8`.
    ///
    /// # Panics
    ///
    /// - Field-bit parameter exceeds 63.
    /// - Limb widths do not sum to the field-bit parameter.
    /// - Bar applications exceed the state width.
    /// - Any non-trailing limb is not 8.
    /// - Trailing limb is outside `3..=8`.
    pub fn new(
        round_constants: [[F; WIDTH]; NUM_FULL_ROUNDS],
        mds_matrix: [[F; WIDTH]; WIDTH],
        limb_bits: &'static [usize],
    ) -> Self
    where
        F: PrimeField64,
    {
        // Modulus must fit in `FIELD_BITS` bits.
        //
        //     M31         : 0x7FFFFFFF                  fits in 31 bits
        //     Goldilocks  : 0xFFFFFFFF00000001          fits in 64 bits
        //
        // The canonical-bit walk reads bit i of the modulus for i in 0..FIELD_BITS;
        // a higher set bit would never satisfy `X < p` after truncation.
        const { assert!(FIELD_BITS <= 64, "FIELD_BITS must fit in a u64") };
        assert!(
            FIELD_BITS == 64 || F::ORDER_U64 < (1u64 << FIELD_BITS),
            "field modulus does not fit in FIELD_BITS bits"
        );

        // Bit budget:  sum(limbs) == FIELD_BITS  (e.g. 8+8+8+7 = 31 for M31).
        assert_eq!(
            limb_bits.iter().sum::<usize>(),
            FIELD_BITS,
            "limb_bits must sum to FIELD_BITS"
        );

        // Bars touches the first u of t state words; u > t leaves nothing to feed.
        const {
            assert!(NUM_BARS <= WIDTH, "NUM_BARS must not exceed WIDTH");
        }

        // Short limb must come last.
        //
        // - Chi code: non-trailing limbs are hard-wired to 3-input AND.
        // - Chi code: the trailing limb may use 2-input AND instead.
        // - Any other order silently applies the wrong S-box.
        let (last, leading) = limb_bits
            .split_last()
            .expect("limb_bits must contain at least one limb");
        assert!(
            leading.iter().all(|&n| n == 8),
            "all limbs except the last must be exactly 8 bits wide"
        );

        // Trailing limb width in `3..=8`.
        //
        // - Chi is invertible only when `gcd(n, 2) = 1` (and `gcd(n, 3) = 1`
        //   for the 3-input variant).
        // - `3..=8` covers every Monolith parameter set and matches the
        //   precomputed S-box tables.
        assert!(
            (3..=8).contains(last),
            "last limb width must lie in 3..=8 bits, got {last}"
        );

        // Cache the modulus bits LSB-first to align with the bit-decomposition layout.
        //
        //     modulus_lsb_to_msb[i] = (ORDER >> i) & 1 == 1
        let modulus = F::ORDER_U64;
        let modulus_lsb_to_msb = core::array::from_fn(|i| (modulus >> i) & 1 == 1);

        Self {
            round_constants,
            mds_matrix,
            limb_bits,
            modulus_lsb_to_msb,
        }
    }

    /// Extract a dense MDS matrix from any `MdsPermutation` implementation.
    ///
    /// Evaluates the MDS on each standard basis vector to produce the
    /// full `WIDTH × WIDTH` matrix.
    pub fn extract_mds_matrix<Mds>(mds: &Mds) -> [[F; WIDTH]; WIDTH]
    where
        F: Copy,
        Mds: p3_mds::MdsPermutation<F, WIDTH>,
    {
        // Compute each column of the matrix by applying MDS to basis vectors.
        let columns: [[F; WIDTH]; WIDTH] = core::array::from_fn(|col| {
            let mut basis = [F::ZERO; WIDTH];
            basis[col] = F::ONE;
            mds.permute_mut(&mut basis);
            basis
        });
        // Transpose: columns[col][row] → matrix[row][col].
        core::array::from_fn(|row| core::array::from_fn(|col| columns[col][row]))
    }

    /// Generate a trace with `num_hashes` random permutations.
    ///
    /// Uses a deterministic PRNG seeded with `1` for reproducible traces.
    ///
    /// The `extra_capacity_bits` parameter pre-allocates extra memory for
    /// the LDE (low-degree extension) blowup during proving.
    pub fn generate_trace_rows<B: MonolithBars<F, WIDTH> + Sync>(
        &self,
        num_hashes: usize,
        bars: &B,
        extra_capacity_bits: usize,
    ) -> RowMajorMatrix<F>
    where
        F: PrimeField64,
        StandardUniform: Distribution<[F; WIDTH]>,
    {
        // Deterministic PRNG for reproducible test inputs.
        let mut rng = SmallRng::seed_from_u64(1);
        let inputs = (0..num_hashes).map(|_| rng.random()).collect();
        generate_trace_rows(inputs, self, bars, extra_capacity_bits)
    }
}

impl<
    F: PrimeCharacteristicRing + Sync,
    const WIDTH: usize,
    const NUM_FULL_ROUNDS: usize,
    const NUM_BARS: usize,
    const FIELD_BITS: usize,
> BaseAir<F> for MonolithAir<F, WIDTH, NUM_FULL_ROUNDS, NUM_BARS, FIELD_BITS>
{
    /// Returns the number of trace columns (the AIR width).
    fn width(&self) -> usize {
        num_cols::<WIDTH, NUM_FULL_ROUNDS, NUM_BARS, FIELD_BITS>()
    }

    /// No next-row columns. Each permutation is fully constrained within one row.
    fn main_next_row_columns(&self) -> Vec<usize> {
        vec![]
    }
}

impl<
    AB: AirBuilder,
    const WIDTH: usize,
    const NUM_FULL_ROUNDS: usize,
    const NUM_BARS: usize,
    const FIELD_BITS: usize,
> Air<AB> for MonolithAir<AB::F, WIDTH, NUM_FULL_ROUNDS, NUM_BARS, FIELD_BITS>
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &MonolithCols<AB::Var, WIDTH, NUM_FULL_ROUNDS, NUM_BARS, FIELD_BITS> =
            main.current_slice().borrow();

        // Initialize the running state from the committed input columns.
        let mut state: [AB::Expr; WIDTH] = local.inputs.map(|x| x.into());

        // Initial Concrete layer: multiply the input state by the MDS matrix.
        // This is a linear operation, so the state remains degree 1.
        mds_multiply::<_, _, WIDTH>(&mut state, &self.mds_matrix);

        // Full rounds: Bars → Bricks → Concrete → AddRoundConstants.
        for round_idx in 0..NUM_FULL_ROUNDS {
            eval_round::<AB, WIDTH, NUM_BARS, FIELD_BITS>(
                &mut state,
                &local.full_rounds[round_idx],
                &self.mds_matrix,
                Some(&self.round_constants[round_idx]),
                self.limb_bits,
                &self.modulus_lsb_to_msb,
                builder,
            );
        }

        // Final round: Bars → Bricks → Concrete (no round constants).
        eval_round::<AB, WIDTH, NUM_BARS, FIELD_BITS>(
            &mut state,
            &local.final_round,
            &self.mds_matrix,
            None,
            self.limb_bits,
            &self.modulus_lsb_to_msb,
            builder,
        );
    }
}

/// Evaluate constraints for one Monolith round.
///
/// Layers applied in order:
///
/// - Bars
/// - Bricks
/// - Concrete
/// - Round constants (skipped on the final round)
///
/// The running state is reset to the committed post-state on exit so the next
/// round restarts from a degree-1 expression.
#[inline]
fn eval_round<
    AB: AirBuilder,
    const WIDTH: usize,
    const NUM_BARS: usize,
    const FIELD_BITS: usize,
>(
    state: &mut [AB::Expr; WIDTH],
    round: &MonolithRoundCols<AB::Var, WIDTH, NUM_BARS, FIELD_BITS>,
    mds_matrix: &[[AB::F; WIDTH]; WIDTH],
    round_constants: Option<&[AB::F; WIDTH]>,
    limb_bits: &[usize],
    modulus_lsb_to_msb: &[bool; FIELD_BITS],
    builder: &mut AB,
) {
    // Phase 1: Bars constraints (per Bar slot):
    //
    //     - boolean checks on each bit
    //     - linear reconstruction back to the Bar input
    //     - chi AND product witness equation (delegated to `eval_bar_sbox`)
    //     - chi S-box output equality
    //     - canonical bit-pattern walk (rules out any encoding `>= p`)
    for (bar_idx, (((input_bits, chi_products), &bar_out), match_flags)) in round
        .bars_input_bits
        .iter()
        .zip(round.bars_chi_products.iter())
        .zip(round.bars_output.iter())
        .zip(round.bars_match_flags.iter())
        .enumerate()
    {
        // Boolean: every committed bit is 0 or 1.
        builder.assert_bools(*input_bits);

        // Lift committed cells to expressions for symbolic algebra.
        let bits: [AB::Expr; FIELD_BITS] = input_bits.map(|b| b.into());
        let chi: [AB::Expr; FIELD_BITS] = chi_products.map(|b| b.into());
        let mflag: [AB::Expr; FIELD_BITS] = match_flags.map(|b| b.into());

        // Reconstruction:  sum_i bits[i] * 2^i  ==  state[bar_idx].
        let reconstructed: AB::Expr = pack_bits_le(bits.iter().cloned());
        builder.assert_eq(reconstructed, state[bar_idx].clone());

        // S-box: committed Bar output == chi(bits), degree 2 thanks to the
        // committed AND product witnesses.
        let sbox_output = eval_bar_sbox::<AB, FIELD_BITS>(&bits, &chi, limb_bits, builder);
        builder.assert_eq(AB::Expr::from(bar_out), sbox_output);

        // Canonical bit-pattern walk (MSB → LSB).
        //
        // Define `m_i` = "bits[i..FIELD_BITS] still match the modulus prefix".
        // Start above the MSB with `m_top = 1` (no info yet):
        //
        //     p_i = 1 : m_i = m_{i+1} * bits[i]
        //     p_i = 0 : m_i = m_{i+1}                  (linear pass-through)
        //               assert m_{i+1} * bits[i] = 0   (a 1 here would force X > p)
        //
        // Final assertion `m_0 = 0` rejects the encoding `bits == p`, which
        // is the only forbidden one in `[p, 2^FIELD_BITS - 1]` that survives
        // the side-constraints above.
        let mut prev = AB::Expr::ONE;
        for i in (0..FIELD_BITS).rev() {
            let m_i = mflag[i].clone();
            let x_i = bits[i].clone();
            if modulus_lsb_to_msb[i] {
                builder.assert_eq(m_i.clone(), prev.clone() * x_i);
            } else {
                builder.assert_eq(m_i.clone(), prev.clone());
                builder.assert_zero(prev.clone() * x_i);
            }
            prev = m_i;
        }
        builder.assert_zero(prev);
    }

    // Phase 2: Build the post-Bars state.
    //
    //     positions 0..u   : committed Bar outputs (degree 1)
    //     positions u..t   : pass through unchanged (degree 1)
    let mut post_bars = state.clone();
    for (bar_idx, &bar_out) in round.bars_output.iter().enumerate() {
        post_bars[bar_idx] = bar_out.into();
    }

    // Phase 3: Bricks (Feistel Type-3 squaring).
    //
    //     post[0] = bars[0]
    //     post[i] = bars[i] + bars[i-1]^2   for i >= 1
    let mut post_bricks = core::array::from_fn(|i| {
        if i == 0 {
            post_bars[0].clone()
        } else {
            post_bars[i].clone() + post_bars[i - 1].clone().square()
        }
    });

    // Phase 4: Concrete (linear, keeps degree 2).
    mds_multiply::<_, _, WIDTH>(&mut post_bricks, mds_matrix);

    // Phase 5: Round constants (skipped on the final round).
    if let Some(rc) = round_constants {
        for (s, c) in post_bricks.iter_mut().zip(rc.iter()) {
            *s += AB::Expr::from(c.clone());
        }
    }

    // Phase 6: Bind to committed post-state (caps degree across rounds).
    for (computed, &committed) in post_bricks.into_iter().zip(round.post.iter()) {
        builder.assert_eq(computed, committed);
    }

    // Reset the running expressions to the freshly-bound committed values.
    *state = round.post.map(|x| x.into());
}

/// Evaluate the chi S-box of one Bar on its bit decomposition.
///
/// # Overview
///
/// - Apply chi independently to each limb (widths from the input slice).
/// - Recombine output bits via little-endian packing.
/// - Use the committed AND product witnesses to cap degree at 3.
///
/// # Chi per limb (width n, indices mod n)
///
/// ```text
///   8-bit:  out[j] = x[j-1] XOR ((NOT x[j-2]) AND x[j-3] AND x[j-4])
///   7-bit:  out[j] = x[j-1] XOR ((NOT x[j-2]) AND x[j-3])
/// ```
fn eval_bar_sbox<AB: AirBuilder, const FIELD_BITS: usize>(
    bits: &[AB::Expr; FIELD_BITS],
    chi_products: &[AB::Expr; FIELD_BITS],
    limb_bits: &[usize],
    builder: &mut AB,
) -> AB::Expr {
    // Accumulator for the recombined field element.
    let mut result = AB::Expr::ZERO;

    // Running offset into the global bit array; advances by limb width.
    let mut bit_offset = 0;

    for (limb_idx, &n) in limb_bits.iter().enumerate() {
        // Slice the input bits and committed AND products for this limb.
        let x = &bits[bit_offset..bit_offset + n];
        let chi = &chi_products[bit_offset..bit_offset + n];

        // Only the trailing limb may be narrower than 8 bits; it uses the
        // 2-input AND variant of chi.
        let is_last_reduced = limb_idx == limb_bits.len() - 1 && n < 8;

        //     sub(j, k) = (j - k) mod n
        let sub = |base: usize, offset: usize| (base + n - (offset % n)) % n;

        // Bind each AND product witness:
        //
        //     8-bit:  chi[j] = (1 - x[j-2]) * x[j-3] * x[j-4]
        //     7-bit:  chi[j] = (1 - x[j-2]) * x[j-3]
        for j in 0..n {
            let andn = x[sub(j, 2)].clone().andn(&x[sub(j, 3)].clone());
            let expected = if is_last_reduced {
                andn
            } else {
                andn * x[sub(j, 4)].clone()
            };
            builder.assert_eq(chi[j].clone(), expected);
        }

        // Output bit:  out[j] = x[j-1] XOR chi[j]  (degree 2).
        let get_out_bit = |j: usize| -> AB::Expr { x[sub(j, 1)].clone().xor(&chi[j].clone()) };

        // Pack the limb's bits into one field element (Horner).
        let limb_value: AB::Expr = pack_bits_le((0..n).map(get_out_bit));

        // Shift into global position and accumulate:  result += limb * 2^offset.
        let limb_shift = AB::F::from_u64(1u64 << bit_offset);
        result += limb_value * limb_shift;

        bit_offset += n;
    }

    result
}
