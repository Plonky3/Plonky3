//! The round-1 message: the univariate skip, swept window by window.
//!
//! A row of 64 bits is a polynomial's values on `S`; the message is its eq-weighted sum on `Lambda`.
//!
//! The fixed coordinates split the eq weight of a row into three factors:
//!
//! - Small: the three rows `k` of a medium position sum as `x^k` products in the AES field.
//! - Medium: the sixteen positions of a window map to the challenge field under the weights `g^b / D`.
//! - Outer: the drawn coordinates, split into a low and a high table.
//!
//! The sweep's output is the message without the factor `C_s`, which the caller restores.

mod convert;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use p3_binary_field::{Poly192, Rijndael8b};
use p3_field::PrimeCharacteristicRing;
use p3_maybe_rayon::prelude::*;

use self::convert::{Convert, ConvertTables};
use super::fixed::{MEDIUM_VALUES, N_FIXED, c_s, d_inv};
use super::{EqHalves, Padding};
use crate::byteslice::transpose_bits;
use crate::skip::{K_SKIP, SKIP_SIZE, SkipLde};

/// Bits of a window: the skip, then the fixed coordinates.
pub(crate) const WINDOW_LOG: usize = K_SKIP + N_FIXED;

/// Bytes of a medium position: eight rows of eight bytes.
const MEDIUM_BYTES: usize = 64;

/// Most variables of the high eq table: one task per high index, enough to balance every worker.
const EQ_HIGH_VARS: usize = 9;

/// The weights `x^k` that sum the eight rows of a medium position.
const ROW_POWERS: [Rijndael8b; 8] = [
    Rijndael8b::from_byte(1),
    Rijndael8b::from_byte(2),
    Rijndael8b::from_byte(4),
    Rijndael8b::from_byte(8),
    Rijndael8b::from_byte(16),
    Rijndael8b::from_byte(32),
    Rijndael8b::from_byte(64),
    Rijndael8b::from_byte(128),
];

/// Which medium positions of a window can hold a nonzero bit.
#[derive(Clone, Debug)]
struct MediumCounts {
    /// Masks a window index to its window within a block.
    mask: usize,
    /// For each window of a block, its medium positions before the padding.
    counts: Vec<usize>,
}

impl MediumCounts {
    /// The live positions of every window of a block.
    fn new(padding: &Padding) -> Self {
        // A block below a window cannot skip at window granularity.
        if padding.k_log < WINDOW_LOG {
            return Self {
                mask: 0,
                counts: vec![MEDIUM_VALUES],
            };
        }
        let windows = 1usize << (padding.k_log - WINDOW_LOG);
        let counts = (0..windows)
            .map(|w| {
                let left = padding.useful_bits.saturating_sub(w << WINDOW_LOG);
                left.div_ceil(8 * MEDIUM_BYTES).min(MEDIUM_VALUES)
            })
            .collect();
        Self {
            mask: windows - 1,
            counts,
        }
    }

    /// The live medium positions of window `w`.
    fn of(&self, w: usize) -> usize {
        self.counts[w & self.mask]
    }
}

/// One worker's scratch and running sums.
struct WorkerState {
    /// The current high index's per-lane sums.
    convert: Convert,
    /// One window's `a b` bytes, a row per medium position.
    ab_rows: [[u8; SKIP_SIZE]; MEDIUM_VALUES],
    /// One window's transposed `c` bytes, a row per medium position.
    c_rows: [[u8; SKIP_SIZE]; MEDIUM_VALUES],
    /// The worker's sums over its high indices, `a b` then `c`.
    sums: [[Poly192; SKIP_SIZE]; 2],
}

impl WorkerState {
    /// Empty sums.
    #[cfg_attr(
        not(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512bw",
            target_feature = "avx512vbmi",
            target_feature = "gfni",
            target_feature = "vpclmulqdq"
        )),
        expect(
            clippy::missing_const_for_fn,
            reason = "the sliced convert starts from registers"
        )
    )]
    fn new() -> Self {
        Self {
            convert: Convert::new(),
            ab_rows: [[0; SKIP_SIZE]; MEDIUM_VALUES],
            c_rows: [[0; SKIP_SIZE]; MEDIUM_VALUES],
            sums: [[Poly192::ZERO; SKIP_SIZE]; 2],
        }
    }

    /// Two workers' sums added.
    fn merge(mut self: Box<Self>, other: &Self) -> Box<Self> {
        for (x, y) in self
            .sums
            .iter_mut()
            .flatten()
            .zip(other.sums.iter().flatten())
        {
            *x += *y;
        }
        self
    }
}

/// The round-1 message of packed `a` and `b` at the eq coordinates `r`, on `Lambda`.
///
/// # Panics
///
/// - The cube is below a window, or a length disagrees with `m`.
pub(crate) fn round1_message(
    a: &[u8],
    b: &[u8],
    m: usize,
    r: &[Poly192],
    lde: &SkipLde,
    padding: &Padding,
) -> Vec<Poly192> {
    assert!(m >= WINDOW_LOG, "the sweep needs a whole window");
    assert_eq!(a.len(), (1 << m) / 8);
    assert_eq!(b.len(), (1 << m) / 8);
    assert_eq!(r.len(), m - K_SKIP);

    // Phase 1: the outer eq split into a low and a high half, `1 / D` riding the low one.
    let outer = &r[N_FIXED..];
    let n_lo = outer.len().saturating_sub(EQ_HIGH_VARS);
    let eq = EqHalves::new(outer, n_lo);
    let d_inv = d_inv();
    let sweep = Sweep {
        a,
        b,
        lde,
        eq_lo: eq.low.iter().map(|&e| e * d_inv).collect(),
        n_lo,
        medium: MediumCounts::new(padding),
        tables: ConvertTables::new(),
    };

    // Phase 2: one task per high eq index, each worker summing into its own state.
    let sums = (0..eq.high.len())
        .into_par_iter()
        .par_fold_reduce(
            // Boxed, so the reduction tree moves pointers rather than kilobytes.
            || Box::new(WorkerState::new()),
            |mut state, hi| {
                sweep.high(&mut state, hi, eq.high[hi]);
                state
            },
            |a, b| a.merge(&b),
        )
        .sums;

    // Phase 3: `c` was summed on `S`, being linear: extend it once, then restore `C_s`.
    let mut c = sums[1].to_vec();
    lde.extend(&mut c);
    let c_s = c_s();
    sums[0]
        .iter()
        .zip(c)
        .map(|(&ab, c)| c_s * (ab + c))
        .collect()
}

/// What every task of one sweep shares.
struct Sweep<'a> {
    /// The packed `a`.
    a: &'a [u8],
    /// The packed `b`.
    b: &'a [u8],
    /// The extension from `S` to `Lambda`.
    lde: &'a SkipLde,
    /// The low half of the outer eq table, times `1 / D`.
    eq_lo: Vec<Poly192>,
    /// The low half's variables.
    n_lo: usize,
    /// The live medium positions of each window.
    medium: MediumCounts,
    /// The convert's tables, on targets that use them.
    tables: ConvertTables,
}

impl Sweep<'_> {
    /// Add every window under high eq index `hi`, at weight `eq_hi`.
    fn high(&self, state: &mut WorkerState, hi: usize, eq_hi: Poly192) {
        state.convert = Convert::new();
        let first = hi << self.n_lo;
        for (lo, &eq_lo) in self.eq_lo.iter().enumerate() {
            let w = first | lo;
            // A full window has a constant trip count, which the unroll depends on.
            match self.medium.of(w) {
                0 => {}
                MEDIUM_VALUES => self.window::<true>(state, w, MEDIUM_VALUES, eq_lo),
                live => self.window::<false>(state, w, live, eq_lo),
            }
        }

        // The outer fold by the high weight.
        let (ab, c) = state.convert.values();
        for (sum, values) in state.sums.iter_mut().zip([ab, c]) {
            for (s, v) in sum.iter_mut().zip(values) {
                *s += eq_hi * v;
            }
        }
    }

    /// Add window `w`'s first `live` medium positions at weight `eq_lo`.
    #[inline(always)]
    fn window<const FULL: bool>(
        &self,
        state: &mut WorkerState,
        w: usize,
        live: usize,
        eq_lo: Poly192,
    ) {
        let live = if FULL { MEDIUM_VALUES } else { live };
        // A window is its medium positions, one after another.
        let base = w * MEDIUM_VALUES;
        for b_med in 0..live {
            let at = base + b_med;
            let [a, b] = [self.a, self.b].map(|p| &p.as_chunks::<MEDIUM_BYTES>().0[at]);
            // `a b`: extended, multiplied, and summed over the rows by powers of `x`.
            state.ab_rows[b_med] = product_bytes(self.lde, a, b);
            // `c = a AND b`, linear: transposed so lane `s` holds skip point `s` of each row.
            let c: [u8; MEDIUM_BYTES] = core::array::from_fn(|i| a[i] & b[i]);
            state.c_rows[b_med] = transpose_bits(&c);
        }
        state.convert.accumulate(
            &self.tables,
            &state.ab_rows[..live],
            &state.c_rows[..live],
            eq_lo,
        );
    }
}

/// The `a b` bytes of one medium position: `sum_k x^k LDE(a_k) LDE(b_k)` per point of `Lambda`.
#[inline]
pub(crate) fn product_bytes(
    lde: &SkipLde,
    a: &[u8; MEDIUM_BYTES],
    b: &[u8; MEDIUM_BYTES],
) -> [u8; SKIP_SIZE] {
    let mut out = [Rijndael8b::ZERO; SKIP_SIZE];
    lde.bytes.weighted_product_sum(a, b, &ROW_POWERS, &mut out);
    out.map(Rijndael8b::to_byte)
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::zerocheck::fixed::eq_coordinates;

    #[test]
    fn skipping_padding_changes_no_value() {
        // Invariant: on blocks zero past their useful bits, skipping whole medium positions is the dense message.
        //
        // Fixture state, (k_log, useful, m): a boundary inside a position, a whole position skipped, several blocks.
        let lde = SkipLde::new();
        for (k_log, useful, m) in [
            (14usize, 9_000usize, 14usize),
            (15, 31_401, 15),
            (14, 512, 17),
        ] {
            let mut rng = SmallRng::seed_from_u64(useful as u64);
            let bits = |rng: &mut SmallRng| -> Vec<u8> {
                (0..(1usize << m) / 8)
                    .map(|i| {
                        let byte: u8 = rng.random();
                        // Clear the bits of this byte past the useful ones.
                        let offset = (8 * i) % (1 << k_log);
                        let keep = useful.saturating_sub(offset).min(8);
                        byte & ((1u16 << keep) - 1) as u8
                    })
                    .collect()
            };
            let (a, b) = (bits(&mut rng), bits(&mut rng));
            let outer: Vec<Poly192> = (0..m - WINDOW_LOG).map(|_| rng.random()).collect();
            let r = eq_coordinates(&outer);
            let padding = Padding {
                k_log,
                useful_bits: useful,
            };
            assert_eq!(
                round1_message(&a, &b, m, &r, &lde, &Padding::dense(m)),
                round1_message(&a, &b, m, &r, &lde, &padding),
                "k_log={k_log}, useful={useful}, m={m}"
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        #[test]
        fn the_product_bytes_are_the_scalar_route(seed: u64) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let lde = SkipLde::new();
            let a: [u8; 64] = core::array::from_fn(|_| rng.random());
            let b: [u8; 64] = core::array::from_fn(|_| rng.random());
            let mut want = [Rijndael8b::ZERO; SKIP_SIZE];
            let (mut ea, mut eb) = ([Rijndael8b::ZERO; SKIP_SIZE], [Rijndael8b::ZERO; SKIP_SIZE]);
            for k in 0..8 {
                lde.bytes.apply(&a[8 * k..8 * k + 8], &mut ea);
                lde.bytes.apply(&b[8 * k..8 * k + 8], &mut eb);
                for l in 0..SKIP_SIZE {
                    want[l] += ea[l] * eb[l] * ROW_POWERS[k];
                }
            }
            prop_assert_eq!(product_bytes(&lde, &a, &b), want.map(Rijndael8b::to_byte));
        }
    }
}
