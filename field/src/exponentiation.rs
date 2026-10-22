use alloc::vec::Vec;
use core::{array, iter};

use p3_maybe_rayon::prelude::*;

use crate::{ExtensionField, Field, PackedFieldExtension, PackedValue, PrimeCharacteristicRing};

pub(crate) const fn bits_u64(n: u64) -> usize {
    (64 - n.leading_zeros()) as usize
}

/// Compute the exponential `x -> x^1717986917` using a custom addition chain.
///
/// This map computes the fifth root of `x` if `x` is a member of the field `Mersenne31`.
/// This follows from the computation: `5 * 1717986917 = 4*(2^31 - 2) + 1 = 1 mod p - 1`.
#[must_use]
pub fn exp_1717986917<R: PrimeCharacteristicRing>(val: R) -> R {
    // Note the binary expansion: 1717986917 = 1100110011001100110011001100101_2
    // This uses 30 Squares + 7 Multiplications => 37 Operations total.
    // Suspect it's possible to improve this with enough effort. For example 1717986918 takes only 4 Multiplications.
    let p1 = val;
    let p10 = p1.square();
    let p11 = p10.dup() * p1;
    let p101 = p10 * p11.dup();
    let p110000 = p11.exp_power_of_2(4);
    let p110011 = p110000 * p11.dup();
    let p11001100000000 = p110011.exp_power_of_2(8);
    let p11001100110011 = p11001100000000.dup() * p110011;
    let p1100110000000000000000 = p11001100000000.exp_power_of_2(8);
    let p1100110011001100110011 = p1100110000000000000000 * p11001100110011;
    let p11001100110011001100110000 = p1100110011001100110011.exp_power_of_2(4);
    let p11001100110011001100110011 = p11001100110011001100110000 * p11;
    let p1100110011001100110011001100000 = p11001100110011001100110011.exp_power_of_2(5);
    p1100110011001100110011001100000 * p101
}

/// Compute the exponential `x -> x^1420470955` using a custom addition chain.
///
/// This map computes the third root of `x` if `x` is a member of the field `KoalaBear`.
/// This follows from the computation: `3 * 1420470955 = 2*(2^31 - 2^24) + 1 = 1 mod (p - 1)`.
#[must_use]
pub fn exp_1420470955<R: PrimeCharacteristicRing>(val: R) -> R {
    // Note the binary expansion: 1420470955 = 1010100101010101010101010101011_2
    // This uses 29 Squares + 7 Multiplications => 36 Operations total.
    // Suspect it's possible to improve this with enough effort.
    let p1 = val;
    let p100 = p1.exp_power_of_2(2);
    let p101 = p100.dup() * p1.dup();
    let p10000 = p100.exp_power_of_2(2);
    let p10101 = p10000 * p101;
    let p10101000000 = p10101.exp_power_of_2(6);
    let p10101010101 = p10101000000.dup() * p10101.dup();
    let p101010010101 = p10101000000 * p10101010101.dup();
    let p101010010101000000000000 = p101010010101.exp_power_of_2(12);
    let p101010010101010101010101 = p101010010101000000000000 * p10101010101;
    let p101010010101010101010101000000 = p101010010101010101010101.exp_power_of_2(6);
    let p101010010101010101010101010101 = p101010010101010101010101000000 * p10101;
    let p1010100101010101010101010101010 = p101010010101010101010101010101.square();
    p1010100101010101010101010101010 * p1
}

/// Compute the exponential `x -> x^1725656503` using a custom addition chain.
///
/// This map computes the seventh root of `x` if `x` is a member of the field `BabyBear`.
/// This follows from the computation: `7 * 1725656503 = 6*(2^31 - 2^27) + 1 = 1 mod (p - 1)`.
#[must_use]
pub fn exp_1725656503<R: PrimeCharacteristicRing>(val: R) -> R {
    // Note the binary expansion: 1725656503 = 1100110110110110110110110110111_2
    // This uses 29 Squares + 8 Multiplications => 37 Operations total.
    // Suspect it's possible to improve this with enough effort.
    let p1 = val;
    let p10 = p1.square();
    let p11 = p10 * p1.dup();
    let p110 = p11.square();
    let p111 = p110.dup() * p1;
    let p11000 = p110.exp_power_of_2(2);
    let p11011 = p11000.dup() * p11;
    let p11000000 = p11000.exp_power_of_2(3);
    let p11011011 = p11000000.dup() * p11011;
    let p110011011 = p11011011.dup() * p11000000;
    let p110011011000000000 = p110011011.exp_power_of_2(9);
    let p110011011011011011 = p110011011000000000 * p11011011.dup();
    let p110011011011011011000000000 = p110011011011011011.exp_power_of_2(9);
    let p110011011011011011011011011 = p110011011011011011000000000 * p11011011;
    let p1100110110110110110110110110000 = p110011011011011011011011011.exp_power_of_2(4);
    p1100110110110110110110110110000 * p111
}

/// Compute the exponential `x -> x^10540996611094048183` using a custom addition chain.
///
/// This map computes the seventh root of `x` if `x` is a member of the field `Goldilocks`.
/// This follows from the computation: `7 * 10540996611094048183 = 4*(2^64 - 2**32) + 1 = 1 mod (p - 1)`.
#[must_use]
pub fn exp_10540996611094048183<R: PrimeCharacteristicRing>(val: R) -> R {
    // Note the binary expansion: 10540996611094048183 = 1001001001001001001001001001000110110110110110110110110110110111_2.
    // This uses 63 Squares + 8 Multiplications => 71 Operations total.
    // Suspect it's possible to improve this a little with enough effort.
    let p1 = val;
    let p10 = p1.square();
    let p11 = p10.dup() * p1;
    let p100 = p10.square();
    let p111 = p100.dup() * p11.dup();
    let p100000000000000000000000000000000 = p100.exp_power_of_2(30);
    let p100000000000000000000000000000011 = p100000000000000000000000000000000 * p11;
    let p100000000000000000000000000000011000 =
        p100000000000000000000000000000011.exp_power_of_2(3);
    let p100100000000000000000000000000011011 =
        p100000000000000000000000000000011000 * p100000000000000000000000000000011;
    let p100100000000000000000000000000011011000000 =
        p100100000000000000000000000000011011.exp_power_of_2(6);
    let p100100100100000000000000000000011011011011 =
        p100100000000000000000000000000011011000000 * p100100000000000000000000000000011011.dup();
    let p100100100100000000000000000000011011011011000000000000 =
        p100100100100000000000000000000011011011011.exp_power_of_2(12);
    let p100100100100100100100100000000011011011011011011011011 =
        p100100100100000000000000000000011011011011000000000000
            * p100100100100000000000000000000011011011011;
    let p100100100100100100100100000000011011011011011011011011000000 =
        p100100100100100100100100000000011011011011011011011011.exp_power_of_2(6);
    let p100100100100100100100100100100011011011011011011011011011011 =
        p100100100100100100100100000000011011011011011011011011000000
            * p100100000000000000000000000000011011;
    let p1001001001001001001001001001000110110110110110110110110110110000 =
        p100100100100100100100100100100011011011011011011011011011011.exp_power_of_2(4);

    p1001001001001001001001001001000110110110110110110110110110110000 * p111
}

/// Window width that `FixedBase::new` uses.
pub const DEFAULT_WINDOW_BITS: u32 = 8;

/// Widest window a table may use, which caps one row at `2^16` elements.
pub const MAX_WINDOW_BITS: u32 = 16;

/// Packed exponentiations that `FixedBase::pow_into` advances together.
const LOCKSTEP_CHAINS: usize = 8;

/// Powers of a fixed base, read from precomputed windowed tables.
///
/// Row `i` holds `base^(d * 2^(w * i))` for every `w`-bit digit `d`, so a power is one product per digit.
///
/// Running time depends on the exponent, so the exponents must be public.
#[derive(Clone, Debug)]
pub struct FixedBase<F> {
    window_bits: u32,
    max_bits: u32,
    table: Vec<F>,
}

impl<F: Field> FixedBase<F> {
    /// Tables for every exponent below `2^max_bits`, with the default window.
    #[must_use]
    pub fn new(base: F, max_bits: u32) -> Self {
        Self::with_window(base, max_bits, DEFAULT_WINDOW_BITS)
    }

    /// Tables for every exponent below `2^max_bits`, with `window_bits`-bit digits.
    ///
    /// # Panics
    ///
    /// Panics if `window_bits` is zero or above `MAX_WINDOW_BITS`.
    #[must_use]
    pub fn with_window(base: F, max_bits: u32, window_bits: u32) -> Self {
        assert!(
            (1..=MAX_WINDOW_BITS).contains(&window_bits),
            "window of {window_bits} bits is outside 1..={MAX_WINDOW_BITS}"
        );
        let row_len = 1 << window_bits;
        // An empty exponent range still keeps one row, so the zero exponent has its entry.
        let rows = max_bits.div_ceil(window_bits).max(1) as usize;

        // Each row's base is the previous one raised to 2^w, which is w squarings.
        let row_bases: Vec<F> =
            iter::successors(Some(base), |b| Some(b.exp_power_of_2(window_bits as usize)))
                .take(rows)
                .collect();

        let mut table = F::zero_vec(rows * row_len);
        table
            .par_chunks_exact_mut(row_len)
            .zip(row_bases.par_iter())
            .for_each_min_task_bytes(row_len * size_of::<F>(), |(row, &row_base)| {
                fill_window_row(row_base, row);
            });

        Self {
            window_bits,
            max_bits,
            table,
        }
    }

    /// Bits of one digit.
    #[must_use]
    pub const fn window_bits(&self) -> u32 {
        self.window_bits
    }

    /// Exponents must lie below `2^max_bits`.
    #[must_use]
    pub const fn max_bits(&self) -> u32 {
        self.max_bits
    }

    /// The base raised to a little-endian multi-limb exponent.
    ///
    /// # Panics
    ///
    /// Panics if the exponent is at least `2^max_bits`.
    #[must_use]
    pub fn pow<const L: usize>(&self, exponent: &[u64; L]) -> F {
        // An exponent without limbs is zero.
        if L == 0 {
            return F::ONE;
        }
        let rows = self.rows_for(exponent);
        let w = self.window_bits;
        let row_len = 1 << w;

        let mut acc = self.table[DigitPosition::new(0, w).read(exponent)];
        for (row, entries) in self
            .table
            .chunks_exact(row_len)
            .enumerate()
            .take(rows)
            .skip(1)
        {
            // A zero digit selects the identity, so its product is skipped.
            let digit = DigitPosition::new(row * w as usize, w).read(exponent);
            if digit != 0 {
                acc *= entries[digit];
            }
        }
        acc
    }

    /// Writes the base raised to `exponents[i]` into `out[i]` for every `i`.
    ///
    /// Exponents run in lockstep across the lanes of the packing of `F` over `B`.
    ///
    /// # Panics
    ///
    /// Panics if the slices differ in length or an exponent is at least `2^max_bits`.
    pub fn pow_into<B: Field, const L: usize>(&self, exponents: &[[u64; L]], out: &mut [F])
    where
        F: ExtensionField<B>,
    {
        assert_eq!(exponents.len(), out.len(), "one output per exponent");
        // Exponents without limbs are all zero.
        if L == 0 {
            out.fill(F::ONE);
            return;
        }
        let group = LOCKSTEP_CHAINS * B::Packing::WIDTH;
        // Every exponent in a group reads one table entry per row and writes one element.
        let item_bytes = group * (self.rows() + 1) * size_of::<F>();

        out.par_chunks_mut(group)
            .zip(exponents.par_chunks(group))
            .for_each_min_task_bytes(item_bytes, |(out, exponents)| {
                if out.len() == group {
                    self.pow_group::<B, L>(exponents, out);
                } else {
                    for (o, e) in out.iter_mut().zip(exponents) {
                        *o = self.pow(e);
                    }
                }
            });
    }

    /// Advances one full group of exponents through the rows in lockstep packed chains.
    fn pow_group<B: Field, const L: usize>(&self, exponents: &[[u64; L]], out: &mut [F])
    where
        F: ExtensionField<B>,
    {
        let width = B::Packing::WIDTH;
        let w = self.window_bits;

        // The group runs to its longest exponent, which is the bit length of their union.
        let union = exponents.iter().fold([0; L], |mut acc, e| {
            acc.iter_mut().zip(e).for_each(|(a, &x)| *a |= x);
            acc
        });
        let rows = self.rows_for(&union);

        // Lane `l` of chain `c` reads the entry its exponent's digit selects in the row.
        let gather = |entries: &[F], chain: usize, digit: DigitPosition<L>| {
            F::ExtensionPacking::from_ext_fn(|lane| {
                entries[digit.read(&exponents[chain * width + lane])]
            })
        };

        let mut entries = self.table.chunks_exact(1 << w);
        let first = entries.next().expect("a table has at least one row");
        let mut acc: [F::ExtensionPacking; LOCKSTEP_CHAINS] =
            array::from_fn(|chain| gather(first, chain, DigitPosition::new(0, w)));
        for (row, entries) in entries.enumerate().take(rows - 1) {
            let digit = DigitPosition::new((row + 1) * w as usize, w);
            for (chain, a) in acc.iter_mut().enumerate() {
                *a *= gather(entries, chain, digit);
            }
        }

        for (dst, src) in out.chunks_exact_mut(width).zip(&acc) {
            src.to_ext_slice(dst);
        }
    }

    /// Rows in the table.
    const fn rows(&self) -> usize {
        self.table.len() >> self.window_bits
    }

    /// Rows an exponent's digits occupy, at least one.
    ///
    /// # Panics
    ///
    /// Panics if the exponent is at least `2^max_bits`.
    fn rows_for<const L: usize>(&self, exponent: &[u64; L]) -> usize {
        let bits = exponent
            .iter()
            .rposition(|&limb| limb != 0)
            .map_or(0, |top| 64 * top + bits_u64(exponent[top]));
        assert!(
            bits <= self.max_bits as usize,
            "exponent of {bits} bits exceeds the table's {} bits",
            self.max_bits
        );
        bits.div_ceil(self.window_bits as usize).max(1)
    }
}

/// Fills `row[d] = base^d` for every `d`, doubling the filled prefix with independent products.
fn fill_window_row<F: Field>(base: F, row: &mut [F]) {
    row[0] = F::ONE;
    let mut filled = 1;
    while filled < row.len() {
        // base^filled, from the entry half way along the filled prefix.
        let step = if filled == 1 {
            base
        } else {
            row[filled / 2].square()
        };
        let (lo, hi) = row.split_at_mut(filled);
        hi[0] = step;
        for (h, &l) in hi[1..filled].iter_mut().zip(&lo[1..]) {
            *h = l * step;
        }
        filled *= 2;
    }
}

/// Where one `w`-bit digit sits in a little-endian exponent of `L` limbs.
///
/// It is the same for every exponent, so a row resolves it once for all of its lanes.
#[derive(Clone, Copy)]
struct DigitPosition<const L: usize> {
    limb: usize,
    shift: u32,
    straddles: bool,
    mask: u64,
}

impl<const L: usize> DigitPosition<L> {
    /// The digit starting at bit `bit`, which must lie inside the exponent.
    #[inline(always)]
    const fn new(bit: usize, w: u32) -> Self {
        let (limb, shift) = (bit / 64, (bit % 64) as u32);
        Self {
            limb,
            shift,
            // A digit crossing a limb boundary takes its top bits from the next limb.
            straddles: shift + w > 64 && limb + 1 < L,
            mask: (1 << w) - 1,
        }
    }

    /// The digit of `exponent` at this position.
    #[inline(always)]
    const fn read(self, exponent: &[u64; L]) -> usize {
        let mut digit = exponent[self.limb] >> self.shift;
        if self.straddles {
            digit |= exponent[self.limb + 1] << (64 - self.shift);
        }
        (digit & self.mask) as usize
    }
}
