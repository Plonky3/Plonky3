//! Word-level witness generation: a gadget computes its hash on words and appends each product's row in slot order.
//!
//! Every sum here commits exactly the rows of `Builder::wrapping_sum`, in its order.

/// Bits 0 to 30 of a 32-bit word: the columns that carry.
const LOW31: u32 = 0x7FFF_FFFF;

/// One instance's `A z` and `B z`, written forward from a word.
pub(crate) struct Rows<'a> {
    /// `A z`.
    az: &'a mut [u64],
    /// `B z`.
    bz: &'a mut [u64],
    /// The word the pending bits belong to.
    word: usize,
    /// Pending bits.
    bit: u32,
    /// Pending bits of `A z`.
    a: u64,
    /// Pending bits of `B z`.
    b: u64,
}

impl<'a> Rows<'a> {
    /// A writer whose first row lands at bit 0 of `word`.
    pub(crate) const fn new(az: &'a mut [u64], bz: &'a mut [u64], word: usize) -> Self {
        Self {
            az,
            bz,
            word,
            bit: 0,
            a: 0,
            b: 0,
        }
    }

    /// Append `n` rows, at most 64: bit `i` of `a` and `b` is row `i`'s left and right factor.
    #[inline(always)]
    pub(crate) fn push(&mut self, a: u64, b: u64, n: u32) {
        debug_assert!(n <= 64 && (n == 64 || (a | b) >> n == 0), "{n} rows");
        self.a |= a << self.bit;
        self.b |= b << self.bit;
        let end = self.bit + n;
        if end >= 64 {
            self.az[self.word] = self.a;
            self.bz[self.word] = self.b;
            self.word += 1;
            // The bits that did not fit, none when the word was empty.
            self.a = (a >> 1) >> (63 - self.bit);
            self.b = (b >> 1) >> (63 - self.bit);
            self.bit = end - 64;
        } else {
            self.bit = end;
        }
    }

    /// Write a whole word out of order: `A z` is the value on the port's bits, `B z` their mask.
    #[inline]
    pub(crate) const fn port(&mut self, word: usize, value: u64, mask: u64) {
        self.az[word] = value & mask;
        self.bz[word] = mask;
    }

    /// Write the pending word and zero the rest of both tables.
    pub(crate) fn finish(self) {
        let mut word = self.word;
        if self.bit > 0 {
            self.az[word] = self.a;
            self.bz[word] = self.b;
            word += 1;
        }
        self.az[word..].fill(0);
        self.bz[word..].fill(0);
    }

    /// `x + y` of two variable words: 31 rows, one per column but the top.
    ///
    /// Column 0 is a half adder on `(x, y)`; a later one a full adder on `(carry, x, y)`.
    #[inline(always)]
    pub(crate) fn add2(&mut self, x: u32, y: u32) -> u32 {
        let sum = x.wrapping_add(y);
        let carry = sum ^ x ^ y;
        let a = ((carry ^ y) & !1) | (x & 1);
        let b = ((x ^ y) & !1) | (y & 1);
        self.push(u64::from(a & LOW31), u64::from(b & LOW31), 31);
        sum
    }

    /// `k + x` of a constant and a variable word: one row per column past `k`'s lowest one.
    ///
    /// The lowest one makes `x` the carry for free; each later column's row is `carry * x`.
    #[inline(always)]
    pub(crate) fn add_const(&mut self, k: u32, x: u32) -> u32 {
        let sum = k.wrapping_add(x);
        let from = k.trailing_zeros();
        // No row when no column below the top receives a carry.
        if from < 30 {
            let carry = sum ^ k ^ x;
            let skip = from + 1;
            self.push(
                u64::from((carry & LOW31) >> skip),
                u64::from((x & LOW31) >> skip),
                30 - from,
            );
        }
        sum
    }

    /// `x + y + z` of three variable words, fused: 61 rows.
    ///
    /// Each column but the top makes a majority row; columns 1 to 30 then make a ripple row after it.
    #[inline(always)]
    pub(crate) fn add3(&mut self, x: u32, y: u32, z: u32) -> u32 {
        // Majority rows: the full adder on `(x, y, z)`.
        let (ma, mb) = (x ^ z, y ^ z);
        let parity = x ^ y ^ z;
        let twice = ((ma & mb) ^ z) << 1;
        let sum = parity.wrapping_add(twice);
        let carry = sum ^ parity ^ twice;

        // Ripple rows: a half adder on column 1, then the full adder on `(majority, carry, parity)`.
        let ra = ((twice ^ parity) & !2) | (twice & 2);
        let rb = carry ^ parity;

        // Column 0's majority, then column `j`'s majority and ripple at slots `2j - 1` and `2j`.
        let lay = |m: u32, r: u32| {
            u64::from(m & 1) | (spread((m & LOW31) >> 1) << 1) | (spread((r & LOW31) >> 1) << 2)
        };
        self.push(lay(ma, ra), lay(mb, rb), 61);
        sum
    }
}

/// Bit `i` of `x` moved to bit `2i`.
#[inline(always)]
pub(crate) fn spread(x: u32) -> u64 {
    #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
    {
        // SAFETY: the target feature is enabled at compile time.
        unsafe { core::arch::x86_64::_pdep_u64(u64::from(x), 0x5555_5555_5555_5555) }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
    {
        let mut x = u64::from(x);
        x = (x | (x << 16)) & 0x0000_FFFF_0000_FFFF;
        x = (x | (x << 8)) & 0x00FF_00FF_00FF_00FF;
        x = (x | (x << 4)) & 0x0F0F_0F0F_0F0F_0F0F;
        x = (x | (x << 2)) & 0x3333_3333_3333_3333;
        (x | (x << 1)) & 0x5555_5555_5555_5555
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    proptest! {
        #[test]
        fn spread_moves_every_bit_to_twice_its_index(x: u32) {
            // Invariant: bit i of x is bit 2i of the spread, odd bits clear.
            let s = spread(x);
            for i in 0..32 {
                prop_assert_eq!((s >> (2 * i)) & 1, u64::from((x >> i) & 1));
            }
            prop_assert_eq!(s & 0xAAAA_AAAA_AAAA_AAAA, 0);
        }

        #[test]
        fn pushed_rows_land_in_order(runs in prop::collection::vec((any::<u64>(), any::<u64>(), 1u32..=64), 0..40)) {
            // Invariant: the tables hold the runs' bits back to back from the first word, zero after.
            let total: u32 = runs.iter().map(|r| r.2).sum();
            let words = 2 + total.div_ceil(64) as usize;
            let (mut az, mut bz) = (alloc::vec![u64::MAX; words + 1], alloc::vec![u64::MAX; words + 1]);
            let mask = |v: u64, n: u32| if n == 64 { v } else { v & ((1 << n) - 1) };
            let mut rows = Rows::new(&mut az, &mut bz, 1);
            for &(a, b, n) in &runs {
                rows.push(mask(a, n), mask(b, n), n);
            }
            rows.finish();
            let bit = |t: &[u64], i: usize| (t[1 + i / 64] >> (i % 64)) & 1;
            let mut at = 0;
            for &(a, b, n) in &runs {
                for i in 0..n as usize {
                    prop_assert_eq!(bit(&az, at + i), (a >> i) & 1);
                    prop_assert_eq!(bit(&bz, at + i), (b >> i) & 1);
                }
                at += n as usize;
            }
            for i in at..64 * words {
                prop_assert_eq!(bit(&az, i) | bit(&bz, i), 0);
            }
            prop_assert_eq!((az[0], bz[0]), (u64::MAX, u64::MAX));
        }
    }
}
