//! Round 2 integer folds `mu_c = sum_b gamma_b bit(c, b)` and their wire encoding (Construction 4.4, Round 2).

mod tables;

use alloc::vec;
use alloc::vec::Vec;

use p3_field::PrimeField64;
pub(crate) use tables::fold;

use crate::error::{ParamsError, VerificationError};
use crate::params::BitShape;

/// The range of a fold, `[0, (p - 1) 2^r1]`, and how it is encoded and accumulated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FoldBound {
    /// `(p - 1) 2^r1`.
    pub(crate) max: u128,
    /// Bits of one encoded fold, the bit length of the bound.
    pub(crate) width: usize,
    /// 32-bit pieces of a lifted row weight.
    pieces: usize,
}

impl FoldBound {
    /// The bound of a shape over the field `F`.
    ///
    /// # Errors
    ///
    /// The bound reaches `ord(g) = 2^192 - 1`, so a fold could wrap in the exponent (Section 4.4).
    pub(crate) fn new<F: PrimeField64>(shape: BitShape) -> Result<Self, ParamsError> {
        let top = F::ORDER_U64 - 1;
        // Any bound below 2^128 is below the group order; a row count below 2^32 keeps it below 2^96.
        let max = u128::from(top)
            .checked_mul(1 << shape.log_rows())
            .ok_or_else(|| ParamsError::FoldOverflow {
                modulus_bits: 64 - F::ORDER_U64.leading_zeros(),
                log_rows: shape.log_rows(),
            })?;
        Ok(Self {
            max,
            width: (128 - max.leading_zeros()) as usize,
            pieces: if top >> 32 == 0 { 1 } else { 2 },
        })
    }

    /// 32-bit pieces of a lifted row weight.
    pub(crate) const fn pieces(&self) -> usize {
        self.pieces
    }

    /// Words encoding `count` folds.
    pub(crate) const fn words(&self, count: usize) -> usize {
        (count * self.width).div_ceil(64)
    }

    /// Pack the folds little-endian, `width` bits each.
    pub(crate) fn encode(&self, folds: &[u128]) -> Vec<u64> {
        let mut out = vec![0u64; self.words(folds.len())];
        for (index, &fold) in folds.iter().enumerate() {
            debug_assert!(fold <= self.max, "a fold within its range");
            let at = index * self.width;
            let (word, offset) = (at / 64, at % 64);
            let low = fold << offset;
            out[word] |= low as u64;
            if offset + self.width > 64 {
                out[word + 1] |= (low >> 64) as u64;
            }
            if offset + self.width > 128 {
                out[word + 2] |= (fold >> (128 - offset)) as u64;
            }
        }
        out
    }

    /// Unpack `count` folds, refusing a wrong length, set padding bits, or a fold past the bound.
    pub(crate) fn decode(
        &self,
        words: &[u64],
        count: usize,
    ) -> Result<Vec<u128>, VerificationError> {
        let expected = self.words(count);
        if words.len() != expected {
            return Err(VerificationError::FoldLength {
                expected,
                actual: words.len(),
            });
        }
        let used = count * self.width;
        if !used.is_multiple_of(64) && words[expected - 1] >> (used % 64) != 0 {
            return Err(VerificationError::FoldPadding);
        }
        let mask = (1u128 << self.width) - 1;
        (0..count)
            .map(|index| {
                let at = index * self.width;
                let (word, offset) = (at / 64, at % 64);
                let next = |k: usize| u128::from(words.get(word + k).copied().unwrap_or(0));
                let mut window = next(0) >> offset | next(1) << (64 - offset);
                if offset + self.width > 128 {
                    window |= next(2) << (128 - offset);
                }
                let fold = window & mask;
                if fold > self.max {
                    Err(VerificationError::FoldRange { index })
                } else {
                    Ok(fold)
                }
            })
            .collect()
    }
}

/// `pi(mu)`, a fold reduced into the field.
pub(crate) fn project<F: PrimeField64>(fold: u128) -> F {
    F::from_u64((fold % u128::from(F::ORDER_U64)) as u64)
}

#[cfg(test)]
mod tests {
    use p3_field::PrimeCharacteristicRing;
    use p3_goldilocks::Goldilocks;
    use p3_koala_bear::KoalaBear;
    use proptest::prelude::*;

    use super::*;

    proptest! {
        #[test]
        fn folds_round_trip_through_their_encoding(
            log_rows in 6usize..=31,
            seeds in prop::collection::vec(any::<u128>(), 1..40),
        ) {
            for bound in [
                FoldBound::new::<Goldilocks>(BitShape::new(log_rows, 3).unwrap()).unwrap(),
                FoldBound::new::<KoalaBear>(BitShape::new(log_rows, 3).unwrap()).unwrap(),
            ] {
                let folds: Vec<u128> = seeds.iter().map(|s| s % (bound.max + 1)).collect();
                let words = bound.encode(&folds);
                prop_assert_eq!(words.len(), (folds.len() * bound.width).div_ceil(64));
                prop_assert_eq!(bound.decode(&words, folds.len()).unwrap(), folds);
            }
        }
    }

    #[test]
    fn decoding_refuses_padding_range_and_length() {
        // KoalaBear over 2^6 rows: 37-bit folds.
        let bound = FoldBound::new::<KoalaBear>(BitShape::new(6, 1).unwrap()).unwrap();
        assert_eq!((bound.width, bound.pieces), (37, 1));
        let folds = [bound.max, 1];
        let words = bound.encode(&folds);
        assert_eq!(bound.decode(&words, 2).unwrap(), folds);

        // A set padding bit.
        let mut padded = words.clone();
        *padded.last_mut().unwrap() |= 1 << 63;
        assert!(matches!(
            bound.decode(&padded, 2),
            Err(VerificationError::FoldPadding)
        ));

        // All 37 bits set is above (p - 1) 2^6.
        let mut over = words.clone();
        over[0] |= (1 << 37) - 1;
        assert!(matches!(
            bound.decode(&over, 2),
            Err(VerificationError::FoldRange { index: 0 })
        ));

        // A missing word.
        assert!(matches!(
            bound.decode(&words[..1], 2),
            Err(VerificationError::FoldLength {
                expected: 2,
                actual: 1
            })
        ));
    }

    #[test]
    fn a_64_bit_field_splits_its_weights_in_two_pieces() {
        let bound = FoldBound::new::<Goldilocks>(BitShape::new(31, 0).unwrap()).unwrap();
        assert_eq!((bound.width, bound.pieces), (95, 2));
        assert_eq!(
            project::<Goldilocks>(bound.max),
            Goldilocks::NEG_ONE * Goldilocks::from_u64(1 << 31)
        );
    }
}
