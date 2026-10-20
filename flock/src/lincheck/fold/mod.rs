//! The witness folded over its instances: `out[i] = sum_o eq_outer[o] * z_o[i]`.
//!
//! The bits of instances `8s .. 8s + 8` at position `i` form the stripe byte `(s, i)`.
//!
//! The fold is GF(2)-linear in each stripe byte, so every kernel maps stripe bytes through a 256-entry sum table or a GFNI matrix.

#[cfg(any(
    test,
    all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi",
        target_feature = "gfni",
        target_feature = "vpclmulqdq"
    )
))]
mod gfni;
#[cfg_attr(
    all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi",
        target_feature = "gfni",
        target_feature = "vpclmulqdq"
    ),
    allow(dead_code)
)]
mod portable;

use alloc::vec;
use alloc::vec::Vec;

use p3_binary_field::PackedGf2x64;
use p3_field::PrimeCharacteristicRing;

use crate::Challenge;
use crate::byteslice::transpose_bits;
use crate::skip::K_SKIP;

/// The instance-major witness, read as stripes of eight instances.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Stripes<'a> {
    /// The witness, `words` packed words per instance.
    z: &'a [u64],
    /// Packed words per instance.
    words: usize,
}

impl<'a> Stripes<'a> {
    /// The stripes of `z`, whose instances are `2^k_log` bits each.
    ///
    /// # Panics
    ///
    /// - An instance is not whole words, or the instances do not tile whole stripes.
    pub(crate) fn new(z: &'a [u64], k_log: usize) -> Self {
        assert!(k_log >= K_SKIP, "an instance is whole words");
        let words = 1 << (k_log - K_SKIP);
        assert!(
            z.len().is_multiple_of(8 * words),
            "the instances tile whole stripes of eight"
        );
        Self { z, words }
    }

    /// Stripes, one per eight instances.
    pub(crate) const fn len(&self) -> usize {
        self.z.len() / (8 * self.words)
    }

    /// Positions per instance.
    pub(crate) const fn positions(&self) -> usize {
        64 * self.words
    }

    /// Stripe `s`'s bytes at positions `64g ..`: byte `i`, bit `r` is bit `64g + i` of instance `8s + r`.
    #[inline(always)]
    pub(crate) fn bytes(&self, s: usize, g: usize) -> [u8; 64] {
        let first = 8 * s * self.words + g;
        let rows: [u64; 8] = core::array::from_fn(|r| self.z[first + r * self.words]);
        let bytes = PackedGf2x64::as_bytes(PackedGf2x64::from_bits_slice(&rows));
        transpose_bits(&bytes.as_chunks::<64>().0[0])
    }
}

/// The eq-weighted sum over instances of every position of the witness.
///
/// Positions from `useful_bits` on are zero in every instance, so they fold to zero unread.
///
/// # Panics
///
/// - There is not one weight per instance.
pub(crate) fn partial_fold(
    z: &[u64],
    k_log: usize,
    useful_bits: usize,
    eq_outer: &[Challenge],
) -> Vec<Challenge> {
    assert_eq!(
        z.len() << K_SKIP,
        eq_outer.len() << k_log,
        "one weight per instance"
    );
    assert!(useful_bits <= 1 << k_log, "useful bits fit an instance");
    // Fewer than eight instances fill no stripe.
    if eq_outer.len() < 8 {
        return fold_bits(z, k_log, eq_outer);
    }
    let stripes = Stripes::new(z, k_log);
    // Groups of 64 positions holding a useful bit; a straddling group folds its zeros harmlessly.
    let groups = useful_bits.div_ceil(64);
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi",
        target_feature = "gfni",
        target_feature = "vpclmulqdq"
    ))]
    {
        use crate::byteslice::Native;
        if gfni::fits(&stripes, gfni::WIDE_TILE) {
            return gfni::partial_fold::<Native, { gfni::WIDE_TILE }>(stripes, groups, eq_outer);
        }
        if gfni::fits(&stripes, gfni::NARROW_TILE) {
            return gfni::partial_fold::<Native, { gfni::NARROW_TILE }>(stripes, groups, eq_outer);
        }
    }
    portable::fold_tables(stripes, groups, eq_outer)
}

/// The definition, bit by bit.
fn fold_bits(z: &[u64], k_log: usize, eq_outer: &[Challenge]) -> Vec<Challenge> {
    let mut out = vec![Challenge::ZERO; 1 << k_log];
    for (o, &eq) in eq_outer.iter().enumerate() {
        for (i, out) in out.iter_mut().enumerate() {
            let bit = (o << k_log) + i;
            if (z[bit / 64] >> (bit % 64)) & 1 == 1 {
                *out += eq;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{Rng, RngExt, SeedableRng};

    use super::*;
    use crate::byteslice::{Model, SliceRegister, transpose_rows};

    /// A random witness of `2^n_log` instances, zero from `useful` on in each.
    fn witness(rng: &mut SmallRng, k_log: usize, n_log: usize, useful: usize) -> Vec<u64> {
        let words = 1usize << (k_log - 6);
        let mut z: Vec<u64> = (0..words << n_log).map(|_| rng.next_u64()).collect();
        for instance in z.chunks_exact_mut(words) {
            for (w, word) in instance.iter_mut().enumerate() {
                let keep = useful.saturating_sub(64 * w).min(64);
                *word &= u64::MAX.checked_shr(64 - keep as u32).unwrap_or(0);
            }
        }
        z
    }

    #[test]
    fn every_kernel_is_the_definition() {
        // Fixture state, (k_log, n_log, useful): no stripe, a few stripes, narrow and wide tiles, padded instances.
        let cases = [
            (6, 0, 64),
            (6, 2, 40),
            (6, 3, 64),
            (7, 5, 100),
            (6, 6, 64),
            (8, 7, 256),
            (9, 6, 300),
            (10, 8, 597),
            (14, 6, 16_000),
            (12, 9, 2_336),
        ];
        let mut rng = SmallRng::seed_from_u64(0xF01D);
        for (k_log, n_log, useful) in cases {
            let z = witness(&mut rng, k_log, n_log, useful);
            let eq: Vec<Challenge> = (0..1 << n_log).map(|_| rng.random()).collect();
            let want = fold_bits(&z, k_log, &eq);
            assert_eq!(
                partial_fold(&z, k_log, useful, &eq),
                want,
                "dispatch {k_log} {n_log} {useful}"
            );
            if n_log < 3 {
                continue;
            }
            let stripes = Stripes::new(&z, k_log);
            let groups = useful.div_ceil(64);
            assert_eq!(
                portable::fold_tables(stripes, groups, &eq),
                want,
                "tables {k_log} {n_log} {useful}"
            );
            if gfni::fits(&stripes, gfni::NARROW_TILE) {
                let got = gfni::partial_fold::<Model, { gfni::NARROW_TILE }>(stripes, groups, &eq);
                assert_eq!(got, want, "narrow {k_log} {n_log} {useful}");
            }
            if gfni::fits(&stripes, gfni::WIDE_TILE) {
                let got = gfni::partial_fold::<Model, { gfni::WIDE_TILE }>(stripes, groups, &eq);
                assert_eq!(got, want, "wide {k_log} {n_log} {useful}");
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(16))]

        #[test]
        fn stripe_bytes_are_eight_instances_bits(seed in any::<u64>()) {
            // Invariant: stripe byte (s, i), bit r is bit i of instance 8s + r.
            let mut rng = SmallRng::seed_from_u64(seed);
            let (k_log, n_log) = (7, 4);
            let z = witness(&mut rng, k_log, n_log, 1 << k_log);
            let stripes = Stripes::new(&z, k_log);
            for s in 0..stripes.len() {
                for g in 0..2 {
                    for (i, byte) in stripes.bytes(s, g).into_iter().enumerate() {
                        for r in 0..8 {
                            let bit = ((8 * s + r) << k_log) + 64 * g + i;
                            prop_assert_eq!(u64::from(byte >> r & 1), (z[bit / 64] >> (bit % 64)) & 1);
                        }
                    }
                }
            }
        }

        #[test]
        fn the_dispatched_fold_is_the_definition(seed in any::<u64>(), k_log in 6usize..11, n_log in 0usize..9) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let useful = rng.random_range(1..=1usize << k_log);
            let z = witness(&mut rng, k_log, n_log, useful);
            let eq: Vec<Challenge> = (0..1 << n_log).map(|_| rng.random()).collect();
            prop_assert_eq!(partial_fold(&z, k_log, useful, &eq), fold_bits(&z, k_log, &eq));
        }

        #[test]
        fn the_model_transposes_are_the_stripe_bytes(seed in any::<u64>()) {
            // Eight instances of eight words: register r is instance r's line.
            let mut rng = SmallRng::seed_from_u64(seed);
            let z: Vec<u64> = (0..64).map(|_| rng.next_u64()).collect();
            let stripes = Stripes::new(&z, 9);
            // SAFETY: each line is eight words of the witness.
            let lines: [Model; 8] = core::array::from_fn(|r| unsafe { Model::load(z[8 * r..].as_ptr().cast()) });
            for (g, register) in Model::transpose_qwords(lines).map(transpose_rows).into_iter().enumerate() {
                let mut bytes = [0u8; 64];
                // SAFETY: the array is 64 bytes.
                unsafe { Model::store(bytes.as_mut_ptr(), register) };
                prop_assert_eq!(bytes, stripes.bytes(0, g));
            }
        }
    }
}
