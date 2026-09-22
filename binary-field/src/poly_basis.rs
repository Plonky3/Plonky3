//! Raw polynomial coordinates for `GF(2^128)` modulo `x^128 + x^7 + x^2 + x + 1`.
//!
//! Prefer the typed field for ordinary arithmetic.
//! These helpers are for buffers whose representation the caller manages.
//!
//! The coordinates stay a bare integer on purpose.
//! Wrapped, the two bases look alike, and a value in the wrong one is silently wrong.

use alloc::vec::Vec;

use crate::tower::TowerLevel;
use crate::{BinaryField128, clmul, poly_slice};

/// Whether multiplication and squaring use hardware carryless multiplication.
///
/// Otherwise multiplication uses masked integer products and squaring uses bit spreading.
pub const HAS_HARDWARE_CLMUL: bool = clmul::HAS_HARDWARE_CLMUL;

/// The polynomial-basis coordinates of a tower element.
#[must_use]
#[inline]
pub fn from_tower(x: BinaryField128) -> u128 {
    clmul::tower_to_poly_128(x.to_repr())
}

/// The tower element with the given polynomial-basis coordinates.
#[inline]
pub fn to_tower(v: u128) -> BinaryField128 {
    BinaryField128::from_repr(clmul::poly_to_tower_128(v))
}

/// The polynomial-basis coordinates of a whole slice of tower-basis bit patterns.
///
/// # Performance
///
/// Whole blocks of 64 elements cross over together on an `x86_64` build with `gfni`, `avx512f`
/// and `avx512bw`, several times faster than the sixteen table lookups per element that the
/// remainder falls back to. Every other build takes the per-element path throughout.
///
/// The choice is the build's own target features, made at compile time; nothing here detects
/// the running processor. `-C target-cpu=x86-64-v4` gives `avx512bw` but not `gfni`, so a v4
/// build is one of the per-element ones.
///
/// Naming the three features and nothing else does reach the kernel, but leaves the compiler
/// no scheduling model to hold a block in registers with. On Zen 5 it then spills every block
/// through memory, which costs about a third of the throughput a `-C target-cpu` build gets.
#[inline]
pub fn from_tower_slice(values: &mut [u128]) {
    // The count of elements the blocked kernel took is of interest only to its own tests.
    clmul::tower_to_poly_128_slice(values);
}

/// The tower-basis bit patterns of a whole slice of polynomial-basis coordinates.
///
/// This undoes [`from_tower_slice`], and blocks the same way under the same target features.
#[inline]
pub fn to_tower_slice(values: &mut [u128]) {
    clmul::poly_to_tower_128_slice(values);
}

/// Multiply two elements expressed in polynomial coordinates.
// Only the software backend is `const`, so leaving this one out keeps the signature the same
// on every target.
#[allow(clippy::missing_const_for_fn)]
#[must_use]
#[inline]
pub fn mul(a: u128, b: u128) -> u128 {
    clmul::poly_mul_128(a, b)
}

/// The square of an element, given and returned in the polynomial basis.
#[allow(clippy::missing_const_for_fn)]
#[must_use]
#[inline]
pub fn square(a: u128) -> u128 {
    clmul::poly_square_128(a)
}

/// Multiply every polynomial-basis element by the same scalar.
#[inline]
pub fn mul_slice(values: &mut [u128], scalar: u128) {
    // A lone element has no loop to amortize any setup over, so it takes the plain product.
    if let [value] = values {
        *value = mul(*value, scalar);
        return;
    }
    poly_slice::scale(values, scalar);
}

/// Apply `(lo, hi) -> (lo + scalar*hi, lo + (scalar + 1)*hi)` in place.
///
/// # Panics
/// Panics if the slice lengths differ.
#[inline]
pub fn butterfly_forward(lo: &mut [u128], hi: &mut [u128], scalar: u128) {
    assert_eq!(lo.len(), hi.len(), "butterfly lengths differ");
    // A zero scalar kills the product, leaving an addition that needs no multiply at all.
    if scalar == 0 {
        for (lo, hi) in lo.iter().zip(hi) {
            *hi ^= *lo;
        }
        return;
    }
    // A lone pair has no loop to amortize any setup over, so it takes the plain product.
    if lo.len() == 1 {
        lo[0] ^= mul(scalar, hi[0]);
        hi[0] ^= lo[0];
        return;
    }
    poly_slice::butterfly_forward(lo, hi, scalar);
}

/// Undo [`butterfly_forward`] with the same scalar.
///
/// # Panics
/// Panics if the slice lengths differ.
#[inline]
pub fn butterfly_inverse(lo: &mut [u128], hi: &mut [u128], scalar: u128) {
    assert_eq!(lo.len(), hi.len(), "butterfly lengths differ");
    // A zero scalar kills the product, leaving an addition that needs no multiply at all.
    if scalar == 0 {
        for (lo, hi) in lo.iter().zip(hi) {
            *hi ^= *lo;
        }
        return;
    }
    // A lone pair has no loop to amortize any setup over, so it takes the plain product.
    if lo.len() == 1 {
        hi[0] ^= lo[0];
        lo[0] ^= mul(scalar, hi[0]);
        return;
    }
    poly_slice::butterfly_inverse(lo, hi, scalar);
}

/// Stages [`LowStageTwiddles::forward`] runs together, over runs of `2^LOW_STAGES` elements.
pub const LOW_STAGES: usize = poly_slice::LOW_STAGES;

/// A value beside its product with `x^64`, the companion the packed multiply scales by.
///
/// The product is linear in the value, so a sum of pairs is the pair of the sum.
type Pair = [u128; 2];

/// The sum of two values, each beside its companion.
#[inline(always)]
const fn add(a: Pair, b: Pair) -> Pair {
    [a[0] ^ b[0], a[1] ^ b[1]]
}

/// The twiddles of the lowest [`LOW_STAGES`] stages of a single-column additive transform.
///
/// Stage `j` pairs elements `2^j` apart, and block `b` of that stage, its `2^(j+1)` elements
/// from `2^(j+1) b` on, scales by one twiddle, affine in `b`:
///
/// ```text
///     twiddle(j, b) = shifts[j] + Σ basis[k]    over the set bits k of b
/// ```
///
/// A run of `2^LOW_STAGES` elements is closed under all of those stages, so they run together
/// while the run sits in registers, instead of in one sweep of the slice each.
#[derive(Clone, Debug)]
pub struct LowStageTwiddles {
    /// `shifts[j]` for each low stage.
    shifts: [Pair; LOW_STAGES],
    /// `prefix[n]` is the sum of the first `n` basis elements.
    prefix: Vec<Pair>,
    /// `steps[t][j]` is what stage `j`'s twiddle moves by past a run index with `t` trailing
    /// ones, for every `t` a run inside the basis can have.
    steps: Vec<[Pair; LOW_STAGES]>,
}

impl LowStageTwiddles {
    /// The twiddles of a transform with these stage shifts and this block basis.
    ///
    /// # Panics
    /// Panics if `shifts` holds fewer than [`LOW_STAGES`] stages.
    pub fn new(shifts: &[u128], basis: &[u128]) -> Self {
        let pair = |value: u128| [value, clmul::poly_mul_128(value, 1 << 64)];
        let prefix: Vec<Pair> = core::iter::once(0)
            .chain(basis.iter().scan(0, |sum, &element| {
                *sum ^= element;
                Some(*sum)
            }))
            .map(pair)
            .collect();
        // Stepping a run index with `t` trailing ones flips its `t + 1` lowest bits, which stage
        // `j` reads from `basis[LOW_STAGES - 1 - j]` on. Stage 0 reads the highest, so it is
        // what bounds `t`.
        let steps = (0..(basis.len() + 1).saturating_sub(LOW_STAGES))
            .map(|trailing| {
                core::array::from_fn(|j| {
                    let from = LOW_STAGES - 1 - j;
                    add(prefix[from + trailing + 1], prefix[from])
                })
            })
            .collect();
        Self {
            shifts: core::array::from_fn(|j| pair(shifts[j])),
            prefix,
            steps,
        }
    }

    /// Runs the low stages of the forward transform over `values`.
    ///
    /// `values` starts at element `2^LOW_STAGES · first_run` of the transform, and ends on a
    /// whole run of `2^LOW_STAGES` elements.
    ///
    /// # Panics
    /// Panics if `values` ends inside a run, or if a run lies past the basis the twiddles hold.
    pub fn forward(&self, values: &mut [u128], first_run: usize) {
        assert_eq!(
            values.len() % (1 << LOW_STAGES),
            0,
            "the low stages run over whole runs"
        );
        poly_slice::forward_low_stages(values, first_run, self);
    }

    /// The sum of the basis elements the set bits of `bits` select, counted from `basis[from]`.
    #[inline(always)]
    pub(crate) fn span(&self, mut bits: usize, from: usize) -> Pair {
        let mut sum = [0; 2];
        while bits != 0 {
            let k = from + bits.trailing_zeros() as usize;
            sum = add(sum, add(self.prefix[k + 1], self.prefix[k]));
            bits &= bits - 1;
        }
        sum
    }

    /// Each low stage's twiddle for the first pair of run `run`.
    ///
    /// Stage `j` sees the run as `2^(LOW_STAGES-1-j)` of its blocks, so the run index reaches
    /// its twiddle shifted up by that many basis elements.
    #[inline(always)]
    pub(crate) fn run_twiddles(&self, run: usize) -> [Pair; LOW_STAGES] {
        core::array::from_fn(|j| add(self.shifts[j], self.span(run, LOW_STAGES - 1 - j)))
    }

    /// What each low stage's twiddle moves by from run `run` to the run after it.
    ///
    /// Stepping `run` flips its trailing ones and the zero above them.
    /// So each stage's twiddle moves by the sum of that many consecutive basis elements.
    #[inline(always)]
    pub(crate) fn step(&self, run: usize) -> &[Pair; LOW_STAGES] {
        &self.steps[run.trailing_ones() as usize]
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_field::PrimeCharacteristicRing;
    use proptest::prelude::*;

    use super::{LOW_STAGES, LowStageTwiddles, from_tower, mul, square, to_tower};
    use crate::BinaryField128;

    /// Elements in one run of the low stages.
    const LOW_RUN: usize = 1 << LOW_STAGES;

    /// Basis elements the low-stage tests draw, enough for every run index they reach.
    const LOW_BASIS: usize = 40;

    /// Building an element from a 128-bit pattern, which every pattern is a valid one of.
    fn element(bits: u128) -> BinaryField128 {
        BinaryField128::from_le_bytes(bits.to_le_bytes())
    }

    /// The low stages one stage at a time over the whole slice, each pair scaled by the twiddle
    /// its block index selects, summed straight from the basis.
    fn reference_low_stages(
        values: &mut [u128],
        first_run: usize,
        shifts: &[u128],
        basis: &[u128],
    ) {
        let first = first_run << LOW_STAGES;
        for j in (0..LOW_STAGES).rev() {
            let half = 1 << j;
            for (index, pair) in values.chunks_exact_mut(2 * half).enumerate() {
                let block = (first >> (j + 1)) + index;
                let twiddle = (0..basis.len())
                    .filter(|k| (block >> k) & 1 == 1)
                    .fold(shifts[j], |twiddle, k| twiddle ^ basis[k]);
                let (lo, hi) = pair.split_at_mut(half);
                for (lo, hi) in lo.iter_mut().zip(hi) {
                    *lo ^= mul(twiddle, *hi);
                    *hi ^= *lo;
                }
            }
        }
    }

    /// Run indices whose successors carry across many bits, beside small and random ones.
    fn run_index() -> impl Strategy<Value = usize> {
        prop_oneof![
            0usize..64,
            Just((1 << 20) - 1),
            Just((1 << 24) - 2),
            0usize..1 << 24,
        ]
    }

    proptest! {
        /// Invariant: running the low stages together within each run reorders nothing that
        /// depends on order, so every element matches the stage-at-a-time schedule.
        #[test]
        fn the_low_stages_match_one_stage_at_a_time(
            shifts in any::<[u128; LOW_STAGES]>(),
            basis in prop::collection::vec(any::<u128>(), LOW_BASIS),
            values in prop::collection::vec(any::<u128>(), 5 * LOW_RUN),
            runs in 0usize..=5,
            first_run in run_index(),
        ) {
            let mut actual: Vec<u128> = values[..runs * LOW_RUN].to_vec();
            let mut expected = actual.clone();
            reference_low_stages(&mut expected, first_run, &shifts, &basis);
            LowStageTwiddles::new(&shifts, &basis).forward(&mut actual, first_run);
            prop_assert_eq!(actual, expected);
        }

        #[test]
        fn the_change_of_basis_round_trips(bits in any::<u128>()) {
            let x = element(bits);
            prop_assert_eq!(to_tower(from_tower(x)), x);
        }

        /// Addition is `XOR` in both bases, so the conversion commutes with it.
        #[test]
        fn the_change_of_basis_is_additive(a in any::<u128>(), b in any::<u128>()) {
            let (x, y) = (element(a), element(b));
            prop_assert_eq!(from_tower(x + y), from_tower(x) ^ from_tower(y));
        }

        /// The polynomial-basis product is the tower product, seen in the other basis.
        #[test]
        fn the_product_agrees_with_the_tower_product(a in any::<u128>(), b in any::<u128>()) {
            let (x, y) = (element(a), element(b));
            prop_assert_eq!(to_tower(mul(from_tower(x), from_tower(y))), x * y);
        }

        #[test]
        fn the_square_agrees_with_the_tower_square(a in any::<u128>()) {
            let x = element(a);
            prop_assert_eq!(to_tower(square(from_tower(x))), x.square());
        }
    }

    #[test]
    fn slice_operations_match_reference_tower_arithmetic() {
        use alloc::vec::Vec;
        // A packed kernel covers whole registers of two or four elements.
        //
        // The lengths below therefore hit every prefix and tail combination.
        for len in [0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 64, 257] {
            // A zero scalar takes the addition-only shortcut and one is the identity.
            for scalar in [
                0,
                1,
                0x87,
                1 << 127,
                0xfeed_9876_0123_4567_89ab_cdef_9876_5432,
            ] {
                let t = element(scalar);
                let xs: Vec<_> = (0..len)
                    .map(|i| {
                        element(
                            (i as u128 + 1).wrapping_mul(0xfeed_8765_dead_beef_cafe_0123_4567_89ab),
                        )
                    })
                    .collect();
                let ys: Vec<_> = xs.iter().rev().copied().collect();
                let mut lo: Vec<_> = xs.iter().copied().map(from_tower).collect();
                let mut hi: Vec<_> = ys.iter().copied().map(from_tower).collect();
                let original = (lo.clone(), hi.clone());
                super::butterfly_forward(&mut lo, &mut hi, from_tower(t));
                for i in 0..len {
                    let expected = xs[i] + ys[i].reference_mul(t);
                    assert_eq!(to_tower(lo[i]), expected);
                    assert_eq!(to_tower(hi[i]), expected + ys[i]);
                }
                super::butterfly_inverse(&mut lo, &mut hi, from_tower(t));
                assert_eq!((&lo, &hi), (&original.0, &original.1));
                super::mul_slice(&mut lo, from_tower(t));
                for i in 0..len {
                    assert_eq!(to_tower(lo[i]), xs[i].reference_mul(t));
                }
            }
        }
    }

    #[test]
    #[should_panic = "the low stages run over whole runs"]
    fn the_low_stages_reject_a_partial_run() {
        let twiddles = LowStageTwiddles::new(&[1; LOW_STAGES], &[1; LOW_BASIS]);
        twiddles.forward(&mut [0; LOW_RUN + 1], 0);
    }

    #[test]
    #[should_panic = "butterfly lengths differ"]
    fn forward_rejects_mismatched_lengths() {
        super::butterfly_forward(&mut [0], &mut [], 0);
    }

    #[test]
    #[should_panic = "butterfly lengths differ"]
    fn inverse_rejects_mismatched_lengths() {
        super::butterfly_inverse(&mut [], &mut [0], 1);
    }

    /// The basis change fixes zero and one, as any field isomorphism must.
    #[test]
    fn the_change_of_basis_fixes_the_constants() {
        assert_eq!(from_tower(BinaryField128::ZERO), 0);
        assert_eq!(from_tower(BinaryField128::ONE), 1);
        assert_eq!(to_tower(0), BinaryField128::ZERO);
        assert_eq!(to_tower(1), BinaryField128::ONE);
    }
}
