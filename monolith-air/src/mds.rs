//! Checked circulant dispatch shared by the witness generator and AIR.

use p3_field::{Algebra, PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;
use p3_mds::karatsuba_convolution::{
    mds_circulant_karatsuba_8, mds_circulant_karatsuba_12, mds_circulant_karatsuba_16,
    mds_circulant_karatsuba_24,
};
use p3_mds::util::mds_multiply;
use p3_mersenne_31::Mersenne31;
use p3_monolith::mds::goldilocks::MonolithMdsMatrixGoldilocks;
use p3_monolith::mds::mersenne31::MonolithMdsMatrixMersenne31;
use p3_symmetric::Permutation;

#[derive(Debug, Clone, Copy)]
pub(crate) enum NativeMds {
    None,
    Mersenne31,
    Goldilocks,
}

#[derive(Debug, Clone)]
pub(crate) enum MdsStrategy<F, const WIDTH: usize> {
    Dense,
    Circulant([F; WIDTH], NativeMds),
}

impl<F: PrimeCharacteristicRing, const WIDTH: usize> MdsStrategy<F, WIDTH> {
    pub(crate) fn new(matrix: &[[F; WIDTH]; WIDTH]) -> Self
    where
        F: PrimeField64,
    {
        // Width alone is insufficient: Monolith-31 width 24 uses a dense Cauchy matrix.
        if matches!(WIDTH, 8 | 12 | 16 | 24)
            && (0..WIDTH)
                .all(|i| (0..WIDTH).all(|j| matrix[i][j] == matrix[(WIDTH + i - j) % WIDTH][0]))
        {
            let column: [F; WIDTH] = core::array::from_fn(|i| matrix[i][0]);
            // The native integer convolution is valid only for the standard matrix,
            // not every circulant with this field and width. Compare its first column
            // after the complete circulant check above, once per AIR construction.
            let native =
                if WIDTH == 16 && F::ORDER_U64 == Mersenne31::ORDER_U64 {
                    let mut basis = [Mersenne31::ZERO; 16];
                    basis[0] = Mersenne31::ONE;
                    // At width 16 the paper's matrix is independent of the round count.
                    let expected = MonolithMdsMatrixMersenne31::<16, 5>::new().permute(basis);
                    if column
                        .iter()
                        .zip(expected)
                        .all(|(x, y)| x.as_canonical_u64() == y.as_canonical_u64())
                    {
                        NativeMds::Mersenne31
                    } else {
                        NativeMds::None
                    }
                } else if F::ORDER_U64 == Goldilocks::ORDER_U64 && matches!(WIDTH, 8 | 12) {
                    let expected: [Goldilocks; WIDTH] = match WIDTH {
                        8 => {
                            let mut basis = [Goldilocks::ZERO; 8];
                            basis[0] = Goldilocks::ONE;
                            let col = MonolithMdsMatrixGoldilocks.permute(basis);
                            core::array::from_fn(|i| col[i])
                        }
                        12 => {
                            let mut basis = [Goldilocks::ZERO; 12];
                            basis[0] = Goldilocks::ONE;
                            let col = MonolithMdsMatrixGoldilocks.permute(basis);
                            core::array::from_fn(|i| col[i])
                        }
                        _ => unreachable!(),
                    };
                    if column.iter().zip(expected).all(|(x, y): (&F, Goldilocks)| {
                        x.as_canonical_u64() == y.as_canonical_u64()
                    }) {
                        NativeMds::Goldilocks
                    } else {
                        NativeMds::None
                    }
                } else {
                    NativeMds::None
                };
            Self::Circulant(column, native)
        } else {
            Self::Dense
        }
    }

    /// Scalar witness path: standard matrices use the native integer convolution.
    #[inline]
    pub(crate) fn apply_field(&self, state: &mut [F; WIDTH], matrix: &[[F; WIDTH]; WIDTH])
    where
        F: PrimeField64,
    {
        match self {
            Self::Circulant(_, NativeMds::Mersenne31) => {
                let input =
                    core::array::from_fn(|i| Mersenne31::from_u64(state[i].as_canonical_u64()));
                let output: [Mersenne31; 16] =
                    MonolithMdsMatrixMersenne31::<16, 5>::new().permute(input);
                for (s, x) in state.iter_mut().zip(output) {
                    *s = F::from_u64(x.as_canonical_u64());
                }
            }
            Self::Circulant(_, NativeMds::Goldilocks) => match WIDTH {
                8 => {
                    let input =
                        core::array::from_fn(|i| Goldilocks::from_u64(state[i].as_canonical_u64()));
                    let output: [Goldilocks; 8] = MonolithMdsMatrixGoldilocks.permute(input);
                    for (s, x) in state.iter_mut().zip(output) {
                        *s = F::from_u64(x.as_canonical_u64());
                    }
                }
                12 => {
                    let input =
                        core::array::from_fn(|i| Goldilocks::from_u64(state[i].as_canonical_u64()));
                    let output: [Goldilocks; 12] = MonolithMdsMatrixGoldilocks.permute(input);
                    for (s, x) in state.iter_mut().zip(output) {
                        *s = F::from_u64(x.as_canonical_u64());
                    }
                }
                _ => unreachable!(),
            },
            _ => self.apply(state, matrix),
        }
    }

    #[inline]
    pub(crate) fn apply<A: Algebra<F>>(
        &self,
        state: &mut [A; WIDTH],
        matrix: &[[F; WIDTH]; WIDTH],
    ) {
        match self {
            Self::Dense => mds_multiply(state, matrix),
            Self::Circulant(col, _) => match WIDTH {
                8 => mds_circulant_karatsuba_8(
                    state.as_mut_slice().try_into().unwrap(),
                    col.as_slice().try_into().unwrap(),
                ),
                12 => mds_circulant_karatsuba_12(
                    state.as_mut_slice().try_into().unwrap(),
                    col.as_slice().try_into().unwrap(),
                ),
                16 => mds_circulant_karatsuba_16(
                    state.as_mut_slice().try_into().unwrap(),
                    col.as_slice().try_into().unwrap(),
                ),
                24 => mds_circulant_karatsuba_24(
                    state.as_mut_slice().try_into().unwrap(),
                    col.as_slice().try_into().unwrap(),
                ),
                _ => unreachable!("only supported circulant widths are selected"),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use p3_field::PrimeField64;
    use p3_goldilocks::Goldilocks;
    use p3_mersenne_31::Mersenne31;
    use p3_monolith::mds::mersenne31::MonolithMdsMatrixMersenne31;
    use p3_symmetric::Permutation;

    use super::*;

    fn check<F: PrimeField64, const WIDTH: usize>() {
        let col: [F; WIDTH] = core::array::from_fn(|i| F::from_usize(i * i + 1));
        let matrix =
            core::array::from_fn(|i| core::array::from_fn(|j| col[(WIDTH + i - j) % WIDTH]));
        let strategy = MdsStrategy::new(&matrix);
        assert!(matches!(strategy, MdsStrategy::Circulant(..)));
        for input in [
            [F::ZERO; WIDTH],
            [F::NEG_ONE; WIDTH],
            core::array::from_fn(|i| F::from_usize(i * 12345)),
        ] {
            let mut actual = input;
            let mut expected = input;
            strategy.apply(&mut actual, &matrix);
            mds_multiply(&mut expected, &matrix);
            assert_eq!(actual, expected);
        }
        // A single off-circulant entry must disable dispatch even at a supported width.
        let mut dense = matrix;
        dense[1][2] += F::ONE;
        let strategy = MdsStrategy::new(&dense);
        assert!(matches!(strategy, MdsStrategy::Dense));
        let mut actual = col;
        let mut expected = col;
        strategy.apply(&mut actual, &dense);
        mds_multiply(&mut expected, &dense);
        assert_eq!(actual, expected);
    }

    #[test]
    fn circulant_dispatch_matches_dense() {
        check::<Goldilocks, 8>();
        check::<Goldilocks, 12>();
        check::<Mersenne31, 16>();
        check::<Mersenne31, 24>();
    }

    #[test]
    fn native_m31_width_24_keeps_dense_cauchy_matrix() {
        let native = MonolithMdsMatrixMersenne31::<24, 5>::new();
        let cols: [[Mersenne31; 24]; 24] = core::array::from_fn(|j| {
            let mut basis = [Mersenne31::ZERO; 24];
            basis[j] = Mersenne31::ONE;
            native.permute(basis)
        });
        let matrix = core::array::from_fn(|i| core::array::from_fn(|j| cols[j][i]));
        let strategy = MdsStrategy::new(&matrix);
        assert!(matches!(strategy, MdsStrategy::Dense));
        let mut state = core::array::from_fn(|i| Mersenne31::from_usize(i * 999));
        let expected = native.permute(state);
        strategy.apply(&mut state, &matrix);
        assert_eq!(state, expected);
    }
}
