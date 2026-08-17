mod vectorized {
    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{
        BasedVectorSpace, PackedValue, PrimeCharacteristicRing, Vectorized, VectorizedExt,
    };

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    fn base_vals(len: usize, offset: u64) -> Vec<F> {
        (0..len).map(|i| F::from_u64(offset + i as u64)).collect()
    }

    /// Build a deterministic `VectorizedExt` whose lane `i` is
    /// `EF::from_basis_coefficients_fn(|d| F::from_u64(offset + i * DIMENSION + d))`.
    fn ext_vectorized<const N: usize>(width: usize, offset: u64) -> VectorizedExt<F, EF, N> {
        let d = <EF as BasedVectorSpace<F>>::DIMENSION;
        let coefficients: Vec<Vectorized<F, N>> = (0..d)
            .map(|dcoef| {
                let vals: Vec<F> = (0..width)
                    .map(|i| F::from_u64(offset + (i * d + dcoef) as u64))
                    .collect();
                *Vectorized::<F, N>::from_slice(&vals)
            })
            .collect();
        VectorizedExt::<F, EF, N>::from_vectorized_basis_coefficients(&coefficients)
    }

    fn ext_lane(offset: u64, i: usize) -> EF {
        let d = <EF as BasedVectorSpace<F>>::DIMENSION;
        EF::from_basis_coefficients_fn(|dcoef| F::from_u64(offset + (i * d + dcoef) as u64))
    }

    fn round_trip<const N: usize>() {
        let width = Vectorized::<F, N>::WIDTH;
        let vals = base_vals(width, 1);

        let from_fn = Vectorized::<F, N>::from_fn(|i| vals[i]);
        assert_eq!(from_fn.as_slice(), vals.as_slice());
        for (i, &v) in vals.iter().enumerate() {
            assert_eq!(from_fn.extract(i), v);
        }

        assert_eq!(
            Vectorized::<F, N>::from_slice(&vals).as_slice(),
            vals.as_slice()
        );

        let mut vals_mut = vals.clone();
        assert_eq!(
            Vectorized::<F, N>::from_slice_mut(&mut vals_mut).as_slice(),
            vals.as_slice()
        );
    }

    fn base_homomorphism<const N: usize>() {
        let width = Vectorized::<F, N>::WIDTH;
        let a_vals = base_vals(width, 1);
        let b_vals = base_vals(width, 1000);

        let a = *Vectorized::<F, N>::from_slice(&a_vals);
        let b = *Vectorized::<F, N>::from_slice(&b_vals);

        let sum = a + b;
        let diff = a - b;
        let prod = a * b;
        let neg = -a;

        for i in 0..width {
            assert_eq!(sum.as_slice()[i], a_vals[i] + b_vals[i]);
            assert_eq!(diff.as_slice()[i], a_vals[i] - b_vals[i]);
            assert_eq!(prod.as_slice()[i], a_vals[i] * b_vals[i]);
            assert_eq!(neg.as_slice()[i], -a_vals[i]);
        }
    }

    fn base_ring_ops<const N: usize>() {
        let width = Vectorized::<F, N>::WIDTH;
        let vals = base_vals(width, 3);
        let a = *Vectorized::<F, N>::from_slice(&vals);

        let cube = a.cube();
        let exp3 = a.exp_const_u64::<3>();
        let shifted = a.mul_2exp_u64(2);

        for (i, &v) in vals.iter().enumerate() {
            assert_eq!(cube.as_slice()[i], v.cube());
            assert_eq!(exp3.as_slice()[i], v.exp_const_u64::<3>());
            assert_eq!(shifted.as_slice()[i], v.mul_2exp_u64(2));
        }
    }

    fn base_dot_product_and_sum<const N: usize>() {
        const M: usize = 3;
        let width = Vectorized::<F, N>::WIDTH;
        let us: [Vectorized<F, N>; M] = core::array::from_fn(|j| {
            *Vectorized::<F, N>::from_slice(&base_vals(width, (10 * j + 1) as u64))
        });
        let vs: [Vectorized<F, N>; M] = core::array::from_fn(|j| {
            *Vectorized::<F, N>::from_slice(&base_vals(width, (10 * j + 50) as u64))
        });

        let dot = Vectorized::<F, N>::dot_product(&us, &vs);
        let sum = Vectorized::<F, N>::sum_array::<M>(&us);

        for i in 0..width {
            let expected_dot: F = (0..M)
                .map(|j| us[j].as_slice()[i] * vs[j].as_slice()[i])
                .sum();
            let expected_sum: F = (0..M).map(|j| us[j].as_slice()[i]).sum();
            assert_eq!(dot.as_slice()[i], expected_dot);
            assert_eq!(sum.as_slice()[i], expected_sum);
        }
    }

    fn ext_homomorphism<const N: usize>() {
        let width = Vectorized::<F, N>::WIDTH;

        let a = ext_vectorized::<N>(width, 1);
        let b = ext_vectorized::<N>(width, 1000);

        for i in 0..width {
            assert_eq!(a.extract(i), ext_lane(1, i));
            assert_eq!(b.extract(i), ext_lane(1000, i));
        }

        let sum = a + b;
        let diff = a - b;
        let prod = a * b;
        let neg = -a;

        for i in 0..width {
            assert_eq!(sum.extract(i), ext_lane(1, i) + ext_lane(1000, i));
            assert_eq!(diff.extract(i), ext_lane(1, i) - ext_lane(1000, i));
            assert_eq!(prod.extract(i), ext_lane(1, i) * ext_lane(1000, i));
            assert_eq!(neg.extract(i), -ext_lane(1, i));
        }

        // Mixed `VectorizedExt` op `Vectorized<F, N>`.
        let raw_base_vals = base_vals(width, 7);
        let base = *Vectorized::<F, N>::from_slice(&raw_base_vals);

        let mixed_sum = a + base;
        let mixed_diff = a - base;
        let mixed_prod = a * base;

        for (i, &v) in raw_base_vals.iter().enumerate() {
            assert_eq!(mixed_sum.extract(i), ext_lane(1, i) + v);
            assert_eq!(mixed_diff.extract(i), ext_lane(1, i) - v);
            assert_eq!(mixed_prod.extract(i), ext_lane(1, i) * v);
        }
    }

    fn ext_ring_ops<const N: usize>() {
        let width = Vectorized::<F, N>::WIDTH;
        let a = ext_vectorized::<N>(width, 5);

        let cube = a.cube();
        let exp3 = a.exp_const_u64::<3>();
        let shifted = a.mul_2exp_u64(3);

        for i in 0..width {
            let lane = ext_lane(5, i);
            assert_eq!(cube.extract(i), lane.cube());
            assert_eq!(exp3.extract(i), lane.exp_const_u64::<3>());
            assert_eq!(shifted.extract(i), lane.mul_2exp_u64(3));
        }
    }

    macro_rules! vectorized_tests {
        ($n:literal, $mod_name:ident) => {
            mod $mod_name {
                use super::*;

                #[test]
                fn round_trip_test() {
                    round_trip::<$n>();
                }

                #[test]
                fn base_homomorphism_test() {
                    base_homomorphism::<$n>();
                }

                #[test]
                fn base_ring_ops_test() {
                    base_ring_ops::<$n>();
                }

                #[test]
                fn base_dot_product_and_sum_test() {
                    base_dot_product_and_sum::<$n>();
                }

                #[test]
                fn ext_homomorphism_test() {
                    ext_homomorphism::<$n>();
                }

                #[test]
                fn ext_ring_ops_test() {
                    ext_ring_ops::<$n>();
                }
            }
        };
    }

    // N = 3 alongside N = 2 so that with W = F::Packing::WIDTH, `i / W` vs `i % W`
    // mistakes in lane indexing don't coincidentally alias to the right answer.
    vectorized_tests!(2, n2);
    vectorized_tests!(3, n3);
}
