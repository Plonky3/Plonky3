use p3_field::TwoAdicField;

pub fn _dft<F: TwoAdicField, const N: usize, const INV: bool>(vals: [F; N]) -> [F; N] {
    assert!(N.is_power_of_two());
    let root: F = if INV {
        F::primitive_root_of_unity(N.trailing_zeros() as usize).inverse()
    } else {
        F::primitive_root_of_unity(N.trailing_zeros() as usize)
    };
    debug_assert!(root.exp_power_of_2(N.trailing_zeros().try_into().unwrap()) == F::ONE);

    let mut ret: [F; N] = [F::default(); N];
    let mut ri = F::ONE; // root^i
    (0..N).for_each(|i| {
        let mut rij = F::ONE; // root^(i*j)
        let mut sum = F::ZERO;
        (0..N).for_each(|j| {
            sum += vals[j] * rij;
            rij *= ri;
        });
        ri *= root;
        debug_assert!(rij == F::ONE);
        ret[i] = sum
            * (if INV {
                F::TWO
                    .exp_u64(N.trailing_zeros().try_into().unwrap())
                    .inverse()
            } else {
                F::ONE
            });
    });
    debug_assert!(ri == F::ONE);
    ret
}

pub fn dft<F: TwoAdicField, const N: usize>(vals: [F; N]) -> [F; N] {
    _dft::<F, N, false>(vals)
}

pub fn idft<F: TwoAdicField, const N: usize>(vals: [F; N]) -> [F; N] {
    _dft::<F, N, true>(vals)
}

#[cfg(test)]
mod tests_mersenne {
    use p3_mersenne_31::{Mersenne31, Mersenne31Complex};
    use rand::Rng;

    use crate::dft::*;

    type B = Mersenne31;
    type F = Mersenne31Complex<Mersenne31>;

    #[test]
    fn test_dft_inversion() {
        const N: usize = 8;
        let mut rng = rand::thread_rng();
        let mut aa = [F::default(); N];
        aa.iter_mut().for_each(|a| *a = F::new_real(rng.gen::<B>()));
        let tt = _dft::<F, 8, false>(aa);
        let aa_t = _dft::<F, 8, true>(tt);
        assert_eq!(aa, aa_t);
    }
}
