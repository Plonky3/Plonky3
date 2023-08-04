#[cfg(test)]
mod tests_fft_cm31 {
    use p3_field::AbstractField;
    use p3_mersenne_31::{Mersenne31, Mersenne31Complex};
    use rand::Rng;
    type B = Mersenne31;
    type F = Mersenne31Complex<Mersenne31>;
    use polynomial::dft::{dft, idft};
    use polynomial::fft::{fft, ifft};
    #[test]
    fn test_fft_ifft_simple() {
        const N: usize = 8;
        let mut aa = [F::ZERO; N];
        aa[0] = F::ONE;
        let aa_0 = aa;
        fft(&mut aa);
        ifft(&mut aa);
        assert_eq!(aa, aa_0);
    }
    #[test]
    fn test_fft_ifft_random() {
        const N: usize = 8;
        let mut rng = rand::thread_rng();
        let mut aa = [F::default(); N];
        aa.iter_mut().for_each(|a| *a = F::new_real(rng.gen::<B>()));
        let aa_0 = aa;
        fft(&mut aa);
        ifft(&mut aa);
        assert_eq!(aa, aa_0);
    }
    #[test]
    fn test_fft_dft_simple_0() {
        const N: usize = 8;
        let mut aa = [F::ZERO; N];
        let aa_0 = dft(aa);
        fft(&mut aa);
        assert_eq!(aa, aa_0);
    }
    #[test]
    fn test_fft_dft_simple_1() {
        const N: usize = 8;
        let mut aa = [F::ZERO; N];
        aa[0] = F::ONE;
        let aa_0 = dft(aa);
        fft(&mut aa);
        assert_eq!(aa, aa_0);
    }
    #[test]
    fn test_fft_dft_simple_2() {
        const N: usize = 8;
        let mut aa = [F::ONE; N];
        let aa_0 = dft(aa);
        fft(&mut aa);
        assert_eq!(aa, aa_0);
    }
    #[test]
    fn test_fft_dft_random() {
        const N: usize = 8;
        let mut rng = rand::thread_rng();
        let mut aa = [F::default(); N];
        aa.iter_mut().for_each(|a| *a = F::new_real(rng.gen::<B>()));
        let aa_0 = dft(aa);
        fft(&mut aa);
        assert_eq!(aa, aa_0);
    }
    #[test]
    fn test_ifft_idft_random() {
        const N: usize = 8;
        let mut rng = rand::thread_rng();
        let mut aa = [F::default(); N];
        aa.iter_mut()
            .for_each(|a| *a = F::new(rng.gen::<B>(), rng.gen::<B>()));
        let aa_0 = idft(aa);
        ifft(&mut aa);
        assert_eq!(aa, aa_0);
    }
}
