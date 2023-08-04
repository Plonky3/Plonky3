#[cfg(test)]
mod tests_poly_fp17 {
    use p3_field::AbstractField;
    use polynomial::{AbstractPolynomialCoefficients, AbstractPolynomialEvaluations};
    use rand::Rng;
    type F = polynomial::fp17::Fp17;
    type PC = polynomial::coeffs::CyclicPolynomialCoefficients<F, 8>;
    #[test]
    fn test_coeff_add_sub() {
        let mut rng = rand::thread_rng();
        let aa: PC = rng.gen();
        let bb: PC = rng.gen();
        let cc = aa + bb;
        let dd = cc - bb;
        assert_eq!(aa, dd);
    }
    #[test]
    fn test_fft_mul_random_half() {
        let mut rng = rand::thread_rng();
        let mut aa: PC = rng.gen();
        let mut bb: PC = rng.gen();
        for i in 4..8 {
            aa.vals[i] = F::ZERO;
            bb.vals[i] = F::ZERO;
        }
        let cc = aa * bb;
        let dd = (aa.fft() * bb.fft()).ifft();
        assert_eq!(cc, dd);
    }
    #[test]
    fn test_fft_mul_random_full() {
        let mut rng = rand::thread_rng();
        let aa: PC = rng.gen();
        let bb: PC = rng.gen();
        let cc = aa * bb;
        let dd = (aa.fft() * bb.fft()).ifft();
        assert_eq!(cc, dd);
    }
    #[test]
    fn test_dft_mul_random_half() {
        let mut rng = rand::thread_rng();
        let mut aa: PC = rng.gen();
        let mut bb: PC = rng.gen();
        for i in 4..8 {
            aa.vals[i] = F::ZERO;
            bb.vals[i] = F::ZERO;
        }
        let cc = aa * bb;
        let dd = (aa.dft() * bb.dft()).idft();
        assert_eq!(cc, dd);
    }
    #[test]
    fn test_fft_add_random() {
        let mut rng = rand::thread_rng();
        let aa: PC = rng.gen();
        let bb: PC = rng.gen();
        let cc = aa + bb;
        let dd = (aa.fft() + bb.fft()).ifft();
        assert_eq!(cc, dd);
    }
    #[test]
    fn test_fft_sub_random() {
        let mut rng = rand::thread_rng();
        let aa: PC = rng.gen();
        let bb: PC = rng.gen();
        let cc = aa - bb;
        let dd = (aa.fft() - bb.fft()).ifft();
        assert_eq!(cc, dd);
    }
    #[test]
    fn test_eval_add_random() {
        let mut rng = rand::thread_rng();
        let aa: PC = rng.gen();
        let bb: PC = rng.gen();
        let x: F = rng.gen();
        let ax = aa.eval(x);
        let bx = bb.eval(x);
        let cx = ax + bx;
        let dx = (aa + bb).eval(x);
        assert_eq!(cx, dx);
    }
    #[test]
    fn test_eval_mul_random_half() {
        let mut rng = rand::thread_rng();
        let mut aa: PC = rng.gen();
        let mut bb: PC = rng.gen();
        for i in 4..8 {
            aa.vals[i] = F::ZERO;
            bb.vals[i] = F::ZERO;
        }
        let x: F = rng.gen();
        let ax = aa.eval(x);
        let bx = bb.eval(x);
        let cx = ax * bx;
        let dx = (aa * bb).eval(x);
        assert_eq!(cx, dx);
    }
}
