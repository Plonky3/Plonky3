use p3_field::AbstractField;

use super::split_at_mut_unchecked;
use crate::{FieldParameters, MontyField31, MontyParameters};

impl<MP: MontyParameters + FieldParameters> MontyField31<MP> {
    #[inline]
    fn backward_pass(a: &mut [Self], root: Self) {
        let half_n = a.len() / 2;
        let mut w = Self::one();
        for i in 0..half_n {
            let s = a[i];
            let tw = a[i + half_n] * w;
            a[i] = s + tw;
            a[i + half_n] = s - tw;
            w *= root;
        }
    }

    #[inline]
    pub fn backward_fft(a: &mut [Self], root: Self) {
        let n = a.len();

        if n > 1 {
            let (a0, a1) = unsafe { split_at_mut_unchecked(a, n / 2) };
            let root_sqr = root * root;
            Self::backward_fft(a0, root_sqr);
            Self::backward_fft(a1, root_sqr);

            Self::backward_pass(a, root);
        }
    }
}
