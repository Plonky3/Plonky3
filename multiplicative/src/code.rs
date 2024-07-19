use alloc::vec::Vec;
use core::marker::PhantomData;
use itertools::Itertools;
use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
use p3_util::{reverse_bits_len, reverse_slice_index_bits, SliceExt};

use p3_field::TwoAdicField;
use p3_fri::{CodeFamily, FoldableCodeFamily, LinearCodeFamily};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RsCode<F> {
    log_blowup: usize,
    log_message_len: usize,
    _phantom: PhantomData<F>,
}

impl<F: TwoAdicField> CodeFamily<F> for RsCode<F> {
    fn log_blowup(&self) -> usize {
        self.log_blowup
    }
    fn log_message_len(&self) -> usize {
        self.log_message_len
    }

    fn encode(&self, message: &[F]) -> Vec<F> {
        let mut coeffs = message.to_vec();
        assert_eq!(coeffs.log_strict_len(), self.log_message_len);
        coeffs.resize(coeffs.len() << self.log_blowup, F::zero());
        let mut evals = Radix2Dit::default().dft(coeffs.to_vec());
        reverse_slice_index_bits(&mut evals);
        evals
    }
    fn decode(&self, word: &[F]) -> Vec<F> {
        let mut evals = word.to_vec();
        reverse_slice_index_bits(&mut evals);
        assert_eq!(
            evals.log_strict_len(),
            self.log_message_len + self.log_blowup
        );
        let mut coeffs = Radix2Dit::default().idft(evals);
        assert!(coeffs
            .drain((1 << self.log_message_len)..)
            .all(|x| x.is_zero()));
        coeffs
    }
}

impl<F: TwoAdicField> LinearCodeFamily<F> for RsCode<F> {}

impl<F: TwoAdicField> FoldableCodeFamily<F> for RsCode<F> {
    fn folded_code(mut self) -> Self {
        self.log_message_len -= 1;
        self
    }
    fn fold_word_at_index(&self, beta: F, pair_index: usize, (e0, e1): (F, F)) -> F {
        let subgroup_start = F::two_adic_generator(self.log_message_len + self.log_blowup).exp_u64(
            reverse_bits_len(pair_index, self.log_message_len + self.log_blowup - 1) as u64,
        );
        let mut xs = F::two_adic_generator(1)
            .shifted_powers(subgroup_start)
            .take(2)
            .collect_vec();
        reverse_slice_index_bits(&mut xs);
        // interpolate and evaluate at beta
        e0 + (beta - xs[0]) * (e1 - e0) / (xs[1] - xs[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /*
    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
    */
}
