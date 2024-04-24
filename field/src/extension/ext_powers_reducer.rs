use alloc::vec;
use alloc::vec::Vec;
use itertools::{izip, Itertools};

use crate::{AbstractField, ExtensionField, Field, PackedValue};

/// For an extension reducing factor a, and base vector x, can calculate sum(a^i * x_i) quickly.
/// You must call prepare_for_width first to populate powers of a, then it is efficient to use
/// with several base vectors. Subsequently preparing for greater widths will reuse precomputation.
pub struct ExtensionPowersReducer<F: Field, EF> {
    factor: EF,
    next: EF,

    powers: Vec<EF>,
    // Say EF::D = 2 and F::Packing::WIDTH = 3
    //    powers       vertically_packed_powers
    // [               [
    //   EF( 0, 1), \
    //   EF( 2, 3),  >   [P(0,2,4), P(1,3,5)],
    //   EF( 4, 5), /
    //   EF( 6, 7), \
    //   EF( 8, 9),  >   [P(6,8,10), P(7,9,11)],
    //   EF(10,11), /
    // ]               ]
    vertically_packed_powers: Vec<Vec<F::Packing>>,
}

impl<F: Field, EF: ExtensionField<F>> ExtensionPowersReducer<F, EF> {
    pub fn new(factor: EF) -> Self {
        /*
        let powers: Vec<EF> = base
            .powers()
            .take(max_width.next_multiple_of(F::Packing::WIDTH))
            .collect();

        let transposed_packed: Vec<Vec<F::Packing>> = transpose_vec(
            (0..EF::D)
                .map(|d| {
                    F::Packing::pack_slice(
                        &powers.iter().map(|a| a.as_base_slice()[d]).collect_vec(),
                    )
                    .to_vec()
                })
                .collect(),
        );
        */
        Self {
            factor,
            next: EF::one(),
            powers: vec![],
            vertically_packed_powers: vec![],
        }
    }

    pub fn prepare_for_width(&mut self, width: usize) {
        while self.powers.len() < width {
            let old_width = self.powers.len();
            for _ in 0..F::Packing::WIDTH {
                self.powers.push(self.next);
                self.next *= self.factor;
            }
            let new_powers = &self.powers[old_width..];
            self.vertically_packed_powers.push(
                (0..EF::D)
                    .map(|d| F::Packing::from_fn(|i| new_powers[i].as_base_slice()[d]))
                    .collect_vec(),
            );
        }
    }

    // Compute sum_i base^i * x_i
    pub fn reduce_ext(&self, xs: &[EF]) -> EF {
        assert!(self.powers.len() >= xs.len());
        self.powers.iter().zip(xs).map(|(&pow, &x)| pow * x).sum()
    }

    // Same as `self.powers.iter().zip(xs).map(|(&pow, &x)| pow * x).sum()`
    pub fn reduce_base(&self, xs: &[F]) -> EF {
        assert!(self.powers.len() >= xs.len());
        let (xs_packed, xs_sfx) = F::Packing::pack_slice_with_suffix(xs);
        let mut sums = (0..EF::D).map(|_| F::Packing::zero()).collect::<Vec<_>>();
        for (&x, pows) in izip!(xs_packed, &self.vertically_packed_powers) {
            for d in 0..EF::D {
                sums[d] += x * pows[d];
            }
        }
        let packed_sum = EF::from_base_fn(|d| sums[d].as_slice().iter().copied().sum());
        let sfx_sum = xs_sfx
            .iter()
            .zip(&self.powers[(xs_packed.len() * F::Packing::WIDTH)..])
            .map(|(&x, &pow)| pow * x)
            .sum::<EF>();
        packed_sum + sfx_sum
    }
}
