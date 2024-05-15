use core::{fmt::Debug, marker::PhantomData};

use crate::{AbstractExtensionAlgebra, AbstractField, Extension, Field};

pub trait BinomialExtensionParams<F, const D: usize>: Sized + Debug {
    const W: F;
    // 1, ...
    const ORDER_D_SUBGROUP: [F; D];
    const GEN: [F; D];
}

#[derive(Debug)]
pub struct BinomialExtensionAlgebra<F, const D: usize, P>(PhantomData<(F, P)>);

impl<F: Field, const D: usize, P: BinomialExtensionParams<F, D>> AbstractExtensionAlgebra<F>
    for BinomialExtensionAlgebra<F, D, P>
where
    Self: 'static,
{
    const D: usize = D;
    type Repr<AF: AbstractField<F = F>> = [AF; D];
    const GEN: Self::Repr<F> = P::GEN;

    fn mul<AF: AbstractField<F = F>>(
        a: Extension<AF, Self>,
        b: Extension<AF, Self>,
    ) -> Extension<AF, Self> {
        let w_af = AF::from_f(P::W);
        let mut res = Extension::<AF, Self>::default();
        #[allow(clippy::needless_range_loop)]
        for i in 0..D {
            for j in 0..D {
                if i + j >= D {
                    res.0[i + j - D] += a.0[i].clone() * w_af.clone() * b.0[j].clone();
                } else {
                    res.0[i + j] += a.0[i].clone() * b.0[j].clone();
                }
            }
        }
        res
    }

    fn repeated_frobenius(a: Extension<F, Self>, mut count: usize) -> Extension<F, Self> {
        if count == 0 {
            return a;
        }
        count %= D;
        Extension::from_base_fn(|i| a[i] * P::ORDER_D_SUBGROUP[(i * count) % D])
    }

    fn inverse(a: Extension<F, Self>) -> Extension<F, Self> {
        // Writing 'a' for self, we need to compute a^(r-1):
        // r = n^D-1/n-1 = n^(D-1)+n^(D-2)+...+n
        let mut f = Extension::<F, Self>::one();
        for _ in 1..D {
            f = (f * a).repeated_frobenius(1);
        }

        let g = a[0] * f[0] + P::W * (1..D).map(|i| a[i] * f[D - i]).sum();

        f * g.inverse()
    }
}
