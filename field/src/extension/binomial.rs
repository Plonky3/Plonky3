use core::{fmt::Debug, marker::PhantomData};

use crate::{
    AbstractExtension, AbstractExtensionAlgebra, AbstractField, Extension, Field, HasBase,
};

pub trait BinomialExtensionParams<F, const D: usize>: Sized + Send + Sync + Debug {
    const W: F;
    // 1, ...
    const ORDER_D_SUBGROUP: [F; D];
    const GEN: [F; D];
}

#[derive(Debug)]
pub struct BinomialExtensionAlgebra<F, const D: usize, P>(PhantomData<(F, P)>);

impl<F: Field, const D: usize, P: BinomialExtensionParams<F, D>> HasBase
    for BinomialExtensionAlgebra<F, D, P>
{
    type Base = F;
}

impl<F: Field, const D: usize, P: BinomialExtensionParams<F, D>> AbstractExtensionAlgebra
    for BinomialExtensionAlgebra<F, D, P>
where
    Self: 'static,
{
    const D: usize = D;
    type Repr<AF: AbstractField<F = F>> = [AF; D];
    const GEN: Self::Repr<F> = P::GEN;

    fn mul<AF: AbstractField<F = F>>(
        a: AbstractExtension<AF, Self>,
        b: AbstractExtension<AF, Self>,
    ) -> AbstractExtension<AF, Self> {
        let w_af = AF::from_f(P::W);
        let mut res = AbstractExtension::<AF, Self>::default();
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

    fn repeated_frobenius<AF: AbstractField<F = F>>(
        a: AbstractExtension<AF, Self>,
        mut count: usize,
    ) -> AbstractExtension<AF, Self> {
        if count == 0 {
            return a;
        }
        count %= D;
        AbstractExtension::from_base_fn(|i| {
            a[i].clone() * AF::from_f(P::ORDER_D_SUBGROUP[(i * count) % D])
        })
    }

    fn inverse(a: Extension<Self>) -> Extension<Self> {
        // Writing 'a' for self, we need to compute a^(r-1):
        // r = n^D-1/n-1 = n^(D-1)+n^(D-2)+...+n
        let mut f = Extension::<Self>::one();
        for _ in 1..D {
            f = (f * a).repeated_frobenius(1);
        }

        let g = a[0] * f[0] + P::W * (1..D).map(|i| a[i] * f[D - i]).sum();

        f * g.inverse()
    }
}
