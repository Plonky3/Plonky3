//! The prime field `F_p` where `p = 2^31 - 1`.

#![no_std]
#![cfg_attr(
    all(
        feature = "nightly-features",
        target_arch = "x86_64",
        target_feature = "avx512f"
    ),
    feature(stdarch_x86_avx512)
)]

extern crate alloc;

// mod complex;
// mod dft;
// mod extension;
mod extensions;
mod mds;
mod mersenne_31;
mod poseidon2;
// mod radix_2_dit;

// pub use dft::Mersenne31Dft;
pub use mds::*;
pub use mersenne_31::*;
pub use poseidon2::*;
// pub use radix_2_dit::Mersenne31ComplexRadix2Dit;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod aarch64_neon;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub use aarch64_neon::*;

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
mod x86_64_avx2;
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub use x86_64_avx2::*;

#[cfg(all(
    feature = "nightly-features",
    target_arch = "x86_64",
    target_feature = "avx512f"
))]
mod x86_64_avx512;
#[cfg(all(
    feature = "nightly-features",
    target_arch = "x86_64",
    target_feature = "avx512f"
))]
pub use x86_64_avx512::*;

/*
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
struct MyExt;
impl ExtensionAlgebra for MyExt {
    type F = Mersenne31;
    const D: usize = 2;
    type Repr<AF: AbstractField<F = Self::F>> = [AF; 2];
    fn mul<AF: AbstractField<F = Self::F>>(a: Ext<AF, Self>, b: Ext<AF, Self>) -> Ext<AF, Self> {
        Ext([
            a.0[0].clone() * a.0[1].clone(),
            b.0[1].clone() * b.0[0].clone(),
        ])
    }
}
*/
