// Characteristic three makes bus-id exhaustion testable without billions of declarations.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(super) struct TinyParameters;
impl p3_monty_31::MontyParameters for TinyParameters {
    const PRIME: u32 = 3;
    const MONTY_BITS: u32 = 32;
    const MONTY_MU: u32 = 0xaaaa_aaab;
}
impl p3_monty_31::PackedMontyParameters for TinyParameters {}
impl p3_monty_31::FieldParameters for TinyParameters {
    const MONTY_GEN: Tiny = Tiny::new(2);
}
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
impl p3_monty_31::MontyParametersNeon for TinyParameters {
    const PACKED_P: core::arch::aarch64::uint32x4_t = unsafe { core::mem::transmute([3u32; 4]) };
    const PACKED_MU: core::arch::aarch64::int32x4_t =
        unsafe { core::mem::transmute([0xaaaa_aaabu32; 4]) };
}
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512f")
))]
impl p3_monty_31::MontyParametersAVX2 for TinyParameters {
    const PACKED_P: core::arch::x86_64::__m256i = unsafe { core::mem::transmute([3u32; 8]) };
    const PACKED_MU: core::arch::x86_64::__m256i =
        unsafe { core::mem::transmute([0xaaaa_aaabu32; 8]) };
}
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
impl p3_monty_31::MontyParametersAVX512 for TinyParameters {
    const PACKED_P: core::arch::x86_64::__m512i = unsafe { core::mem::transmute([3u32; 16]) };
    const PACKED_MU: core::arch::x86_64::__m512i =
        unsafe { core::mem::transmute([0xaaaa_aaabu32; 16]) };
}
pub(super) type Tiny = p3_monty_31::MontyField31<TinyParameters>;
