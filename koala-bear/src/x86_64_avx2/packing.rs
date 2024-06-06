use core::arch::x86_64::__m256i;
use core::mem::transmute;

use p3_monty_31::{FieldParametersAVX2, PackedMontyField31AVX2};

use crate::KoalaBearParameters;

pub type PackedKoalaBearAVX2 = PackedMontyField31AVX2<KoalaBearParameters>;

const WIDTH: usize = 8;

impl FieldParametersAVX2 for KoalaBearParameters {
    const PACKEDP: __m256i = unsafe { transmute::<[u32; WIDTH], _>([0x7f000001; WIDTH]) };
    const PACKEDMU: __m256i = unsafe { transmute::<[u32; WIDTH], _>([0x81000001; WIDTH]) };
}

#[cfg(test)]
mod tests {
    use p3_field_testing::test_packed_field;
    use p3_monty_31::to_monty_array;

    use super::WIDTH;
    use crate::KoalaBear;

    const SPECIAL_VALS: [KoalaBear; WIDTH] = to_monty_array([
        0x00000000, 0x00000001, 0x7f000000, 0x7effffff, 0x3f800000, 0x0ffffffe, 0x68000003,
        0x70000002,
    ]);

    test_packed_field!(
        crate::PackedKoalaBearAVX2,
        crate::PackedKoalaBearAVX2::zero(),
        p3_monty_31::PackedMontyField31AVX2::<crate::KoalaBearParameters>(super::SPECIAL_VALS)
    );
}
