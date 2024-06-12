use core::arch::x86_64::__m512i;
use core::mem::transmute;

use p3_monty_31::{FieldParametersAVX512, PackedMontyField31AVX512};

use crate::KoalaBearParameters;

pub type PackedKoalaBearAVX512 = PackedMontyField31AVX512<KoalaBearParameters>;

const WIDTH: usize = 16;

impl FieldParametersAVX512 for KoalaBearParameters {
    const PACKEDP: __m512i = unsafe { transmute::<[u32; WIDTH], _>([0x7f000001; WIDTH]) };
    const PACKEDMU: __m512i = unsafe { transmute::<[u32; WIDTH], _>([0x81000001; WIDTH]) };
}

#[cfg(test)]
mod tests {
    use p3_field_testing::test_packed_field;
    use p3_monty_31::to_monty_array;

    use super::WIDTH;
    use crate::KoalaBear;

    const SPECIAL_VALS: [KoalaBear; WIDTH] = to_monty_array([
        0x00000000, 0x00000001, 0x78000000, 0x77ffffff, 0x3c000000, 0x0ffffffe, 0x68000003,
        0x70000002, 0x00000000, 0x00000001, 0x78000000, 0x77ffffff, 0x3c000000, 0x0ffffffe,
        0x68000003, 0x70000002,
    ]);

    test_packed_field!(
        crate::PackedKoalaBearAVX512,
        crate::PackedKoalaBearAVX512::zero(),
        p3_monty_31::PackedMontyField31AVX512::<crate::KoalaBearParameters>(super::SPECIAL_VALS)
    );
}
