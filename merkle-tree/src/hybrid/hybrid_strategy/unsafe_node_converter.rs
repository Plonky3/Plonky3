use core::mem;

use p3_baby_bear::BabyBear;
use p3_field::AbstractField;

use super::NodeConverter;

#[derive(Clone)]
pub struct UnsafeNodeConverter256BabyBearBytes {}

impl NodeConverter<[BabyBear; 8], [u8; 32]> for UnsafeNodeConverter256BabyBearBytes {
    fn to_n1(input: [u8; 32]) -> [BabyBear; 8] {
        unsafe {
            let flat_input: [u32; 8] = mem::transmute(input);
            flat_input
                .map(|bb| BabyBear::from_wrapped_u32(bb))
                .try_into()
                .unwrap()
        }
    }

    fn to_n2(input: [BabyBear; 8]) -> [u8; 32] {
        unsafe { mem::transmute(input) }
    }
}
