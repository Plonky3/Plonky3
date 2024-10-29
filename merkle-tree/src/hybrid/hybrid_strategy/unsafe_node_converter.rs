use core::mem;

use p3_baby_bear::BabyBear;
use p3_field::AbstractField;

use super::NodeConverter;

#[derive(Clone)]
pub struct UnsafeNodeConverter256BabyBearBytes {}

impl<const WIDTH: usize> NodeConverter<[[BabyBear; WIDTH]; 8], [[u8; WIDTH]; 32]>
    for UnsafeNodeConverter256BabyBearBytes
{
    fn to_n1(input: [[u8; WIDTH]; 32]) -> [[BabyBear; WIDTH]; 8] {
        // TODO try to avoid copying; cannot use transmute() because of WIDTH
        // unsafe {
        //     let u32_array = mem::transmute_copy::<[[u8; WIDTH]; 32], [u32; WIDTH * 8]>(&input);

        //     let babybear_array: [BabyBear; WIDTH * 8] = u32_array.map(|bb| BabyBear::from_wrapped_u32(bb)).try_into().unwrap();

        //     mem::transmute::<[BabyBear; WIDTH * 8], [[BabyBear; WIDTH]; 8]>(babybear_array)
        // }

        unimplemented!()
    }

    fn to_n2(input: [[BabyBear; WIDTH]; 8]) -> [[u8; WIDTH]; 32] {
        // let u8_array: Vec<u8> = input.into_iter().flatten().map(bb_to_bytes).into_iter().collect();

        // let u8_subarrays: Vec<[u8; WIDTH]> = u8_array.chunks(WIDTH).map(|chunk| chunk.try_into().unwrap()).collect();

        // u8_array.try_into().unwrap()

        // unsafe {
        //     mem::transmute::<[[BabyBear; WIDTH]; 8], [[u8; WIDTH]; 32]>(input)

        // }

        // TODO try to avoid copying; cannot use transmute() because of WIDTH
        unsafe { mem::transmute_copy::<[[BabyBear; WIDTH]; 8], [[u8; WIDTH]; 32]>(&input) }
    }
}

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
