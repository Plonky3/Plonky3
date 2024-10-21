use alloc::vec::Vec;

use p3_baby_bear::{BabyBear, BabyBearParameters};
use p3_field::AbstractField;
use p3_monty_31::{MontyField31, MontyParameters};

use super::NodeConverter;

#[derive(Clone)]
pub struct NodeConverter256BabyBearBytes {}

// TODO study endianness, security, etc.
// TODO improve efficiency?
impl NodeConverter<[BabyBear; 8], [u8; 32]> for NodeConverter256BabyBearBytes {
    fn to_n1(input: [u8; 32]) -> [BabyBear; 8] {
        let mut values = [BabyBear::default(); 8];

        for i in 0..8 {
            let value = u32::from_le_bytes(input[i * 4..i * 4 + 4].try_into().unwrap());
            values[i] = BabyBear::from_wrapped_u32(value);
        }

        values
    }

    fn to_n2(input: [BabyBear; 8]) -> [u8; 32] {
        let mut values = [0u8; 32];
        for i in 0..8 {
            values[i * 4..i * 4 + 4].copy_from_slice(&BabyBear::to_u32(&input[i]).to_le_bytes());
        }
        values
    }
}

mod tests {
    use alloc::vec::Vec;

    use p3_baby_bear::{BabyBear, BabyBearParameters};
    use p3_monty_31::{MontyField31, MontyParameters};

    use crate::hybrid::hybrid_strategy::NodeConverter;
    use crate::NodeConverter256BabyBearBytes;

    #[test]
    fn test_conversion_baby_bear_bytes() {
        const P: u32 = <BabyBearParameters as MontyParameters>::PRIME;

        // Check BabyBear -> bytes -> BabyBear is the identity
        for _ in 0..1000 {
            let elems: [BabyBear; 8] = rand::random();
            let bytes = NodeConverter256BabyBearBytes::to_n2(elems.clone());
            assert_eq!(elems, NodeConverter256BabyBearBytes::to_n1(bytes));
        }

        // Check bytes -> BabyBear -> bytes is the (element-wise)
        // modular-reduction map modulo BabyBear::PRIME
        fn test_bytes(bytes: [u8; 32]) {
            let reduced_bytes: [u8; 32] = bytes
                .chunks(4)
                .map(|chunk| {
                    let value = u32::from_le_bytes(chunk.try_into().unwrap());
                    let modular_value = value % P;
                    modular_value.to_le_bytes()
                })
                .flatten()
                .collect::<Vec<u8>>()
                .try_into()
                .unwrap();

            let elems = NodeConverter256BabyBearBytes::to_n1(bytes);
            assert_eq!(reduced_bytes, NodeConverter256BabyBearBytes::to_n2(elems));
        }

        // - on random values
        for _ in 0..1000 {
            test_bytes(rand::random());
        }

        // - on wrap-around values
        let test_vector = [
            [P, P + 2, P, P, P + 1, P + 2, P + 3, P + 4],
            [P - 1, P + 42, P - 1000, P + 7, P, P + 2, P + 9, P + 123],
        ];

        for elems in test_vector {
            let bytes: [u8; 32] = elems
                .map(|elem| elem.to_le_bytes())
                .into_iter()
                .flatten()
                .collect::<Vec<u8>>()
                .try_into()
                .unwrap();

            test_bytes(bytes);
        }
    }
}
