use alloc::vec::Vec;

use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, PackedValue};

use super::NodeConverter;
use crate::unpack_array;

/// Converter for 256-bit nodes between `BabyBear` representation (8 field
/// elements) and `u8` representation (32 bytes). This converter treats
/// `PackedValue`s over those two basic types (such as `PackedBabyBearNeon`)
/// abstractly and is therefore forced the trait methods, which involve
/// copying/cloning from references and are therefore not very performant. For a
/// more optimal implementation which works only for arrays and uses hard-casts,
/// cf.
/// [UnsafeNodeConverter256BabyBearBytes][crate::UnsafeNodeConverter256BabyBearBytes].
///
/// Converting packed types with `WIDTH` > 1 is done through a transposition (to
/// match the expected relation between the types) and therefore incurs some
/// overhead (also avoided in
/// [UnsafeNodeConverter256BabyBearBytes][crate::UnsafeNodeConverter256BabyBearBytes]).
/// Suppose, for instance, that `PW1::WIDTH` and `PW2::WIDTH` are 2,
/// `PW1::Value` is `u8`, and `PW2::Value` is `BabyBear`. Then conversion of an
/// input of type `[PW2; 8]` into `[PW1; 32]` proceeds roughly as follows:
/// 1. Unpack `[PW2; 8]` into `[[BabyBear; 2]; 8]`
/// 2. Transpose the result into `[[BabyBear; 8]; 2]`
/// 3. Convert each `[BabyBear; 8]` into `[u8; 32]`, resulting in `[[u8; 32];
///    2]`
/// 4. Transpose into `[[u8; 2]; 32]`
/// 5. Pack the result into `[PW1; 32]`
#[derive(Clone)]
pub struct NodeConverter256BabyBearBytes {}

impl<PW1, PW2> NodeConverter<[PW1; 8], [PW2; 32]> for NodeConverter256BabyBearBytes
where
    PW1: PackedValue<Value = BabyBear>,
    PW2: PackedValue<Value = u8>,
{
    fn to_n1(input: [PW2; 32]) -> [PW1; 8] {
        assert_eq!(PW1::WIDTH, PW2::WIDTH);

        let values_w2 = unpack_array(input);

        let values_w1: Vec<[BabyBear; 8]> = values_w2.map(|v| bytes32_to_bb8(v)).collect();

        let mut result: Vec<PW1> = Vec::new();

        for col in 0..8 {
            let col_w1: Vec<BabyBear> = values_w1.iter().map(|row| row[col]).collect();
            result.push(PW1::from_slice(&col_w1).clone());
        }

        // PW1 may not implement Debug and so unwrap() cannot be used.
        match result.try_into() {
            Ok(v) => v,
            Err(_) => panic!("Failed to convert to array"),
        }
    }

    fn to_n2(input: [PW1; 8]) -> [PW2; 32] {
        assert_eq!(PW1::WIDTH, PW2::WIDTH);

        let values_w1 = unpack_array(input);

        let values_w2: Vec<[u8; 32]> = values_w1.map(|v| bb8_to_bytes32(v)).collect();

        let mut result: Vec<PW2> = Vec::new();

        for col in 0..32 {
            let col_w2: Vec<u8> = values_w2.iter().map(|row| row[col]).collect();
            result.push(PW2::from_slice(&col_w2).clone());
        }

        // PW2 may not implement Debug and so unwrap() cannot be used.
        match result.try_into() {
            Ok(v) => v,
            Err(_) => panic!("Failed to convert to array"),
        }
    }
}

// Implementation of conversion in the other direction.
impl<PW1, PW2> NodeConverter<[PW1; 32], [PW2; 8]> for NodeConverter256BabyBearBytes
where
    PW1: PackedValue<Value = u8>,
    PW2: PackedValue<Value = BabyBear>,
{
    fn to_n1(input: [PW2; 8]) -> [PW1; 32] {
        <Self as NodeConverter<[PW2; 8], [PW1; 32]>>::to_n2(input)
    }

    fn to_n2(input: [PW1; 32]) -> [PW2; 8] {
        <Self as NodeConverter<[PW2; 8], [PW1; 32]>>::to_n1(input)
    }
}

// Converts an array of 32 bytes into an array of 8 BabyBear field elements
fn bytes32_to_bb8(input: [u8; 32]) -> [BabyBear; 8] {
    let mut values = [BabyBear::default(); 8];

    for i in 0..8 {
        let value = u32::from_le_bytes(input[i * 4..i * 4 + 4].try_into().unwrap());
        values[i] = BabyBear::from_wrapped_u32(value);
    }

    values
}

// Converts an array of 8 BabyBear field elements into an array of 32 bytes
fn bb8_to_bytes32(input: [BabyBear; 8]) -> [u8; 32] {
    let mut values = [0u8; 32];
    for i in 0..8 {
        values[i * 4..i * 4 + 4].copy_from_slice(&BabyBear::to_u32(&input[i]).to_le_bytes());
    }
    values
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_baby_bear::{BabyBear, BabyBearParameters};
    use p3_monty_31::MontyParameters;

    use crate::hybrid::hybrid_strategy::NodeConverter;
    use crate::NodeConverter256BabyBearBytes;

    #[test]
    fn test_conversion_baby_bear_bytes() {
        const P: u32 = <BabyBearParameters as MontyParameters>::PRIME;

        // Check BabyBear -> bytes -> BabyBear is the identity
        for _ in 0..1000 {
            let elems: [BabyBear; 8] = rand::random();
            let bytes: [u8; 32] = NodeConverter256BabyBearBytes::to_n2(elems.clone());
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

            let elems: [BabyBear; 8] = NodeConverter256BabyBearBytes::to_n1(bytes);
            assert_eq!(reduced_bytes, NodeConverter256BabyBearBytes::to_n2(elems));
        }

        // On random values
        for _ in 0..1000 {
            test_bytes(rand::random());
        }

        // On wrap-around values
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
