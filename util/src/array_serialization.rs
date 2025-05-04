use alloc::vec::Vec;
use core::marker::PhantomData;

use serde::de::{SeqAccess, Visitor};
use serde::ser::SerializeTuple;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

pub fn serialize<S: Serializer, T: Serialize, const N: usize>(
    data: &[T; N],
    ser: S,
) -> Result<S::Ok, S::Error> {
    let mut s = ser.serialize_tuple(N)?;
    for item in data {
        s.serialize_element(item)?;
    }
    s.end()
}

struct ArrayVisitor<T, const N: usize>(PhantomData<T>);

impl<'de, T, const N: usize> Visitor<'de> for ArrayVisitor<T, N>
where
    T: Deserialize<'de>,
{
    type Value = [T; N];

    fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_fmt(format_args!("an array of length {}", N))
    }

    #[inline]
    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut data = Vec::with_capacity(N);
        for _ in 0..N {
            match seq.next_element()? {
                Some(val) => data.push(val),
                None => return Err(serde::de::Error::invalid_length(N, &self)),
            }
        }
        match data.try_into() {
            Ok(arr) => Ok(arr),
            Err(_) => unreachable!(),
        }
    }
}
pub fn deserialize<'de, D, T, const N: usize>(deserializer: D) -> Result<[T; N], D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    deserializer.deserialize_tuple(N, ArrayVisitor::<T, N>(PhantomData))
}

#[cfg(test)]
mod tests {
    use serde::{Deserialize, Serialize};
    use serde_json;

    use super::*;

    /// A helper wrapper struct to use serialize/deserialize hooks on arrays.
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Wrapper {
        #[serde(serialize_with = "serialize", deserialize_with = "deserialize")]
        arr: [u32; 3],
    }

    #[test]
    fn test_serialize_array() {
        let data = Wrapper { arr: [10, 20, 30] };

        // Serialize using serde_json
        let json = serde_json::to_string(&data).unwrap();

        // Should match JSON array inside object
        assert_eq!(json, r#"{"arr":[10,20,30]}"#);
    }

    #[test]
    fn test_deserialize_array() {
        let json = r#"{"arr":[1,2,3]}"#;

        // Deserialize JSON into struct
        let parsed: Wrapper = serde_json::from_str(json).unwrap();

        // Verify the contents
        assert_eq!(parsed.arr, [1, 2, 3]);
    }

    #[test]
    fn test_roundtrip_array() {
        let original = Wrapper { arr: [7, 8, 9] };

        // Serialize → Deserialize → Compare
        let json = serde_json::to_string(&original).unwrap();
        let restored: Wrapper = serde_json::from_str(&json).unwrap();

        assert_eq!(original, restored);
    }

    #[test]
    fn test_deserialize_wrong_length() {
        let json = r#"{"arr":[1,2]}"#;

        // Fails because array has length 2 but we expect 3
        let result: Result<Wrapper, _> = serde_json::from_str(json);

        // Must error with invalid length
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_array() {
        #[derive(Serialize, Deserialize, Debug, PartialEq)]
        struct EmptyWrapper {
            #[serde(serialize_with = "serialize", deserialize_with = "deserialize")]
            arr: [u8; 0],
        }

        // Serialize empty array
        let data = EmptyWrapper { arr: [] };
        let json = serde_json::to_string(&data).unwrap();
        assert_eq!(json, r#"{"arr":[]}"#);

        // Deserialize back
        let parsed: EmptyWrapper = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, data);
    }
}
