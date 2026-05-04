//! Codec trait and concrete implementations.

pub mod bytes_to_field;
pub mod codec;
pub mod decode_field;
pub mod extension;
pub mod field_to_bytes;
pub mod field_to_field;

pub use bytes_to_field::BytesToFieldCodec;
pub use codec::Codec;
pub use extension::ExtensionFieldCodec;
pub use field_to_bytes::FieldToBytesCodec;
pub use field_to_field::FieldToFieldCodec;
