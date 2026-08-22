//! Codec trait and concrete implementations.

mod bytes_to_field;
mod codec;
mod decode_field;
mod extension;
mod field_to_field;
mod length_prefix;

pub use bytes_to_field::BytesToFieldCodec;
pub use codec::Codec;
pub(crate) use decode_field::{
    decode_field_be_canonical, decode_field_via_extra_bytes, encode_field_be, field_byte_size,
    required_bytes,
};
pub use extension::ExtensionFieldCodec;
pub use field_to_field::FieldToFieldCodec;
pub(crate) use length_prefix::{bound_byte_width, decode_len_be, encode_len_be};
