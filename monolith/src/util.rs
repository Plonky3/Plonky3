//! SHAKE-128 helper utilities for Monolith constant generation.

use sha3::Shake128Reader;
use sha3::digest::XofReader;

/// Read 4 bytes from a SHAKE-128 stream and interpret as a little-endian u32.
pub(crate) fn get_random_u32(shake: &mut Shake128Reader) -> u32 {
    let mut rand = [0u8; 4];
    shake.read(&mut rand);
    u32::from_le_bytes(rand)
}

/// Read 8 bytes from a SHAKE-128 stream and interpret as a little-endian u64.
pub(crate) fn get_random_u64(shake: &mut Shake128Reader) -> u64 {
    let mut rand = [0u8; 8];
    shake.read(&mut rand);
    u64::from_le_bytes(rand)
}
