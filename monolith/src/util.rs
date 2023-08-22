use sha3::digest::XofReader;
use sha3::Shake128Reader;

pub(crate) fn get_random_u32(shake: &mut Shake128Reader) -> u32 {
    let mut rand = [0u8; 4];
    shake.read(&mut rand);
    u32::from_le_bytes(rand)
}
