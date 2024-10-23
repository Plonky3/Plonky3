use core::any::type_name;

const KNOWN_HASHERS: [&str; 5] = ["Blake3", "Keccak", "Poseidon2", "Rescue", "Sha256"];

// Debug function to print the type of the hash
pub fn pretty_hash_type<T>() -> &'static str {
    let name = type_name::<T>();

    for known_hasher in KNOWN_HASHERS.iter() {
        if name.contains(known_hasher) {
            return known_hasher;
        }
    }

    let pretty_len = if name.len() > 64 { 64 } else { name.len() };
    &name[0..pretty_len]
}
