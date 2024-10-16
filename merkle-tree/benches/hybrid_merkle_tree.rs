use p3_baby_bear::{BabyBear, DiffusionMatrixBabyBear};
use p3_blake3::Blake3;
use p3_field::Field;
use p3_merkle_tree::{HybridCryptographicHasher, SimpleHybridHasher};
use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
use p3_symmetric::{PaddingFreeSponge, SerializingHasher32};
use rand::thread_rng;

type PermPoseidon =
    Poseidon2<BabyBear, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 16, 7>;
type HPoseidon = PaddingFreeSponge<PermPoseidon, 16, 8, 8>;

type HBlake3 = SerializingHasher32<Blake3>;

fn main() {
    let perm_poseidon = PermPoseidon::new_from_rng_128(
        Poseidon2ExternalMatrixGeneral,
        DiffusionMatrixBabyBear::default(),
        &mut thread_rng(),
    );

    let h_poseidon = HPoseidon::new(perm_poseidon.clone());

    let h_blake3 = HBlake3::new(Blake3 {});

    let h_hybrid = SimpleHybridHasher::new::<
        <BabyBear as Field>::Packing,
        <BabyBear as Field>::Packing,
    >(h_poseidon, h_blake3);

    // bench_merkle_tree::<<F as Field>::Packing, <F as Field>::Packing, H, C, 8>(criterion, h, c);
    // bench_merkle_tree<P, PW, H, C, const DIGEST_ELEMS: usize>

    println!("Hello, hybrid world!");
}
