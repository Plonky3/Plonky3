use p3_baby_bear::{BabyBear, DiffusionMatrixBabyBear};
use p3_blake3::Blake3;
use p3_field::{Field, PackedValue};
use p3_merkle_tree::{
    HybridCryptographicHasher, HybridPseudoCompressionFunction, SimpleHybridCompressor,
    SimpleHybridHasher,
};
use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
use p3_symmetric::{
    CompressionFunctionFromHasher, PaddingFreeSponge, SerializingHasher32, TruncatedPermutation,
};
use rand::thread_rng;

type PermPoseidon =
    Poseidon2<BabyBear, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 16, 7>;
type HPoseidon = PaddingFreeSponge<PermPoseidon, 16, 8, 8>;
type CPoseidon = TruncatedPermutation<PermPoseidon, 2, 8, 16>;

type HBlake3 = SerializingHasher32<Blake3>;
type CBlake3 = CompressionFunctionFromHasher<Blake3, 2, 32>;

fn main() {
    let perm_poseidon = PermPoseidon::new_from_rng_128(
        Poseidon2ExternalMatrixGeneral,
        DiffusionMatrixBabyBear::default(),
        &mut thread_rng(),
    );

    let h_poseidon = HPoseidon::new(perm_poseidon.clone());
    let c_poseidon = CPoseidon::new(perm_poseidon);

    let h_blake3 = HBlake3::new(Blake3 {});
    let c_blake3 = CBlake3::new(Blake3 {});

    // let h_hybrid = SimpleHybridHasher::new::<
    //     <BabyBear as Field>::Packing,
    //     <BabyBear as Field>::Packing,
    // >(h_poseidon, h_blake3);

    // println!("{}", std::any::type_name::<BabyBear>());
    // println!("{}", std::any::type_name::<<BabyBear as Field>::Packing>());
    // println!("{}", std::any::type_name::<<<BabyBear as Field>::Packing as PackedValue>::Value>());
    // println!("Size of value: {} bytes", std::mem::size_of::<<<BabyBear as Field>::Packing as PackedValue>::Value>());
    // println!("Width: {}", <<BabyBear as Field>::Packing as PackedValue>::WIDTH);

    // let h_hybrid: SimpleHybridHasher<HPoseidon, HBlake3, BabyBear, BabyBear, u8, 8, 32> =
    //     SimpleHybridHasher::new::<<BabyBear as Field>::Packing, <BabyBear as Field>::Packing>(
    //         h_poseidon, h_blake3,
    //     );

    // let c_hybrid = SimpleHybridCompressor::<_, _, <BabyBear as Field>::Packing, <BabyBear as Field>::Packing::new(c_poseidon, c_blake3);
    // let c_hybrid = SimpleHybridCompressor::new(c_poseidon, c_blake3);

    // TODO create a generic vector of 10 BabyBear elements
    let dat1 = vec![rand::random::<BabyBear>(); 10];
    let dat2 = vec![rand::random::<BabyBear>(); 10];

    println!("ELEMENTS:\n{:?}\n{:?}", dat1, dat2);

    // let dig1 = h_hybrid.hash_iter(dat1);
    // let dig2 = h_hybrid.hash_iter(dat2);

    // c_hybrid.compress([[0; 16]; 2], &[8, 8], 16);

    // bench_merkle_tree::<<F as Field>::Packing, <F as Field>::Packing, H, C, 8>(criterion, h, c);
    // bench_merkle_tree<P, PW, H, C, const DIGEST_ELEMS: usize>

    println!("Hello, hybrid world!");
}
