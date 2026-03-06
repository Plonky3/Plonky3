//! Poseidon permutation for Goldilocks.
//!
//! # Overview
//!
//! This module provides the Poseidon1 hash permutation instantiated for the
//! Goldilocks field (p = 2^64 - 2^32 + 1). The public API is a single type
//! alias that transparently dispatches to the best available implementation.
//!
//! # Platform Dispatch
//!
//! On **aarch64**, the type alias resolves to a dual-dispatch wrapper:
//! scalar permutations delegate to the generic LLVM-optimized path
//! (avoiding regression from sequential inline ASM), while packed NEON
//! permutations delegate to the fused dual-lane ASM path.
//!
//! On **all other platforms**, it resolves to the generic Poseidon
//! implementation with Karatsuba MDS convolution.
//!
//! No `#[cfg]` is needed in calling code.
//!
//! # MDS Matrix
//!
//! The MDS matrix is a **circulant** matrix sourced from the MDS crate.
//! At runtime, it is applied via fast Karatsuba convolution (sub-O(t^2)).
//! During initialization only, it is expanded to dense form for the
//! sparse matrix decomposition of partial rounds.
//!
//! # Round Constants
//!
//! Sourced from the HorizenLabs reference implementation:
//! <https://github.com/HorizenLabs/poseidon2/blob/main/plain_implementations/src/poseidon/poseidon_instance_goldilocks.rs>

use p3_poseidon::{
    Poseidon, PoseidonConstants, PoseidonExternalLayerGeneric, PoseidonInternalLayerGeneric,
};

use crate::mds::{MATRIX_CIRC_MDS_8_COL, MATRIX_CIRC_MDS_12_COL};
use crate::{Goldilocks, MdsMatrixGoldilocks};

/// S-box degree for Goldilocks Poseidon.
///
/// The S-box raises each element to this power. The Goldilocks prime
/// factors as p - 1 = 2^32 * 3 * 5 * 17 * 257 * 65537. Neither 3 nor 5
/// are coprime to p - 1, so the smallest valid exponent is 7.
const GOLDILOCKS_S_BOX_DEGREE: u64 = 7;

/// Generic (non-fused) Poseidon permutation for Goldilocks.
///
/// Uses the platform-independent Poseidon implementation with Karatsuba
/// MDS convolution. Used directly for widths not supported by the fused
/// type (e.g. 16, 24) and as the non-aarch64 fallback for widths 8 and 12.
pub type PoseidonGoldilocksGeneric<const WIDTH: usize> = Poseidon<
    Goldilocks,
    PoseidonExternalLayerGeneric<Goldilocks, MdsMatrixGoldilocks, WIDTH>,
    PoseidonInternalLayerGeneric<Goldilocks, WIDTH>,
    WIDTH,
    GOLDILOCKS_S_BOX_DEGREE,
>;

/// Unified Poseidon permutation for Goldilocks.
///
/// On aarch64, resolves to a dual-dispatch wrapper: scalar permutations
/// use the generic LLVM-optimized path, packed NEON permutations use the
/// fused dual-lane ASM path.
///
/// On all other platforms, resolves to the generic implementation with
/// Karatsuba MDS convolution.
///
/// Supports both scalar and packed state representations transparently.
#[cfg(target_arch = "aarch64")]
pub type PoseidonGoldilocks<const WIDTH: usize> = crate::Poseidon1GoldilocksDispatch<WIDTH>;

/// Unified Poseidon permutation for Goldilocks.
///
/// On aarch64, resolves to the fused ASM-optimized implementation that
/// uses inline assembly and dual-lane NEON processing.
///
/// On all other platforms, resolves to the generic implementation with
/// Karatsuba MDS convolution.
///
/// Supports both scalar and packed state representations transparently.
#[cfg(not(target_arch = "aarch64"))]
pub type PoseidonGoldilocks<const WIDTH: usize> = PoseidonGoldilocksGeneric<WIDTH>;

/// Round constants for the width-8 Poseidon permutation.
///
/// 30 rows of 8 elements each, laid out as:
/// - 4 initial full rounds.
/// - 22 partial rounds.
/// - 4 terminal full rounds.
pub const GOLDILOCKS_POSEIDON_RC_8: [[Goldilocks; 8]; 30] = Goldilocks::new_2d_array([
    // Initial full rounds (4)
    [
        0x57056152cedf0fe7,
        0x44b125d16e93ca85,
        0x8e8ea2ff8b7a6d2a,
        0xcce7c6cc1468fa13,
        0x47f5feb953ce5073,
        0xfd8f41d8ee6b700e,
        0xe40f59b8db57aeb7,
        0x78b572234ff68244,
    ],
    [
        0x926b547a9712ed0b,
        0xb1525da069ba226c,
        0xf37650e9d8ef46d3,
        0x3146518c7738aefc,
        0x04aa9f4d916e9e5b,
        0xde603b81bb63d21c,
        0x8382c29e88cf2c81,
        0x50456f59f404cb88,
    ],
    [
        0x44bda4a6711f6ddb,
        0xe4c94cbc9e7d15b7,
        0x7faec52ce37a8256,
        0x7748e71fd7803107,
        0x9b6baf83e49be593,
        0xd47fe8a5c8b27ed3,
        0xfcdf1e28d16392ad,
        0x976753b4b516a9ee,
    ],
    [
        0xc16ea705aa7ee467,
        0x18183d87f912ebbb,
        0x02d3b175b21777fe,
        0x98e4c2d93e0aaaef,
        0xc31191d90cd41c96,
        0x69f8f94595ad453e,
        0x1de4127f3e248a2d,
        0xbcce9849c99a069c,
    ],
    // Partial rounds (22)
    [
        0x8b8e707932590779,
        0x4d7fff707c77890f,
        0x7d36116962851777,
        0x1dc9f40fbb3146b7,
        0x6a235e2d5bef54e0,
        0x4d1a9ae6dd337207,
        0x46ab49a6009cda1a,
        0x78e759e819648587,
    ],
    [
        0xee6e84b7763598a4,
        0x0b426bdcaad3050e,
        0x1f3cd981be91490e,
        0xd54572f7ecf947a1,
        0x393c4432d0e86a1e,
        0x3f1b43149ef3f4f8,
        0x3705f6a66d25dce4,
        0x3e809302b3d41471,
    ],
    [
        0x6e50830e082b17f1,
        0x711232bf2d77ac38,
        0x4235f7d079c78096,
        0xab1bbdc696a72a25,
        0xdb1ef6f3f7fed243,
        0xd21981014e77d809,
        0x5b2cb2bd03a18856,
        0x8e45a3e4bf30df6c,
    ],
    [
        0x3f9948080379716d,
        0x41c2ba50c09d6c70,
        0x5c2f57c6f81d2c6b,
        0x91cfb3d3b4b04a7a,
        0x81327090650355f6,
        0x06957eabf4817942,
        0x7f08201e9da0e064,
        0x7467dfc268e1d6e0,
    ],
    [
        0x38a9992ed589cc80,
        0x266a6e035fee9286,
        0xd19ebfbf75ffbf79,
        0x9f1dc0303ca0acfb,
        0x230f2d6a36b23347,
        0xde0cdaab08319a52,
        0xff9e2984d5f675ba,
        0x27a10c5aca2fcf50,
    ],
    [
        0x8982ec2da08deb87,
        0x89f9b8d33e98a684,
        0x269bcee2edb77b24,
        0xcd7fb3f592ab464f,
        0x05060bc8d4341e72,
        0xa75ab333263a6658,
        0x3962fe1b4bb486e7,
        0x52160689b78a2fd1,
    ],
    [
        0x9e953026b7be93e6,
        0x7215465ca2fa2b5a,
        0x458b8385c2107d5b,
        0xd86fd0264024aad9,
        0x2cb61942ee72b44c,
        0x50784c715273f7e7,
        0x5fdedb33fa9f3a87,
        0x6a4697bec73efb10,
    ],
    [
        0xb47744b651d0a93e,
        0x9b133ce9b34f9e24,
        0xb2af63941bc5c8d1,
        0xa7377cdf898e11ee,
        0x022a6e8af3f38e95,
        0xd4b6b57ec3cc0a8d,
        0xd73929bdd8d1b66b,
        0x81eb6a301c25206c,
    ],
    [
        0x0c7f9ff80801a8ed,
        0x7a26ac369d2b6d42,
        0xe0b8317c071c45d8,
        0xfede923925964753,
        0xa97836d6da89a463,
        0xa5ee4da013de472b,
        0xe677204adbfd65bf,
        0x3a22524d07758c9f,
    ],
    [
        0xd97c24115d694727,
        0x253300c8bcb8a257,
        0x1353c0984c181a15,
        0xebe483bff324731c,
        0x1511ed8fb6844846,
        0x53b461511c6ba3a1,
        0x21fdeb8122efac83,
        0xbb0db2349bc191d4,
    ],
    [
        0x6b4ab5a6fa05727a,
        0xd0dff6b2b7431743,
        0xd52b20f2e2546daa,
        0xfd44d15c2598333a,
        0xe93aa689ca1f82b1,
        0xfcf18baf99617666,
        0x05e145ac14f62606,
        0x0ffd185d90368f67,
    ],
    [
        0x692477023da0015b,
        0x3aacae49256f4c3a,
        0x5918382485b68fe7,
        0x3c603fa51f5ddcf3,
        0x51bd24c9676f0c84,
        0xbd46645f6871643c,
        0xa70bae909556881b,
        0x0973cdf1fd534f39,
    ],
    [
        0x23f5dc5ef40c635f,
        0xd454ee01ff625a5f,
        0x649bd02e30734e2d,
        0xafe63b1172c383c8,
        0xda8a6d4a224ea3b9,
        0x938e7e814aac0b6f,
        0xd4e3a299d450f03e,
        0x98e542e75908c80c,
    ],
    [
        0x7c172f4ac0637345,
        0x068bbf5eea717d21,
        0x17aae532ad95953a,
        0x079fee6318fdacf4,
        0x132c35dc67906d22,
        0x4be60392c1b57a95,
        0xec51e19a49dc4a4e,
        0xb8ac48db7c41af33,
    ],
    [
        0x71ad50f9a91de51b,
        0x75a74d0ea941a6a8,
        0x72a6eb718b6babe7,
        0x06bec5a7937f7aa4,
        0x8113e4862f68345d,
        0xa72c44ea227ee9c7,
        0x42210a1f7c406b32,
        0xc0760b6236faf43c,
    ],
    [
        0x26d1410a43365117,
        0xf8ceb9f950d53940,
        0xf628fd2af2b6e5bb,
        0xc057934a34308393,
        0xf3cc773469d56186,
        0x065237420ff7c2bb,
        0x5453eb7e3bc08a16,
        0x9aff5a4ee199aadc,
    ],
    [
        0x64e829d61e5fa797,
        0x7c9588d6c404a8b1,
        0x15a0d6c9d7ef0aec,
        0x7dd3bdf746a39db2,
        0x84aede73cd8bb3af,
        0x606fc5ceb97a60c6,
        0xcf3c5f62ef49f02f,
        0xdfef8f0f1e87bc85,
    ],
    [
        0x48639fc9447f5fd7,
        0x504b752b242a66b4,
        0xadf53e4f4f743697,
        0x845bedb45f513c52,
        0x91e707477184e4d1,
        0xd6e5ca76b7f56559,
        0x755a6089bf66dd30,
        0xe602a9a86fecfd8b,
    ],
    [
        0x3ff57e8d1c2d1687,
        0x6f43f7917b1c3394,
        0x533fb5ae21bf2d04,
        0x444dc4db45ebb4c0,
        0x2e471c803b72d085,
        0x2b53f2c7f50ba37a,
        0xa6c07f0bfbe85015,
        0x54a214f32cc02d51,
    ],
    [
        0x7b27404b26406739,
        0xfc4bef705a7f7dc9,
        0xd6dd660d6a9d30a6,
        0x40a59c90c7ebe34c,
        0x33a78eca623d7bbc,
        0xa7c7daa3adb60427,
        0xca4bfceeca59dcad,
        0x8f7c56e3381710a4,
    ],
    [
        0x569ca673f517e4d1,
        0x55970d6132c40df6,
        0x454236654b451433,
        0xfe684e4a876de175,
        0x580a016a208b1f39,
        0xd7e9971806ea0c8b,
        0xe38d7016c93b40ff,
        0x62185e718394a29b,
    ],
    [
        0xf810fb01ebb26cea,
        0x21ddc29682b46878,
        0xf8ab79b18c09dd5c,
        0x76bd4cfc027aaad6,
        0x622f038a4b5cd454,
        0x77f63fe804886afa,
        0xed6c954619819ed3,
        0x64d46e1a53bd56bf,
    ],
    // Terminal full rounds (4)
    [
        0x1851bad534d7b9d0,
        0xd0a80f4993519712,
        0x867049a072afe789,
        0x2ccffd104db57b35,
        0x6e1170f9d4efa571,
        0xa61eb4ab448fb4c7,
        0x7175bbae1d097494,
        0x90fb4923de14b485,
    ],
    [
        0xbe9a25abdc7c0b56,
        0xc27e243f8582657a,
        0x6bbdb9dd602700fa,
        0x5b546e6a5c2337df,
        0x73f53b99a4056a62,
        0xe85369b678eab9dc,
        0xee4ddbd911c67cfd,
        0x3666de148390a60c,
    ],
    [
        0xc305f73f8e7ed55e,
        0x98f6589bb92766fe,
        0xed68a166b50e3cd5,
        0x7bba85aee4a42e8d,
        0xeba72e21d22d3ba2,
        0x2df46cf41c763e5a,
        0x79ba0ca148458c41,
        0x97a5abcddbaf3e6b,
    ],
    [
        0x22f193a7fcc2dfc1,
        0x48fc93e221d3c3b3,
        0xb1100bbb875c4e32,
        0x1384efcf6056a457,
        0xd2a77ce00115fd04,
        0x523f48a919d6bfa1,
        0xb3e11e53c3dd625a,
        0x5113945a42f34dfa,
    ],
]);

/// Round constants for the width-12 Poseidon permutation.
///
/// 30 rows of 12 elements each, laid out as:
/// - 4 initial full rounds.
/// - 22 partial rounds.
/// - 4 terminal full rounds.
pub const GOLDILOCKS_POSEIDON_RC_12: [[Goldilocks; 12]; 30] = Goldilocks::new_2d_array([
    // Initial full rounds (4)
    [
        0xe034a8785fd284a7,
        0xe2463f1ea42e1b80,
        0x048742e681ae290a,
        0xe4af50ade990154c,
        0x8b13ffaaf4f78f8a,
        0xe3fbead7dccd8d63,
        0x631a47705eb92bf8,
        0x88fbbb8698548659,
        0x74cd2003b0f349c9,
        0xe16a3df6764a3f5d,
        0x57ce63971a71aaa2,
        0xdc1f7fd3e7823051,
    ],
    [
        0xbb8423be34c18d7a,
        0xf8bc5a2a0c1b3d6d,
        0xf1a01bbd6f7123e5,
        0xed960a080f5e348b,
        0x1b9c0c1e87e2390e,
        0x18c83caf729a613e,
        0x671ab9fe037a72c4,
        0x508565f67d4c276a,
        0x4d2cd8827a482590,
        0xa48e11e84dd3500b,
        0x825a8c955fc2442b,
        0xf573a6ee07cddc68,
    ],
    [
        0x7dd3f19c73a39e0b,
        0xcc0f13537a796fa6,
        0x1d9006bfaedac57f,
        0x4705f69b68b0b7de,
        0x5b62bfb718bcc57f,
        0x879d821770563827,
        0x3da5ccb7f8dff0e3,
        0xb49d6a706923fc5b,
        0xb6a0babe883a969d,
        0x2984f9b055401960,
        0xcd3496f05511d79d,
        0x4791da5d63854fc5,
    ],
    [
        0xdb7344d0580a39d4,
        0x5aedc4dad1de120a,
        0x5e1bdc1fb8e1abf0,
        0x3904c09a0e46747c,
        0xb54a0e23ab85ddcd,
        0xc0c3cf05bccbdb3a,
        0xb362076a73baf7e9,
        0x212c953d81a5d5ba,
        0x212d4cc965d898bd,
        0xdd44ddd0f41509b9,
        0x8931329fa67823c0,
        0xc65510f4d2a873be,
    ],
    // Partial rounds (22)
    [
        0xe3ecbb6ba1e16211,
        0x70f5b3266792bbb6,
        0xe7560e690634757e,
        0xafd0202bc7eaf66e,
        0x349f4c5871f220fd,
        0x3697eb3e31529e0d,
        0x7735d5b0622d9900,
        0x5f5b58b9cf997668,
        0x645534b6548af9d9,
        0x4232d29d91a426a8,
        0xb987278aed485d35,
        0x6dabeef669bb406e,
    ],
    [
        0x35ee78288b749d40,
        0x6dcd560f14af0fc3,
        0x71ed3dc007ea6383,
        0x8b6b51caab7f5b6f,
        0xcf2e8cc4181dbfa8,
        0xa01d3f1c306f825a,
        0xccee646a5d8ddb87,
        0x70df6f277cbaffeb,
        0x64ec0a6556b8f45c,
        0x6f68c9664fda6e37,
        0x387356e4516fab6f,
        0x35310dce33903e67,
    ],
    [
        0x45f3e5251d30f912,
        0x7c97f480ca428f45,
        0x74d5874c20b50de2,
        0xff1d5b7cee3dc67f,
        0xa04d5d5ac0ff3de9,
        0x1cefb5eb7d24580e,
        0xf685e1bfcc0104ad,
        0x6204dd95db22ead4,
        0x8265c6c57c73c440,
        0x4f708ab0b4e1e382,
        0xcfc60c7a52fbffa7,
        0x9c0c1951d8910306,
    ],
    [
        0x4d06df27c89819f2,
        0x621bdb0e75eca660,
        0x343adffd079cee57,
        0xa760f0e5debde398,
        0xe3110fefd97b188a,
        0x0ed6584e6b150297,
        0x2b10e625d0d079c0,
        0xefa493442057264f,
        0xebcfaa7b3f26a2b6,
        0xf36bcda28e343e2a,
        0xa1183cb63b67aa9e,
        0x40f3e415d5e5b0ba,
    ],
    [
        0xc51fc2367eff7b15,
        0xe07fe5f3aebc649f,
        0xc9cb2be56968e8aa,
        0x648600db69078a0e,
        0x4e9135ab1256edb9,
        0x00382c73435556c2,
        0x1d78cafac9150ddf,
        0xb8df60ab6215a233,
        0xa7a65ba31f8fcd9a,
        0x907d436dd964006b,
        0x3bdf7fd528633b97,
        0x265adb359c0cc0f8,
    ],
    [
        0xf16cfc4034b39614,
        0x71f0751b08fa0947,
        0x3165eda4b5403a37,
        0xca30fc5680467e46,
        0x4c743354d37777c5,
        0x3d1f0a4e6bba4a09,
        0xc0c2e289afa75181,
        0x1e4fa2ad948978b7,
        0x2a226a127a0bb26a,
        0xe61738a70357ce76,
        0x965f66eada2905e4,
        0xe2560d17d78f997b,
    ],
    [
        0x491135cfb0b193c4,
        0x88a7d811ba8c8172,
        0x7d41dd29039a1493,
        0x0cd9e984523d5ef9,
        0xe758a718011f225e,
        0xa1b19355c645c206,
        0x2b3f098c8ba43f7e,
        0xa79ec72dff37d575,
        0x7f64bb4c1a7ba443,
        0x0e9b008eda4002d2,
        0xcf6f77ac16722afa,
        0x3fd4c0ca0672aebd,
    ],
    [
        0x9b72bf1c1c3d08a9,
        0xe4940f84b71e4ac3,
        0x61b27b077118bc73,
        0x2efd8379b8e661e3,
        0x858edcf353df0342,
        0x2d9c20affb5c4517,
        0x5120143f0695defc,
        0x62fc898ae34a5c5c,
        0xa3d9560c99123ed3,
        0x98fd739d8e7fc934,
        0x49c0bad1b2023adf,
        0x2bc9cf2aec60ebf6,
    ],
    [
        0x43b995c4ef12dcc6,
        0x3c69a9d2d4555790,
        0x43fc8b0b247132c0,
        0x217b0f6f3b52feff,
        0xa10cebbb66f1f5b3,
        0xcc81ed1130a2c36e,
        0x63da22539da7b97e,
        0x8bf756d728bf5553,
        0x373ce92bcc4dfab8,
        0x2f1720d02fb0b850,
        0xa04ad342d9e5071b,
        0x4a2758463a4bc975,
    ],
    [
        0x3d95dc5c0aae0025,
        0xadaca762a70a4139,
        0x19eafc0ac322234c,
        0x90f7d010345e3191,
        0x8de683caae7e23c6,
        0xc8c4de0badedd6a5,
        0x2982932ceb559a26,
        0xecd9864a9b3046f5,
        0xfe26e58fc0fdfc38,
        0x34d8ccf408b18b11,
        0x305263531bf413e5,
        0x9d740626025dc1e2,
    ],
    [
        0x9bb62131c8873ac5,
        0x39a2e90839d2b1b5,
        0x79b976bd771e389a,
        0xdd518ffeb5b209d6,
        0xcc0ae5430247d957,
        0x69d45a7c309cece9,
        0xcab8f84a11dc89c2,
        0xa2f298798ac57fd0,
        0x31a1581972494aab,
        0x3d7247a8bad73c1b,
        0xcfd1cf2abfdf8e9d,
        0xd2b3f55a53c0f36e,
    ],
    [
        0xc1b5c91f9528675f,
        0xaea262169d3ce0d6,
        0xcf744aeea9a85186,
        0xb14caf9c8e665072,
        0x9036fdf0f750c7d7,
        0xd50ed0260936689a,
        0x67d5900d08a564a7,
        0x02ff541f79857b26,
        0xc83ad8ef65a5dd34,
        0xcd9087f66c8913a6,
        0xc81bcc8f049cec68,
        0xd04bb2943fdb30ad,
    ],
    [
        0x16e3179444fdc702,
        0x16b1d70d695e019e,
        0x4ff845a458f34230,
        0x766e3f44b85c9dbe,
        0x7c4c3acf8514557c,
        0x57adabf739fd4da0,
        0xc66f77381d247cbc,
        0x6e2a4a99a74cfc14,
        0xea8d1862edc8f863,
        0xc6db4cd97dc665da,
        0x949056b5e892bf9a,
        0x462fe0ca6f15ab99,
    ],
    [
        0x9adb62723963213e,
        0xe6727c5fd42965fc,
        0x7fbdd9a508ac0f2d,
        0x81fe716d0f7abc16,
        0x9576f06f87da27da,
        0xfc365eab4c817bc4,
        0xfd9ef2ac09e90378,
        0xd617a6538caecd71,
        0x65b8045fbc97224e,
        0xa9d715b7578ad6db,
        0xc2dea08b2f8e0fec,
        0xb2a75f374961bd18,
    ],
    [
        0xbd384569c776ea85,
        0x3830b682c3aaaf39,
        0xae0fd86a8479f28a,
        0x3af2201bcba3c6c8,
        0xc62b22ab3d6edcef,
        0xd82a8399ca086539,
        0x9a8a1adb11b997e1,
        0x62e9c6079f0f4489,
        0x5b42e26cad54c3eb,
        0x0d23026116e75052,
        0x117b3df6a1bcabd3,
        0xdc15f849a793f4ff,
    ],
    [
        0xcb55e856cff1a9fc,
        0x4f276ed50185804c,
        0x9a2a1c18334e8eee,
        0x942eae69c7b2ac12,
        0x372123fca5367880,
        0x7299f05b81f6ab8c,
        0xc4b6d222335c0d40,
        0x54e175dc9898de82,
        0x9a59ed2ff8185bb6,
        0xe31cffc4a4d1595c,
        0x953ea6a8cf91eb62,
        0xd73df50b58e3de87,
    ],
    [
        0xcba164ac5529a437,
        0xed86532f13f5a01b,
        0x48638620ea9a8cdf,
        0xbf39c6d292e61897,
        0xcbeafcace75ae54b,
        0x9b1c4ba273aef896,
        0x34a3fa4e9ea8b222,
        0x388497890444f9e4,
        0x1ceeb6d09ae44039,
        0xc934ed066fc000bd,
        0x5420b49b40809695,
        0x227bb866b6e43b27,
    ],
    [
        0x023cce4d47323bd5,
        0x9bdd445cfb266aa9,
        0xba558b69d5e89ac6,
        0x45e50280e3d7c220,
        0xc7b336bf7db5785e,
        0x17c3a2296aa7cabc,
        0xe7a055c8663e8ece,
        0x7014aeac12a9562a,
        0xf1a5396bf65b5aa2,
        0x4e6642abd7507fb0,
        0x630e0222d5393a15,
        0x173af02aa4f69206,
    ],
    [
        0xaf5c39865a5eb017,
        0x4ae10acdc3c41602,
        0x0058046e6d9df692,
        0xa44bff2bee5f1073,
        0x944a687060c16827,
        0xc43390133b0d7316,
        0xd41f77f6bad6185a,
        0x9af59b9c3c1d1cfc,
        0x3e36dd171e4a675d,
        0xc7ba958b07eb9943,
        0xc4b47b8808de11c0,
        0x001ef692416bc9cd,
    ],
    [
        0xa2fb05ec5711129e,
        0x3a2f12f04b368596,
        0x44fd1b36bd05ded4,
        0xc132a1c940e7efac,
        0x0990f606eef60c22,
        0xed8ec2a68338a212,
        0xbfb310dd70919411,
        0xadf2a5d0d908ac0d,
        0x6fd5fa590b36a39b,
        0xa264d5481bcadb21,
        0x2c0452d2bc532534,
        0xfa6641a6cf17cc0b,
    ],
    [
        0x6ada4c9390f0eafa,
        0x152c1b3439da0ac2,
        0x615013a63de9adb0,
        0xcd17255ec2e4cb01,
        0xba7715fb4a4fadfa,
        0xa0ca4b6d43eebdf4,
        0xf290b6fed5af6f62,
        0xdc0fe55c9e65aa26,
        0xbd600ace449304c2,
        0x0e53a360f26da9b5,
        0x78605b519f96abe6,
        0xea7e408734243799,
    ],
    // Terminal full rounds (4)
    [
        0xf53ac0707eb51726,
        0x0336f478469cef21,
        0xbc6af9b810b9f89c,
        0xb722092616785496,
        0xbca7b0ca58c04422,
        0xf7870109a513441b,
        0x8c71931c2de63eb8,
        0xf79815be37e5ce04,
        0xeee4e1205eab3d44,
        0x52a23d6299839b6c,
        0x9fc5362010ac1103,
        0x9690d2f4abc80294,
    ],
    [
        0x0481281fb649ab93,
        0xf2cd1f90ecede2fc,
        0x301c378877734c25,
        0xd20b8a3b7d6679d3,
        0x33a8b5db96979da3,
        0x13034e5c7269d9ef,
        0xcfbce2ab85636d0e,
        0x3f37a4c42edf4a97,
        0xb63fd6ec7ff50302,
        0x436b1e86ddc7362a,
        0x6a54ae4b5c97b739,
        0xf3bd6dd9365f3915,
    ],
    [
        0x4181aca49c9b271e,
        0x49d4d7643da6aafb,
        0x2036bf0f76786aa0,
        0x7a42c4d2c7ae05b7,
        0xd5ace3058744be86,
        0x181a59418ab1c592,
        0x77b67d60a5a07b36,
        0x1e7cd334ecbf8178,
        0x4c6e85d690a6141e,
        0xf4b6a9f1be304bc9,
        0x60d3f578fb9c343a,
        0xdaac75db3c11fc58,
    ],
    [
        0x0864ae3ba35af1b9,
        0xb6bc40765fb2570b,
        0xd46b53cbe6a6f811,
        0x8429e09e6ac7d398,
        0x1ffb73140f60b153,
        0x803d688fe62a93f8,
        0xb41e9f0d9c051046,
        0x098746d28211cd65,
        0xe919f936e43f4b3b,
        0x690416052f3471f3,
        0x656d94333b449fc7,
        0xf2b8a970984acf87,
    ],
    [
        0x8a3f96f9ca67752e,
        0xce1efcd7a468c992,
        0x5a4f3f1df0662069,
        0xbece1eb8967e9e42,
        0x872f99ff7891f554,
        0xfd7cb913022fb888,
        0xa4d257eb39902d5d,
        0x67c99dfda1508416,
        0x50cda61566da959e,
        0xec462bbe31e2c852,
        0x2569ce8db808f43c,
        0x679a9dfdcf0fccb4,
    ],
]);

/// Create the default width-8 Poseidon permutation for Goldilocks.
///
/// Returns the platform-optimal implementation: dual-dispatch on aarch64
/// (generic for scalar, fused ASM for packed), generic Karatsuba on all
/// other platforms.
#[cfg(target_arch = "aarch64")]
pub fn default_goldilocks_poseidon_8() -> PoseidonGoldilocks<8> {
    let constants = PoseidonConstants {
        rounds_f: 8,
        rounds_p: 22,
        mds_circ_col: MATRIX_CIRC_MDS_8_COL,
        round_constants: GOLDILOCKS_POSEIDON_RC_8.to_vec(),
    };
    let generic = Poseidon::new(&constants);
    let (full, partial) = constants.to_optimized();
    let fused = crate::Poseidon1GoldilocksFused::new(full.clone(), partial.clone());
    crate::Poseidon1GoldilocksDispatch::new(generic, fused, full, partial)
}

/// Create the default width-8 Poseidon permutation for Goldilocks.
///
/// Returns the platform-optimal implementation: fused ASM on aarch64,
/// generic Karatsuba on all other platforms.
#[cfg(not(target_arch = "aarch64"))]
pub fn default_goldilocks_poseidon_8() -> PoseidonGoldilocks<8> {
    Poseidon::new(&PoseidonConstants {
        rounds_f: 8,
        rounds_p: 22,
        mds_circ_col: MATRIX_CIRC_MDS_8_COL,
        round_constants: GOLDILOCKS_POSEIDON_RC_8.to_vec(),
    })
}

/// Create the default width-12 Poseidon permutation for Goldilocks.
///
/// Returns the platform-optimal implementation: dual-dispatch on aarch64
/// (generic for scalar, fused ASM for packed), generic Karatsuba on all
/// other platforms.
#[cfg(target_arch = "aarch64")]
pub fn default_goldilocks_poseidon_12() -> PoseidonGoldilocks<12> {
    let constants = PoseidonConstants {
        rounds_f: 8,
        rounds_p: 22,
        mds_circ_col: MATRIX_CIRC_MDS_12_COL,
        round_constants: GOLDILOCKS_POSEIDON_RC_12.to_vec(),
    };
    let generic = Poseidon::new(&constants);
    let (full, partial) = constants.to_optimized();
    let fused = crate::Poseidon1GoldilocksFused::new(full.clone(), partial.clone());
    crate::Poseidon1GoldilocksDispatch::new(generic, fused, full, partial)
}

/// Create the default width-12 Poseidon permutation for Goldilocks.
///
/// Returns the platform-optimal implementation: fused ASM on aarch64,
/// generic Karatsuba on all other platforms.
#[cfg(not(target_arch = "aarch64"))]
pub fn default_goldilocks_poseidon_12() -> PoseidonGoldilocks<12> {
    Poseidon::new(&PoseidonConstants {
        rounds_f: 8,
        rounds_p: 22,
        mds_circ_col: MATRIX_CIRC_MDS_12_COL,
        round_constants: GOLDILOCKS_POSEIDON_RC_12.to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use p3_symmetric::Permutation;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;

    type F = Goldilocks;

    /// Known-answer test for width 8 (sequential 0..7 input).
    #[test]
    fn test_poseidon_goldilocks_width_8() {
        let perm = default_goldilocks_poseidon_8();

        let mut input: [F; 8] = F::new_array([0, 1, 2, 3, 4, 5, 6, 7]);
        perm.permute_mut(&mut input);

        let expected: [F; 8] = F::new_array([
            9402631278636174262,
            17004635335047130969,
            4753012512511375168,
            2367239025173374637,
            16921623581418299817,
            2583582247381514966,
            12095377248421862739,
            5553966757107483887,
        ]);
        assert_eq!(input, expected);
    }

    /// Known-answer test for width 12 (sequential 0..11 input).
    #[test]
    fn test_poseidon_goldilocks_width_12() {
        let perm = default_goldilocks_poseidon_12();

        let mut input: [F; 12] = F::new_array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
        perm.permute_mut(&mut input);

        let expected: [F; 12] = F::new_array([
            71109932875369129,
            6358490863509936162,
            15845417984153754708,
            6622548371661487295,
            16665750330794722584,
            3390336522757414137,
            9832650793018174136,
            5390692944999521363,
            15168680663824027226,
            4054910678692513992,
            14678252141200722212,
            3716442817880027191,
        ]);
        assert_eq!(input, expected);
    }

    /// Smoke test for width 16 with random constants.
    /// Uses the generic type directly since the fused type only supports 8 and 12.
    #[test]
    fn test_poseidon_goldilocks_width_16() {
        let mut rng = SmallRng::seed_from_u64(1);
        let poseidon =
            PoseidonGoldilocksGeneric::<16>::new_from_rng(4, 22, &MdsMatrixGoldilocks, &mut rng);
        let input: [F; 16] = rand::RngExt::random(&mut rng);
        let output = poseidon.permute(input);
        assert_ne!(output, input);
    }

    /// Smoke test for width 24 with random constants.
    #[test]
    fn test_poseidon_goldilocks_width_24() {
        let mut rng = SmallRng::seed_from_u64(1);
        let poseidon =
            PoseidonGoldilocksGeneric::<24>::new_from_rng(4, 22, &MdsMatrixGoldilocks, &mut rng);
        let input: [F; 24] = rand::RngExt::random(&mut rng);
        let output = poseidon.permute(input);
        assert_ne!(output, input);
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    mod avx512 {
        use super::*;
        use crate::PackedGoldilocksAVX512;

        #[test]
        fn test_avx512_poseidon_width_16() {
            let mut rng = SmallRng::seed_from_u64(1);
            let poseidon = PoseidonGoldilocksGeneric::<16>::new_from_rng(
                4,
                22,
                &MdsMatrixGoldilocks,
                &mut rng,
            );
            let input: [F; 16] = rand::RngExt::random(&mut rng);

            let mut expected = input;
            poseidon.permute_mut(&mut expected);

            let mut avx512_input = input.map(Into::<PackedGoldilocksAVX512>::into);
            poseidon.permute_mut(&mut avx512_input);

            let avx512_output = avx512_input.map(|x| x.0[0]);
            assert_eq!(avx512_output, expected);
        }

        #[test]
        fn test_avx512_poseidon_width_24() {
            let mut rng = SmallRng::seed_from_u64(1);
            let poseidon = PoseidonGoldilocksGeneric::<24>::new_from_rng(
                4,
                22,
                &MdsMatrixGoldilocks,
                &mut rng,
            );
            let input: [F; 24] = rand::RngExt::random(&mut rng);

            let mut expected = input;
            poseidon.permute_mut(&mut expected);

            let mut avx512_input = input.map(Into::<PackedGoldilocksAVX512>::into);
            poseidon.permute_mut(&mut avx512_input);

            let avx512_output = avx512_input.map(|x| x.0[0]);
            assert_eq!(avx512_output, expected);
        }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        not(target_feature = "avx512f")
    ))]
    mod avx2 {
        use super::*;
        use crate::PackedGoldilocksAVX2;

        #[test]
        fn test_avx2_poseidon_width_16() {
            let mut rng = SmallRng::seed_from_u64(1);
            let poseidon = PoseidonGoldilocksGeneric::<16>::new_from_rng(
                4,
                22,
                &MdsMatrixGoldilocks,
                &mut rng,
            );
            let input: [F; 16] = rand::RngExt::random(&mut rng);

            let mut expected = input;
            poseidon.permute_mut(&mut expected);

            let mut avx2_input = input.map(Into::<PackedGoldilocksAVX2>::into);
            poseidon.permute_mut(&mut avx2_input);

            let avx2_output = avx2_input.map(|x| x.0[0]);
            assert_eq!(avx2_output, expected);
        }

        #[test]
        fn test_avx2_poseidon_width_24() {
            let mut rng = SmallRng::seed_from_u64(1);
            let poseidon = PoseidonGoldilocksGeneric::<24>::new_from_rng(
                4,
                22,
                &MdsMatrixGoldilocks,
                &mut rng,
            );
            let input: [F; 24] = rand::RngExt::random(&mut rng);

            let mut expected = input;
            poseidon.permute_mut(&mut expected);

            let mut avx2_input = input.map(Into::<PackedGoldilocksAVX2>::into);
            poseidon.permute_mut(&mut avx2_input);

            let avx2_output = avx2_input.map(|x| x.0[0]);
            assert_eq!(avx2_output, expected);
        }
    }

    #[cfg(target_arch = "aarch64")]
    mod neon {
        use super::*;
        use crate::PackedGoldilocksNeon;

        #[test]
        fn test_neon_poseidon_width_8() {
            let perm = default_goldilocks_poseidon_8();
            let input: [F; 8] = F::new_array([0, 1, 2, 3, 4, 5, 6, 7]);

            let mut expected = input;
            perm.permute_mut(&mut expected);

            let mut neon_input = input.map(Into::<PackedGoldilocksNeon>::into);
            perm.permute_mut(&mut neon_input);

            let neon_output = neon_input.map(|x| x.0[0]);
            assert_eq!(neon_output, expected);
        }

        #[test]
        fn test_neon_poseidon_width_12() {
            let perm = default_goldilocks_poseidon_12();
            let input: [F; 12] = F::new_array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);

            let mut expected = input;
            perm.permute_mut(&mut expected);

            let mut neon_input = input.map(Into::<PackedGoldilocksNeon>::into);
            perm.permute_mut(&mut neon_input);

            let neon_output = neon_input.map(|x| x.0[0]);
            assert_eq!(neon_output, expected);
        }
    }
}
